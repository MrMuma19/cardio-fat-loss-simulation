"""
validation_external_v2_2.py  —  V2.2.1

External validation with outcome-type-aware comparison and explicit handling of
studies longer than the model horizon.

V2.2.1 fixes over V2.2:
- Rows with duration_weeks > MAX_MODEL_DURATION_WEEKS are predicted using a
  truncated 24-week model horizon instead of extrapolating to 45-69 weeks.
- Output includes model_duration_weeks and duration_handling.
- Extended-duration Donnelly comparisons are labelled as 24-week truncated
  predictions against longer observed interventions, not as full-duration
  external validation.

Run from the project root:
    python code_v2_2/validation_external_v2_2.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from simulation_v2_2 import (ModelConfig, HyperParameters, Scenario,
                               generate_population, simulate_scenario, summarize)

ROOT = Path(__file__).resolve().parents[1]
MAX_MODEL_DURATION_WEEKS = 24


def _load_dataset(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    for col in ['exercise_dose_min_week', 'duration_weeks',
                'delta_fm_observed', 'delta_bw_observed', 'observed_sd']:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('~', '', regex=False).str.strip(),
                errors='coerce')
    return df


def _prediction_duration(duration_weeks: int) -> tuple[int, str]:
    """Return the model duration and a human-readable handling label."""
    if duration_weeks > MAX_MODEL_DURATION_WEEKS:
        return MAX_MODEL_DURATION_WEEKS, f'truncated_to_{MAX_MODEL_DURATION_WEEKS}wk_model_horizon'
    return duration_weeks, 'within_model_horizon'


def _predict_for_row(row: pd.Series,
                     cfg: ModelConfig,
                     hp: HyperParameters,
                     comp_scenario: float = 0.25) -> dict:
    dose = int(row['exercise_dose_min_week']) if not pd.isna(row.get('exercise_dose_min_week')) else 0
    actual_duration = int(row['duration_weeks']) if not pd.isna(row.get('duration_weeks')) else 12
    model_duration, duration_handling = _prediction_duration(actual_duration)

    rng = np.random.default_rng(42)
    pop = generate_population(cfg, hp, rng)
    sc = Scenario(dose, model_duration, comp_scenario)
    ind = simulate_scenario(pop, sc, cfg, hp, rng)
    s = summarize(ind, sc, cfg)

    # Model predicts loss as positive; observed values are negative (loss convention).
    predicted_delta_fm = -s['mean_fm']
    predicted_delta_bw = -s['mean_fm'] / 0.82   # rough BW from FM, FM ≈ 82% of BW loss
    return {
        'model_duration_weeks': model_duration,
        'duration_handling': duration_handling,
        'predicted_mean_delta_fm': round(predicted_delta_fm, 4),
        'predicted_mean_delta_bw': round(predicted_delta_bw, 4),
    }


def run_validation(output_dir: Path | None = None,
                   val_dir: Path | None = None) -> pd.DataFrame | None:
    out = output_dir or ROOT / 'outputs_v2_2'
    val_dir = val_dir or ROOT / 'validation_v2_2'
    out.mkdir(parents=True, exist_ok=True)

    val_file = val_dir / 'external_validation_dataset_v2_2.csv'
    if not val_file.exists():
        print(f"ERROR: validation dataset not found at {val_file}")
        return None

    df = _load_dataset(val_file)
    if 'validation_role' in df.columns and 'role' not in df.columns:
        df = df.rename(columns={'validation_role': 'role'})
    cfg = ModelConfig()
    hp = HyperParameters()

    results = []
    for _, row in df.iterrows():
        role = str(row.get('role', '')).strip().lower()
        outcome_type = str(row.get('outcome_type', '')).strip().lower()

        pred = {}
        if not pd.isna(row.get('exercise_dose_min_week')):
            pred = _predict_for_row(row, cfg, hp)

        if outcome_type == 'delta_fm':
            obs_val = row.get('delta_fm_observed', np.nan)
            pred_val = pred.get('predicted_mean_delta_fm', np.nan)
            pred_type = 'predicted_mean_delta_fm'
        elif outcome_type == 'delta_bw':
            obs_val = row.get('delta_bw_observed', np.nan)
            pred_val = pred.get('predicted_mean_delta_bw', np.nan)
            pred_type = 'predicted_mean_delta_bw'
        else:
            obs_val = pred_val = pred_type = np.nan

        abs_err = (abs(float(pred_val) - float(obs_val))
                   if (not pd.isna(obs_val) and not pd.isna(pred_val))
                   else np.nan)

        results.append({
            'study': row.get('study', ''),
            'arm_label': row.get('arm_label', ''),
            'exercise_dose_min_week': row.get('exercise_dose_min_week', np.nan),
            'duration_weeks': row.get('duration_weeks', np.nan),  # retained for backward compatibility
            'duration_weeks_observed': row.get('duration_weeks', np.nan),
            'model_duration_weeks': pred.get('model_duration_weeks', np.nan),
            'duration_handling': pred.get('duration_handling', ''),
            'outcome_type': outcome_type,
            'delta_fm_observed': row.get('delta_fm_observed', np.nan),
            'delta_bw_observed': row.get('delta_bw_observed', np.nan),
            'predicted_mean_delta_fm': pred.get('predicted_mean_delta_fm', np.nan),
            'predicted_mean_delta_bw': pred.get('predicted_mean_delta_bw', np.nan),
            'observed_for_comparison': obs_val,
            'prediction_for_comparison': pred_val,
            'prediction_type_used': pred_type,
            'absolute_error_corrected': abs_err,
            'role': role,
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(out / 'external_validation_results_v2_2.csv', index=False)

    church_mask = (
        df_out['role'].isin(['validation_primary']) &
        df_out['observed_for_comparison'].notna() &
        df_out['prediction_for_comparison'].notna()
    )
    strride_mask = (
        df_out['role'].str.contains('sensitivity', na=False) &
        df_out['observed_for_comparison'].notna() &
        df_out['prediction_for_comparison'].notna()
    )
    donnelly_mask = (
        df_out['role'].str.contains('extended_duration', na=False) &
        df_out['observed_for_comparison'].notna() &
        df_out['prediction_for_comparison'].notna()
    )
    midwest_missing = df_out['study'].str.contains('Midwest', na=False).any() and \
                      df_out[df_out['study'].str.contains('Midwest', na=False)]['observed_for_comparison'].isna().all()

    lines = [
        "External Validation Metrics — V2.2.1",
        "=" * 45,
        "",
        "Model: compensation-adjusted probabilistic simulation with explicit",
        "component accounting (intake 50%, NEAT 30%, AT 20%).",
        "Note: NEAT and AT caps non-binding at studied doses — component accounting",
        "does not materially alter predictions relative to scalar compensation.",
        "",
        f"Primary model horizon for validation comparisons: {MAX_MODEL_DURATION_WEEKS} weeks.",
        "Rows longer than this horizon are compared against a truncated 24-week",
        "model prediction and are not counted as primary validation.",
        "",
    ]

    if midwest_missing:
        lines += [
            "WARNING: Midwest Exercise Trial data still missing (NA).",
            "Full external validation cannot be completed.",
            "Metrics below are PRELIMINARY and based on Church et al. only.",
            "",
        ]

    if church_mask.sum() > 0:
        sub = df_out[church_mask]
        obs = sub['observed_for_comparison'].values.astype(float)
        pred = sub['prediction_for_comparison'].values.astype(float)
        rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
        mae = float(np.mean(np.abs(pred - obs)))
        label = "preliminary_validation_church_only" if midwest_missing else "preliminary_primary_validation"
        lines += [
            f"[{label}]",
            f"  Source       : Church et al. 2009 (n={church_mask.sum()} arms)",
            f"  Outcome type : {sub['outcome_type'].mode()[0]}",
            f"  RMSE         : {rmse:.3f} kg",
            f"  MAE          : {mae:.3f} kg",
            f"  n data points: {church_mask.sum()}",
            "",
            "  NOTE: ICC and coverage not computed.",
            "  Rationale: ICC requires ≥10 independent validation estimates.",
            "  Coverage (% within 95% PI) not computed without sufficient independent data.",
            "",
        ]
    else:
        lines += ["No primary validation rows with complete data available.", ""]

    if strride_mask.sum() > 0:
        sub_s = df_out[strride_mask]
        obs_s = sub_s['observed_for_comparison'].values.astype(float)
        prd_s = sub_s['prediction_for_comparison'].values.astype(float)
        rmse_s = float(np.sqrt(np.mean((prd_s - obs_s) ** 2)))
        mae_s = float(np.mean(np.abs(prd_s - obs_s)))
        lines += [
            "[strride_sensitivity — NOT primary validation]",
            "  Source       : STRRIDE (Slentz 2004) — partial circularity (adherence parametrisation)",
            f"  RMSE         : {rmse_s:.3f} kg",
            f"  MAE          : {mae_s:.3f} kg",
            f"  n data points: {strride_mask.sum()}",
            "",
        ]

    if donnelly_mask.sum() > 0:
        sub_m = df_out[donnelly_mask]
        obs_m = sub_m['observed_for_comparison'].values.astype(float)
        prd_m = sub_m['prediction_for_comparison'].values.astype(float)
        rmse_m = float(np.sqrt(np.mean((prd_m - obs_m) ** 2)))
        mae_m = float(np.mean(np.abs(prd_m - obs_m)))
        truncated = int((sub_m['duration_handling'] == f'truncated_to_{MAX_MODEL_DURATION_WEEKS}wk_model_horizon').sum())
        lines += [
            "[extended_duration_external_comparison — Donnelly 2003 & 2013]",
            "  Donnelly 2003 and Donnelly 2013 exceeded the model's primary 24-week horizon",
            "  and were therefore treated as extended-duration comparisons rather than",
            "  primary external validation datasets.",
            f"  n = {donnelly_mask.sum()} arms | truncated predictions = {truncated}/{donnelly_mask.sum()}",
            f"  RMSE = {rmse_m:.3f} kg | MAE = {mae_m:.3f} kg",
            "  NOTE: observed changes are from 45-69 week trials; predictions are capped at 24 weeks.",
            "  Directional mismatch is expected when comparing longer observed interventions",
            "  with a shorter model horizon.",
            "",
        ]

    lines += [
        "Items required to complete validation:",
        "  1. Extract more independent external arms with observed delta_fm/delta_bw, SD, sample size, dose, and duration.",
        "  2. Once available, re-run this script.",
        "  3. With ≥10 independent estimates: compute ICC and 95% PI coverage.",
    ]

    report = "\n".join(lines)
    (out / 'validation_metrics_v2_2.txt').write_text(report)
    print(report)
    return df_out


if __name__ == '__main__':
    run_validation()
