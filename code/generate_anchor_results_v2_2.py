"""
generate_anchor_results_v2_2.py  —  V2.2.1

Regenerates the manuscript anchor table directly from the canonical scenario
summary and parametric PSA summary.

Rationale:
- scenario_summary_v2_2.csv contains the deterministic model mean and low-response
  probability for each scenario.
- psa_parametric_summary_v2_2.csv contains the outer-loop uncertainty interval.
- anchor_results_v2_2.csv should therefore be a derived reporting table, not a
  separately simulated file.

Run from the project root:
    python code_v2_2/generate_anchor_results_v2_2.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs_v2_2"

ANCHOR_DEFINITIONS = [
    {
        "anchor": "primary",
        "duration_weeks": 12,
        "dose_min_week": 300,
        "compensation_mean": 0.50,
    },
    {
        "anchor": "secondary_high_volume",
        "duration_weeks": 12,
        "dose_min_week": 450,
        "compensation_mean": 0.50,
    },
]


def _one_value(df: pd.DataFrame, mask: pd.Series, column: str) -> float:
    vals = df.loc[mask, column]
    if len(vals) != 1:
        raise ValueError(f"Expected one value for {column}, found {len(vals)}")
    return float(vals.iloc[0])


def generate_anchor_results(output_dir: Path | None = None) -> pd.DataFrame:
    out = output_dir or OUT_DIR
    scenario_path = out / "scenario_summary_v2_2.csv"
    psa_path = out / "psa_parametric_summary_v2_2.csv"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing {scenario_path}; run simulation_v2_2.py first.")
    if not psa_path.exists():
        raise FileNotFoundError(f"Missing {psa_path}; run psa_parametric_v2_2.py first.")

    scenario = pd.read_csv(scenario_path)
    psa = pd.read_csv(psa_path)
    rows = []
    for spec in ANCHOR_DEFINITIONS:
        dur = spec["duration_weeks"]
        dose = spec["dose_min_week"]
        comp = spec["compensation_mean"]

        base_mask = (
            (scenario["duration"] == dur) &
            (scenario["dose"] == dose) &
            (scenario["comp"] == comp)
        )
        additive_mask = (
            (scenario["duration"] == dur) &
            (scenario["dose"] == dose) &
            (scenario["comp"] == 0.00)
        )
        psa_mask = (
            (psa["duration"] == dur) &
            (psa["dose"] == dose) &
            (psa["comp"] == comp)
        )

        mean_delta_fm = _one_value(scenario, base_mask, "mean_fm")
        additive_prediction = _one_value(scenario, additive_mask, "mean_fm")
        low_response_probability = _one_value(scenario, base_mask, "low_response_prop")
        attenuation = 100.0 * (1.0 - mean_delta_fm / additive_prediction) if additive_prediction else float("nan")

        rows.append({
            "anchor": spec["anchor"],
            "duration_weeks": dur,
            "dose_min_week": dose,
            "compensation_mean": comp,
            "mean_delta_fm": round(mean_delta_fm, 4),
            "psa_lower": round(_one_value(psa, psa_mask, "psa_l"), 4),
            "psa_median": round(_one_value(psa, psa_mask, "psa_median"), 4),
            "psa_upper": round(_one_value(psa, psa_mask, "psa_u"), 4),
            "low_response_probability": round(low_response_probability, 4),
            "additive_prediction": round(additive_prediction, 4),
            "attenuation_vs_additive_pct": round(attenuation, 2),
            "source_note": "mean_delta_fm and additive_prediction derived from scenario_summary_v2_2.csv; PSA interval derived from psa_parametric_summary_v2_2.csv",
        })

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "anchor_results_v2_2.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to {out / 'anchor_results_v2_2.csv'}")
    return df


if __name__ == "__main__":
    generate_anchor_results()
