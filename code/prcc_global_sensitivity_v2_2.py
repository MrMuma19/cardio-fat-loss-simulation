"""
prcc_global_sensitivity_v2_2.py  —  V2.2

Partial Rank Correlation Coefficient (PRCC) global sensitivity analysis.

PRCC estimates monotonic rank-based associations between uncertain inputs
and model outputs. PRCC does NOT estimate variance explained.
Results are used as global sensitivity ranking, not causal attribution.

Primary scenario : 12 weeks, 300 min/week, 50% compensation
Secondary scenario: 12 weeks, 450 min/week, 50% compensation

Run: python code_v2_2/prcc_global_sensitivity_v2_2.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from simulation_v2_2 import (ModelConfig, HyperParameters, Scenario,
                               generate_population, simulate_scenario, summarize)

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs_v2_2"
N_SAMPLES = 2_000  # LHS sample size


def latin_hypercube_sample(n: int, seed: int = 99) -> pd.DataFrame:
    """
    Generate n Latin Hypercube samples for the uncertainty parameters.
    Each parameter is sampled uniformly across its plausible range.
    """
    rng = np.random.default_rng(seed)

    # Parameter ranges [low, high] — broad plausible ranges
    params = {
        'met_mean':                 (3.8,  5.2),
        'met_sd':                   (0.3,  0.9),
        'adherence_base_mean':      (0.68, 0.96),
        'adherence_dose_penalty':   (0.00, 0.10),
        'rmr_residual_sd':          (0.05, 0.18),
        'comp_var':                 (0.005,0.065),
        'neat_cap_fraction':        (0.20, 0.90),
        'at_cap_fraction':          (0.04, 0.25),
        'comp_mean_perturbation':   (-0.10, 0.10),  # perturbation around scenario comp=0.50
    }

    rows = {}
    for name, (lo, hi) in params.items():
        # LHS: divide [0,1] into n equal intervals, sample one point per interval
        cuts   = np.linspace(0, 1, n+1)
        points = cuts[:-1] + rng.uniform(0, 1/n, n)
        rng.shuffle(points)
        rows[name] = lo + points * (hi - lo)

    return pd.DataFrame(rows)


def run_prcc(output_dir: Path | None = None) -> pd.DataFrame:
    out = output_dir or OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {'label': 'primary_300min_50comp_12wk',   'dose': 300, 'dur': 12, 'comp_base': 0.50},
        {'label': 'secondary_450min_50comp_12wk',  'dose': 450, 'dur': 12, 'comp_base': 0.50},
    ]

    lhs = latin_hypercube_sample(N_SAMPLES)
    all_rows = []

    for sc_def in scenarios:
        print(f"\nRunning PRCC for: {sc_def['label']}")
        outputs = []

        for i, sample in lhs.iterrows():
            hp = HyperParameters(
                met_mean      = float(sample['met_mean']),
                met_sd        = float(sample['met_sd']),
                adherence_base_mean = float(sample['adherence_base_mean']),
                adherence_dose_penalty_per_150min_above_150 = float(sample['adherence_dose_penalty']),
                rmr_residual_sd = float(sample['rmr_residual_sd']),
            )
            comp_perturbed = float(np.clip(sc_def['comp_base'] + sample['comp_mean_perturbation'], 0.01, 0.95))
            cfg = ModelConfig(
                n=500,
                comp_var = float(sample['comp_var']),
                neat_cap_fraction_of_nonresting = float(sample['neat_cap_fraction']),
                at_cap_fraction_of_rmr = float(sample['at_cap_fraction']),
            )
            rng_i = np.random.default_rng(42 + i)
            pop   = generate_population(cfg, hp, rng_i)
            sc    = Scenario(sc_def['dose'], sc_def['dur'], comp_perturbed)
            ind   = simulate_scenario(pop, sc, cfg, hp, rng_i)
            s     = summarize(ind, sc, cfg)
            outputs.append(s['mean_fm'])

            if (i+1) % 200 == 0:
                print(f"  Sample {i+1}/{N_SAMPLES}")

        y = np.array(outputs)
        # PRCC: partial rank correlation — approximate via Spearman pairwise
        # For true PRCC, regress out all other parameters; here use Spearman
        # as a first-pass global sensitivity index
        param_names = list(lhs.columns)
        X = lhs.values

        # Rank-transform
        from scipy.stats import rankdata
        y_rank = rankdata(y)
        X_rank = np.apply_along_axis(rankdata, 0, X)

        prcc_vals, p_vals = [], []
        for j in range(X.shape[1]):
            # Partial rank correlation: residualise y and x_j on all other x's
            other_cols = [k for k in range(X.shape[1]) if k != j]
            X_others   = X_rank[:, other_cols]
            # OLS residuals
            from numpy.linalg import lstsq
            def resid(target, predictors):
                A = np.column_stack([np.ones(len(target)), predictors])
                coef, _, _, _ = lstsq(A, target, rcond=None)
                return target - A @ coef

            r_y = resid(y_rank, X_others)
            r_x = resid(X_rank[:, j], X_others)
            rho, pv = spearmanr(r_x, r_y)
            prcc_vals.append(float(rho))
            p_vals.append(float(pv))

        # Rank by absolute PRCC
        abs_prcc  = np.abs(prcc_vals)
        rank_order = np.argsort(-abs_prcc) + 1  # 1 = largest

        for j, pname in enumerate(param_names):
            interp = ('positive association' if prcc_vals[j] > 0 else 'negative association')
            interp += ' (monotonic, rank-based; does not imply variance explained)'
            all_rows.append({
                'scenario': sc_def['label'],
                'parameter': pname,
                'prcc': round(prcc_vals[j], 4),
                'p_value': round(p_vals[j], 5),
                'rank_abs_prcc': int(np.where(rank_order == j+1)[0][0]+1) if j < len(rank_order) else j+1,
                'interpretation': interp,
            })
        # fix rank
        prcc_df_sc = pd.DataFrame([r for r in all_rows if r['scenario']==sc_def['label']])
        sorted_by_abs = prcc_df_sc.reindex(prcc_df_sc['prcc'].abs().sort_values(ascending=False).index)
        rank_map = {row['parameter']: i+1 for i, (_, row) in enumerate(sorted_by_abs.iterrows())}
        for r in all_rows:
            if r['scenario'] == sc_def['label']:
                r['rank_abs_prcc'] = rank_map.get(r['parameter'], r['rank_abs_prcc'])

    df = pd.DataFrame(all_rows)
    df.to_csv(out / 'prcc_global_sensitivity_v2_2.csv', index=False)

    # Print summary
    print("\n=== PRCC Summary ===")
    for sc_label in df['scenario'].unique():
        print(f"\n{sc_label}:")
        sub = df[df['scenario']==sc_label].sort_values('rank_abs_prcc')
        print(sub[['parameter','prcc','p_value','rank_abs_prcc']].to_string(index=False))

    print(f"\nSaved to {out/'prcc_global_sensitivity_v2_2.csv'}")
    return df


if __name__ == '__main__':
    run_prcc()
