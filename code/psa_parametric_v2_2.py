"""
psa_parametric_v2_2.py  —  V2.2 hotfix

Path fix: all outputs written to ROOT/outputs_v2_2.
Model unchanged. Column names: psa_l / psa_median / psa_u.
PSA covers all 4 durations; primary manuscript analysis uses 12 weeks.

Run: python code_v2_2/psa_parametric_v2_2.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from simulation_v2_2 import (ModelConfig, HyperParameters, Scenario,
                               generate_population, simulate_scenario, summarize)

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs_v2_2"


def run_psa(n_outer: int = 500, n_inner: int = 2_000,
            seed: int = 20250526, output_dir: Path | None = None) -> pd.DataFrame:
    out = output_dir or OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    scenarios = [Scenario(dose, dur, comp)
                 for dur  in [8, 12, 16, 24]
                 for dose in [0, 75, 150, 225, 300, 375, 450]
                 for comp in [0.00, 0.25, 0.50, 0.75]]

    rng_outer = np.random.default_rng(seed)
    rows = []
    for i in range(n_outer):
        hp = HyperParameters(
            met_mean=float(np.clip(rng_outer.normal(4.5, 0.30), 3.5, 6.0)),
            met_sd=float(np.clip(rng_outer.normal(0.6, 0.08), 0.3, 1.0)),
            adherence_base_mean=float(np.clip(rng_outer.normal(0.82, 0.06), 0.60, 0.95)),
            adherence_dose_penalty_per_150min_above_150=float(
                np.clip(rng_outer.normal(0.04, 0.015), 0.00, 0.10)),
            rmr_residual_sd=float(np.clip(rng_outer.normal(0.10, 0.02), 0.05, 0.20)),
        )
        cfg = ModelConfig(n=n_inner, seed=seed+i)
        rng_inner = np.random.default_rng(seed+i)
        pop = generate_population(cfg, hp, rng_inner)
        for sc in scenarios:
            ind = simulate_scenario(pop, sc, cfg, hp, rng_inner)
            s   = summarize(ind, sc, cfg)
            rows.append({'psa_iter':i,'duration':sc.duration_weeks,
                         'dose':sc.dose_min_week,'comp':sc.comp_mean,'mean_fm':s['mean_fm']})
        if (i+1) % 50 == 0:
            print(f"  PSA iteration {i+1}/{n_outer}")

    df = pd.DataFrame(rows)
    df.to_csv(out / 'psa_parametric_outerloop_v2_2.csv', index=False)

    summary = (df.groupby(['duration','dose','comp'])['mean_fm']
               .quantile([0.025, 0.50, 0.975]).unstack().reset_index()
               .rename(columns={0.025:'psa_l', 0.50:'psa_median', 0.975:'psa_u'}))
    summary.to_csv(out / 'psa_parametric_summary_v2_2.csv', index=False)

    print(f"\nPSA complete. Saved to {out}")
    print("\n=== 12-week PSA summary ===")
    print(summary[summary.duration==12][['dose','comp','psa_l','psa_median','psa_u']].to_string(index=False))
    return summary


if __name__ == '__main__':
    run_psa()
