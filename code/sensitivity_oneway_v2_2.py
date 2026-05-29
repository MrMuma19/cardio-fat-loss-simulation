"""
sensitivity_oneway_v2_2.py  —  V2.2 hotfix

Path fix: outputs written to ROOT/outputs_v2_2.
All sensitivity values computed from simulation reruns (no hard-coded values).
Interpretation: largest single-parameter effect on mean fat-mass loss predictions.
"Variance explained" language is NOT used.

Run: python code_v2_2/sensitivity_oneway_v2_2.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from simulation_v2_2 import (ModelConfig, HyperParameters, Scenario,
                               generate_population, simulate_scenario, summarize)

ROOT     = Path(__file__).resolve().parents[1]
OUT_DIR  = ROOT / "outputs_v2_2"
BASE_SC  = Scenario(300, 12, 0.50)
BASE_CFG = ModelConfig()
BASE_HP  = HyperParameters()

PARAMETER_RANGES = [
    ('MET mean',               'hp',  'met_mean',                                   4.0,  5.0),
    ('MET SD',                 'hp',  'met_sd',                                     0.4,  0.9),
    ('Adherence base mean',    'hp',  'adherence_base_mean',                        0.72, 0.92),
    ('Adherence dose penalty', 'hp',  'adherence_dose_penalty_per_150min_above_150',0.01, 0.08),
    ('RMR residual SD',        'hp',  'rmr_residual_sd',                            0.06, 0.16),
    ('Comp variance',          'cfg', 'comp_var',                                   0.010,0.060),
    ('NEAT cap fraction',      'cfg', 'neat_cap_fraction_of_nonresting',            0.20, 0.90),
    ('AT cap fraction',        'cfg', 'at_cap_fraction_of_rmr',                     0.04, 0.25),
]


def _run_one(cfg, hp, seed=42):
    rng = np.random.default_rng(seed)
    pop = generate_population(cfg, hp, rng)
    ind = simulate_scenario(pop, BASE_SC, cfg, hp, rng)
    return summarize(ind, BASE_SC, cfg)['mean_fm']


def run_sensitivity(output_dir: Path | None = None) -> pd.DataFrame:
    out = output_dir or OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    baseline = _run_one(BASE_CFG, BASE_HP)
    print(f"Base-case mean_fm: {baseline:.4f} kg")

    rows = []
    for label, ptype, attr, p10, p90 in PARAMETER_RANGES:
        for level, val in [('low', p10), ('high', p90)]:
            cfg_mod = ModelConfig(**{**BASE_CFG.__dict__, attr:val}) if ptype=='cfg' else BASE_CFG
            hp_mod  = HyperParameters(**{**BASE_HP.__dict__, attr:val}) if ptype=='hp'  else BASE_HP
            result  = _run_one(cfg_mod, hp_mod)
            rel_ch  = 100*(result-baseline)/baseline if baseline>0 else 0.0
            rows.append({'parameter':label,'attribute':attr,'level':level,
                         'parameter_value':val,'mean_fm':result,
                         'absolute_change':result-baseline,'relative_change_pct':rel_ch})
            print(f"  {label:30s} [{level:4s}={val}] → {result:.4f} kg ({rel_ch:+.2f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(out / 'sensitivity_oneway_v2_2.csv', index=False)
    print(f"\nSaved to {out/'sensitivity_oneway_v2_2.csv'}")
    piv = df.pivot(index='parameter', columns='level', values='relative_change_pct')
    piv['range'] = (piv['high']-piv['low']).abs()
    print(piv.sort_values('range', ascending=False)[['low','high','range']].round(2).to_string())
    return df


if __name__ == '__main__':
    run_sensitivity()
