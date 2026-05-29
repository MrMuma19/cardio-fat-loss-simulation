"""
split_sensitivity_v2_2.py  —  V2.2.1

Sensitivity analysis for alternative compensation subfraction splits.

The total compensation fraction is held constant while the assumed split across
intake, NEAT, and adaptive thermogenesis is varied. Because NEAT and AT caps are
non-binding at the studied doses, changes in the split should have negligible or
zero impact on predicted fat-mass loss.

Run from the project root:
    python code_v2_2/split_sensitivity_v2_2.py
"""

from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
from simulation_v2_2 import (ModelConfig, HyperParameters, Scenario,
                               generate_population, simulate_scenario, summarize)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs_v2_2"

SPLIT_SCENARIOS = [
    ("base",            0.50, 0.30, 0.20),
    ("intake_dominant", 0.70, 0.20, 0.10),
    ("neat_dominant",   0.40, 0.45, 0.15),
    ("at_dominant",     0.40, 0.20, 0.40),
    ("balanced",        0.34, 0.33, 0.33),
]

TEST_SCENARIOS = [
    *(Scenario(dose, 12, 0.50) for dose in [75, 150, 225, 300, 375, 450]),
    Scenario(300, 12, 0.25),
    Scenario(300, 12, 0.75),
    Scenario(300, 24, 0.50),
]


def _simulate_with_state(pop: pd.DataFrame, sc: Scenario, cfg: ModelConfig,
                         hp: HyperParameters, rng_state: dict) -> float:
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(rng_state)
    ind = simulate_scenario(pop, sc, cfg, hp, rng)
    return summarize(ind, sc, cfg)["mean_fm"]


def run_split_sensitivity(output_dir: Path | None = None) -> pd.DataFrame:
    out = output_dir or OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    hp = HyperParameters()
    base_cfg = ModelConfig()

    rng_pop = np.random.default_rng(base_cfg.seed)
    pop = generate_population(base_cfg, hp, rng_pop)
    simulation_rng_state = deepcopy(rng_pop.bit_generator.state)

    baseline_by_scenario = {}
    for sc in TEST_SCENARIOS:
        baseline_by_scenario[(sc.duration_weeks, sc.dose_min_week, sc.comp_mean)] = _simulate_with_state(
            pop, sc, base_cfg, hp, simulation_rng_state)

    rows = []
    for split_name, intake_share, neat_share, at_share in SPLIT_SCENARIOS:
        cfg = ModelConfig(
            intake_share=intake_share,
            neat_share=neat_share,
            at_share=at_share,
        )
        for sc in TEST_SCENARIOS:
            key = (sc.duration_weeks, sc.dose_min_week, sc.comp_mean)
            result = _simulate_with_state(pop, sc, cfg, hp, simulation_rng_state)
            baseline = baseline_by_scenario[key]
            diff = result - baseline
            pct = 100.0 * diff / baseline if baseline else 0.0
            rows.append({
                "split_scenario": split_name,
                "intake_share": intake_share,
                "neat_share": neat_share,
                "at_share": at_share,
                "duration_weeks": sc.duration_weeks,
                "dose_min_week": sc.dose_min_week,
                "compensation_mean": sc.comp_mean,
                "mean_delta_fm": round(result, 6),
                "difference_vs_base": round(diff, 10),
                "percent_difference_vs_base": round(pct, 8),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / "split_sensitivity_v2_2.csv", index=False)
    max_abs_pct = df["percent_difference_vs_base"].abs().max()
    print(df.to_string(index=False))
    print(f"\nMaximum absolute percent difference vs base split: {max_abs_pct:.8f}%")
    print(f"Saved to {out / 'split_sensitivity_v2_2.csv'}")
    return df


if __name__ == "__main__":
    run_split_sensitivity()
