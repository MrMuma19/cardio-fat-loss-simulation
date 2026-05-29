"""
test_model_v2_2.py  —  V2.2 hotfix

Import fix: uses simulation_v2_2 (not simulation_v2_1).
Run: python -m pytest code_v2_2/test_model_v2_2.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
from simulation_v2_2 import (
    ModelConfig, HyperParameters, Scenario,
    generate_population, simulate_scenario, summarize,
)


@pytest.fixture(scope='module')
def base_pop():
    cfg = ModelConfig(n=2000)
    hp  = HyperParameters()
    rng = np.random.default_rng(42)
    return generate_population(cfg, hp, rng), cfg, hp


def test_control_condition_zero(base_pop):
    pop, cfg, hp = base_pop
    rng = np.random.default_rng(42)
    ind = simulate_scenario(pop, Scenario(0, 12, 0.50), cfg, hp, rng)
    assert ind['delta_fm'].max() == pytest.approx(0.0, abs=1e-9)


def test_higher_compensation_lowers_loss(base_pop):
    pop, cfg, hp = base_pop
    means = {}
    for comp in [0.0, 0.25, 0.50, 0.75]:
        rng = np.random.default_rng(42)
        ind = simulate_scenario(pop, Scenario(300, 12, comp), cfg, hp, rng)
        means[comp] = summarize(ind, Scenario(300, 12, comp), cfg)['mean_fm']
    assert means[0.0] > means[0.25] > means[0.50] > means[0.75]


def test_higher_dose_increases_loss_at_same_compensation(base_pop):
    pop, cfg, hp = base_pop
    means = {}
    for dose in [75, 150, 300, 450]:
        rng = np.random.default_rng(42)
        ind = simulate_scenario(pop, Scenario(dose, 12, 0.50), cfg, hp, rng)
        means[dose] = summarize(ind, Scenario(dose, 12, 0.50), cfg)['mean_fm']
    assert means[75] < means[150] < means[300] < means[450]


def test_component_accounting_close(base_pop):
    pop, cfg, hp = base_pop
    rng = np.random.default_rng(42)
    ind = simulate_scenario(pop, Scenario(300, 12, 0.50), cfg, hp, rng)
    exee_eff = ind['exee_week'] * ind['adherence']
    total_c  = exee_eff * ind['comp_total_fraction']
    comp_sum = (ind['intake_comp_kcal_week']
                + ind['neat_reduction_kcal_week']
                + ind['at_reduction_kcal_week'])
    ratio = (comp_sum / total_c.replace(0, np.nan)).dropna()
    assert ratio.mean() == pytest.approx(1.0, abs=0.01)


def test_shares_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        ModelConfig(intake_share=0.5, neat_share=0.4, at_share=0.4)


def test_rng_reproducibility():
    cfg = ModelConfig(n=500); hp = HyperParameters()
    sc  = Scenario(150, 12, 0.50); results = []
    for _ in range(2):
        rng = np.random.default_rng(42)
        pop = generate_population(cfg, hp, rng)
        ind = simulate_scenario(pop, sc, cfg, hp, rng)
        results.append(summarize(ind, sc, cfg)['mean_fm'])
    assert results[0] == pytest.approx(results[1], abs=1e-12)
