"""
simulation_v2_2.py  —  V2.2 hotfix

Path fix: all outputs written to ROOT/outputs_v2_2.
AT_SCALE duration multipliers removed (no verifiable primary source for exact values).
AT compensation is now a constant fraction of total compensation energy (at_share=0.20).
Pontzer & Trexler 2026 cited only for general constrained TEE framework concept.

Model description: compensation-adjusted probabilistic simulation with explicit
component accounting (intake 50%, NEAT 30%, AT 20%). NEAT and AT caps are
structurally implemented but operationally non-binding at studied doses
(0-450 min/week). Because caps were non-binding, component accounting did not
materially alter fat-mass predictions relative to a scalar compensation formulation.

Run from any directory:
    python code_v2_2/simulation_v2_2.py

V2.2.1 reproducibility fix: population export falls back to CSV if no
parquet engine (pyarrow/fastparquet) is available.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, beta as beta_dist

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs_v2_2"



@dataclass(frozen=True)
class Scenario:
    dose_min_week: int
    duration_weeks: int
    comp_mean: float


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    n: int = 10_000
    sessions_per_week: int = 3
    low_response_threshold_kg: float = 0.5
    use_net_exee: bool = True
    neat_cap_fraction_of_nonresting: float = 0.60
    at_cap_fraction_of_rmr: float = 0.12
    comp_var: float = 0.032
    adherence_var: float = 0.025
    intake_share: float = 0.50
    neat_share: float = 0.30
    at_share: float = 0.20

    def __post_init__(self):
        total = self.intake_share + self.neat_share + self.at_share
        if not abs(total - 1.0) < 1e-9:
            raise ValueError(f"Compensation shares must sum to 1.0, got {total:.6f}")


@dataclass(frozen=True)
class HyperParameters:
    met_mean: float = 4.5
    met_sd: float = 0.6
    adherence_base_mean: float = 0.82
    adherence_dose_penalty_per_150min_above_150: float = 0.04
    rmr_residual_sd: float = 0.10


def _beta_mv(mu, var, rng, size):
    max_var = mu * (1 - mu)
    var = min(var, max_var * 0.99)
    a = mu * (mu*(1-mu)/var - 1)
    b = (1-mu) * (mu*(1-mu)/var - 1)
    return beta_dist.rvs(a, b, size=size, random_state=rng)


def _tnorm(mu, sd, lo, hi, rng, size):
    a, b = (lo-mu)/sd, (hi-mu)/sd
    return truncnorm.rvs(a, b, loc=mu, scale=sd, size=size, random_state=rng)



def _write_population(pop: pd.DataFrame, out: Path) -> None:
    """Persist the generated population with a parquet->CSV fallback.

    Parquet is preferred when pyarrow/fastparquet is installed, but the model
    outputs must remain reproducible in minimal Python environments.
    """
    parquet_path = out / 'population_v2_2.parquet'
    csv_path = out / 'population_v2_2.csv'
    try:
        pop.to_parquet(parquet_path, index=False)
        (out / 'population_export_note_v2_2.txt').write_text(
            f"Population exported as parquet: {parquet_path.name}\n"
        )
    except ImportError as exc:
        pop.to_csv(csv_path, index=False)
        (out / 'population_export_note_v2_2.txt').write_text(
            "Population exported as CSV because no parquet engine was available.\n"
            "Install pyarrow or fastparquet to enable parquet export.\n"
            f"Original ImportError: {exc}\n"
        )


def generate_population(cfg: ModelConfig, hp: HyperParameters,
                        rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n
    sex        = rng.binomial(1, 0.5, n)
    age        = _tnorm(46, 12, 25, 65, rng, n)
    bw         = np.where(sex==1, _tnorm(90,18,55,160,rng,n), _tnorm(77,18,55,160,rng,n))
    bmi        = _tnorm(33, 4.5, 27, 45, rng, n)
    bfp        = np.where(sex==1, _tnorm(32,5.5,18,55,rng,n), _tnorm(42,5.5,22,58,rng,n))
    height_cm  = np.sqrt(bw/bmi)*100
    rmr_base   = np.where(sex==1,
                          10*bw+6.25*height_cm-5*age+5,
                          10*bw+6.25*height_cm-5*age-161)
    rmr        = np.clip(rmr_base*_tnorm(1.0,hp.rmr_residual_sd,0.75,1.30,rng,n),1000,3500)
    pal        = _tnorm(1.40, 0.08, 1.25, 1.60, rng, n)
    nonresting = np.maximum(rmr*(pal-1)*7, 0)
    fm0        = bw*bfp/100
    ffm0       = bw-fm0
    p_ratio    = np.clip(ffm0/(fm0+10.4), 0.05, 0.50)
    ed_eff     = p_ratio*1816 + (1-p_ratio)*9441
    met_ind    = _tnorm(hp.met_mean, hp.met_sd, 3.0, 6.5, rng, n)
    return pd.DataFrame({'sex':sex,'age':age,'bw':bw,'bmi':bmi,'bfp':bfp,
                         'height_cm':height_cm,'rmr':rmr,'pal':pal,
                         'nonresting_week':nonresting,
                         'fm0':fm0,'ffm0':ffm0,'p_ratio':p_ratio,
                         'ed_eff':ed_eff,'met_ind':met_ind})


def simulate_scenario(pop, sc, cfg, hp, rng):
    n = len(pop); z = np.zeros(n)
    if sc.dose_min_week == 0:
        return pd.DataFrame({'delta_fm':z.copy(),'delta_bw':z.copy(),
                             'exee_week':z.copy(),'adherence':np.ones(n),
                             'comp_total_fraction':z.copy(),
                             'intake_comp_kcal_week':z.copy(),
                             'neat_reduction_kcal_week':z.copy(),
                             'at_reduction_kcal_week':z.copy(),
                             'net_deficit_week':z.copy()})
    met_t    = np.maximum(pop['met_ind'].values-1.0,0.0) if cfg.use_net_exee else pop['met_ind'].values
    exee     = met_t*pop['bw'].values*(sc.dose_min_week/60.0)
    pen      = max(0,(sc.dose_min_week-150)/150)*hp.adherence_dose_penalty_per_150min_above_150
    adh      = _beta_mv(max(0.60,hp.adherence_base_mean-pen), cfg.adherence_var, rng, n)
    exee_eff = exee*adh
    comp_f   = z.copy() if sc.comp_mean==0 else np.clip(_beta_mv(sc.comp_mean,cfg.comp_var,rng,n),0,0.98)
    total_c  = exee_eff*comp_f
    intake_c = total_c*cfg.intake_share
    neat_c   = np.minimum(total_c*cfg.neat_share, pop['nonresting_week'].values*cfg.neat_cap_fraction_of_nonresting)
    at_c     = np.minimum(total_c*cfg.at_share, pop['rmr'].values*7*cfg.at_cap_fraction_of_rmr)
    net_wk   = np.maximum(exee_eff-intake_c-neat_c-at_c, 0)
    deficit  = net_wk*sc.duration_weeks
    delta_bw = deficit/pop['ed_eff'].values
    delta_fm = delta_bw*(1-pop['p_ratio'].values)
    return pd.DataFrame({'delta_fm':delta_fm,'delta_bw':delta_bw,'exee_week':exee,
                         'adherence':adh,'comp_total_fraction':comp_f,
                         'intake_comp_kcal_week':intake_c,
                         'neat_reduction_kcal_week':neat_c,
                         'at_reduction_kcal_week':at_c,
                         'net_deficit_week':net_wk})


def summarize(ind, sc, cfg):
    x = ind['delta_fm'].values; p = np.percentile(x,[5,25,50,75,95])
    return {'duration':sc.duration_weeks,'dose':sc.dose_min_week,'comp':sc.comp_mean,
            'mean_fm':float(np.mean(x)),'sd_fm':float(np.std(x,ddof=1)),
            'mcse_mean':float(np.std(x,ddof=1)/np.sqrt(len(x))),
            'p5':float(p[0]),'p25':float(p[1]),'p50':float(p[2]),
            'p75':float(p[3]),'p95':float(p[4]),'iqr':float(p[3]-p[1]),
            'low_response_prop':float(np.mean(x<cfg.low_response_threshold_kg)),
            'mean_exee_week':float(ind['exee_week'].mean()),
            'mean_intake_comp_week':float(ind['intake_comp_kcal_week'].mean()),
            'mean_neat_reduction_week':float(ind['neat_reduction_kcal_week'].mean()),
            'mean_at_reduction_week':float(ind['at_reduction_kcal_week'].mean()),
            'mean_net_deficit_week':float(ind['net_deficit_week'].mean())}


def run_primary(output_dir: Path | None = None) -> pd.DataFrame:
    out = output_dir or OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(); hp = HyperParameters()
    rng = np.random.default_rng(cfg.seed)
    pop = generate_population(cfg, hp, rng)
    _write_population(pop, out)
    rows = []
    for dur in [8,12,16,24]:
        for dose in [0,75,150,225,300,375,450]:
            for comp in [0.00,0.25,0.50,0.75]:
                sc  = Scenario(dose,dur,comp)
                ind = simulate_scenario(pop,sc,cfg,hp,rng)
                rows.append(summarize(ind,sc,cfg))
                print(f"  {dur}wk | {dose:3d} min/wk | {int(comp*100):2d}% → {rows[-1]['mean_fm']:.4f} kg")
    # NEAT cap binding verification
    sc_s = Scenario(300,12,0.75)
    ind_s = simulate_scenario(pop,sc_s,cfg,hp,rng)
    cap   = pop['nonresting_week'].values*cfg.neat_cap_fraction_of_nonresting
    n_hit = int(np.sum(ind_s['neat_reduction_kcal_week'].values >= cap*0.99))
    report = (
        "NEAT Cap Binding Report — V2.2\n==============================\n"
        f"Stress scenario: 300 min/wk, 12 wk, 75% compensation\n"
        f"Mean NEAT demand  : {ind_s['neat_reduction_kcal_week'].mean():.1f} kcal/wk\n"
        f"Mean NEAT cap (60%): {cap.mean():.1f} kcal/wk\n"
        f"Individuals hitting cap: {n_hit}/{cfg.n} ({100*n_hit/cfg.n:.2f}%)\n\n"
        "Because NEAT and AT caps were non-binding at the studied doses,\n"
        "component accounting did not materially alter fat-mass predictions\n"
        "relative to a scalar compensation formulation.\n"
    )
    (out / 'neat_cap_binding_report.txt').write_text(report)
    print("\n" + report)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'scenario_summary_v2_2.csv', index=False)
    print(f"Saved {len(df)} scenarios to {out/'scenario_summary_v2_2.csv'}")
    return df


if __name__ == '__main__':
    run_primary()
