"""
More Cardio, Less Fat?
A Probabilistic Simulation of Exercise-Induced Fat Loss
Under Different Energy Compensation Scenarios

Fit Generation Research Institute — Andorra la Vella, Andorra
Corresponding author: José Francisco Tornero-Aguilera
jtornero@fitgeneration.es

Preregistration: OSF (link to be added after registration)
Random seed: 42 (fixed for full reproducibility)

Usage:
    python simulation_main.py

Outputs:
    outputs/sim_results.csv   — full scenario-level statistics
    outputs/sim_raw.pkl       — raw individual-level data (all scenarios)
"""

import numpy as np
import pandas as pd
import pickle
from scipy.stats import truncnorm, beta as beta_dist, sem
from scipy.stats import mannwhitneyu, ks_2samp

# ─────────────────────────────────────────────────────────────────
# RANDOM SEED — DO NOT CHANGE (preregistered)
# ─────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

N = 10_000  # individuals per scenario (preregistered)

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def truncated_normal(mu, sd, lo, hi, size=N):
    """Sample from a truncated normal distribution."""
    a, b = (lo - mu) / sd, (hi - mu) / sd
    return truncnorm.rvs(a, b, loc=mu, scale=sd, size=size,
                         random_state=np.random.get_state()[1][0])


def beta_from_mean_var(mu, var, size=N):
    """
    Sample from a Beta distribution parameterised by mean and variance.
    Clips alpha and beta to minimum 0.5 to avoid degenerate distributions.
    """
    alpha = mu * (mu * (1 - mu) / var - 1)
    beta_p = (1 - mu) * (mu * (1 - mu) / var - 1)
    alpha = max(alpha, 0.5)
    beta_p = max(beta_p, 0.5)
    return beta_dist.rvs(alpha, beta_p, size=size)


def wilson_ci(n_success, n_total, z=1.96):
    """Wilson score confidence interval for a proportion."""
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ─────────────────────────────────────────────────────────────────
# VIRTUAL POPULATION
# (distributions pre-specified in preregistration — do not modify)
# ─────────────────────────────────────────────────────────────────

def generate_population(n=N):
    """
    Generate a virtual population of n individuals.

    All distributions are calibrated to NHANES 2017-2018 and the
    Jayedi et al. (2024) meta-analysis population, as preregistered.
    """
    sex    = np.random.binomial(1, 0.5, n)          # 1 = male, 0 = female
    age    = truncated_normal(46, 12, 25, 65, n)

    bw     = np.where(
                sex == 1,
                truncated_normal(90, 18, 55, 160, n),
                truncated_normal(77, 18, 55, 160, n)
             )
    bmi    = truncated_normal(33, 4.5, 27, 45, n)
    bfp    = np.where(
                sex == 1,
                truncated_normal(32, 5.5, 18, 55, n),
                truncated_normal(42, 5.5, 22, 58, n)
             )

    # Height derived algebraically from BW and BMI
    height = np.sqrt(bw / bmi) * 100  # cm

    # RMR — Mifflin-St Jeor equation with individual residual error
    rmr_base = np.where(
        sex == 1,
        10 * bw + 6.25 * height - 5 * age + 5,
        10 * bw + 6.25 * height - 5 * age - 161
    )
    rmr = np.clip(
        rmr_base * truncated_normal(1.0, 0.10, 0.75, 1.30, n),
        1000, 3500
    )

    pal    = truncated_normal(1.40, 0.08, 1.25, 1.60, n)

    # Body composition
    fm0    = bw * bfp / 100          # initial fat mass (kg)
    ffm0   = bw - fm0                # initial fat-free mass (kg)

    # Forbes-Hall P-ratio: fraction of weight change from FFM
    # P = FFM / (FM + c),  c = 10.4 kg  (Hall 2007)
    P_ratio = np.clip(ffm0 / (fm0 + 10.4), 0.05, 0.50)

    # Effective energy density of tissue lost
    # fat fraction = 9441 kcal/kg; FFM fraction = 1816 kcal/kg
    ED_eff  = P_ratio * 1816.0 + (1 - P_ratio) * 9441.0

    # Individual MET value (exercise economy variability)
    MET_ind = truncated_normal(4.5, 0.6, 3.0, 6.5, n)

    return {
        'sex': sex, 'age': age, 'bw': bw, 'bmi': bmi, 'bfp': bfp,
        'height': height, 'rmr': rmr, 'pal': pal,
        'fm0': fm0, 'ffm0': ffm0, 'P_ratio': P_ratio,
        'ED_eff': ED_eff, 'MET_ind': MET_ind
    }


# ─────────────────────────────────────────────────────────────────
# SCENARIO PARAMETERS (preregistered)
# ─────────────────────────────────────────────────────────────────

DOSES_MIN     = [0, 75, 150, 300, 450]     # min/week
COMP_LEVELS   = [0.0, 0.25, 0.50, 0.75]   # fraction of ExEE compensated
DURATIONS_WK  = [8, 12, 16, 24]            # weeks
SESSIONS_PW   = 3                           # sessions per week

# Adaptive thermogenesis scaling by duration (Pontzer & Trexler 2026)
AT_SCALE = {8: 0.80, 12: 1.00, 16: 1.15, 24: 1.35}

# Non-responder threshold (preregistered)
NR_THRESHOLD_KG = 0.50


def get_adherence(dose_min, size=N):
    """
    Sample individual adherence from a Beta distribution.
    Mean adherence decreases by 0.04 per 150 min/wk above 150 min/wk
    (preregistered; calibrated to STRRIDE adherence data).
    """
    penalty = max(0.0, (dose_min - 150) / 150) * 0.04
    mu_adh  = max(0.60, 0.82 - penalty)
    var_adh = 0.025
    return beta_from_mean_var(mu_adh, var_adh, size)


def get_compensation(comp_mean, size=N):
    """
    Sample individual compensation fraction from a Beta distribution.
    Variance fixed at 0.032 (calibrated to Riou et al. 2015 SD = 93%).
    Returns zeros for the 0% compensation (additive model) scenario.
    """
    if comp_mean == 0.0:
        return np.zeros(size)
    return np.clip(beta_from_mean_var(comp_mean, 0.032, size), 0.0, 0.99)


# ─────────────────────────────────────────────────────────────────
# MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────

def simulate_scenario(pop, dose_min, comp_mean, duration_wk):
    """
    Simulate fat-mass loss for n individuals under one scenario.

    Parameters
    ----------
    pop        : dict — virtual population arrays
    dose_min   : int  — weekly exercise volume (min/week)
    comp_mean  : float — mean energy compensation fraction
    duration_wk: int  — intervention duration (weeks)

    Returns
    -------
    delta_fm   : np.ndarray — individual fat-mass losses (kg)
    """
    if dose_min == 0:
        return np.zeros(N)

    sess_dur_h = (dose_min / SESSIONS_PW) / 60.0

    # Gross weekly exercise energy expenditure (kcal)
    ExEE_week = pop['MET_ind'] * pop['bw'] * sess_dur_h * SESSIONS_PW

    # Adherence
    adh = get_adherence(dose_min)

    # Individual compensation with adaptive thermogenesis scaling
    comp_ind = get_compensation(comp_mean)
    comp_adj = np.clip(comp_ind * AT_SCALE[duration_wk], 0.0, 0.98)

    # Net accumulated deficit (kcal) over full intervention
    net_deficit = ExEE_week * adh * (1.0 - comp_adj) * duration_wk

    # Fat-mass loss via Forbes-Hall model
    delta_bw = net_deficit / pop['ED_eff']
    delta_fm = delta_bw * (1.0 - pop['P_ratio'])

    return delta_fm


def compute_statistics(delta_fm, dose_min, comp_mean, duration_wk):
    """Compute full statistical summary for one scenario."""
    mn   = np.mean(delta_fm)
    sd   = np.std(delta_fm, ddof=1)
    se   = sem(delta_fm)
    ci_l = mn - 1.96 * se
    ci_u = mn + 1.96 * se

    # Bootstrap PSA uncertainty intervals (1000 resamples)
    boot = [np.mean(np.random.choice(delta_fm, N, replace=True))
            for _ in range(1000)]
    psa_l = np.percentile(boot, 2.5)
    psa_u = np.percentile(boot, 97.5)

    # Distribution percentiles
    pcts = np.percentile(delta_fm, [5, 25, 50, 75, 95])

    # Low-responder proportion with Wilson CI
    n_nr    = int(np.sum(delta_fm < NR_THRESHOLD_KG))
    nr_prop = n_nr / N
    nr_lo, nr_hi = wilson_ci(n_nr, N)

    return {
        'duration':  duration_wk,
        'dose':      dose_min,
        'comp':      comp_mean,
        'mean_fm':   round(mn,   4),
        'sd_fm':     round(sd,   4),
        'se_fm':     round(se,   5),
        'ci95_l':    round(ci_l, 4),
        'ci95_u':    round(ci_u, 4),
        'psa_l':     round(psa_l, 4),
        'psa_u':     round(psa_u, 4),
        'p5':        round(pcts[0], 4),
        'p25':       round(pcts[1], 4),
        'p50':       round(pcts[2], 4),
        'p75':       round(pcts[3], 4),
        'p95':       round(pcts[4], 4),
        'iqr':       round(pcts[3] - pcts[1], 4),
        'nr_prop':   round(nr_prop, 5),
        'nr_ci_l':   round(nr_lo, 5),
        'nr_ci_u':   round(nr_hi, 5),
        'n_nr':      n_nr,
    }


def run_full_simulation():
    """Run all 80 primary scenarios and return results."""
    print(f"Generating virtual population (n={N}, seed={SEED})...")
    pop = generate_population()

    results     = []
    raw_records = {}
    total = len(DURATIONS_WK) * len(DOSES_MIN) * len(COMP_LEVELS)
    done  = 0

    for dur in DURATIONS_WK:
        for dose in DOSES_MIN:
            for comp in COMP_LEVELS:
                delta_fm = simulate_scenario(pop, dose, comp, dur)
                stats    = compute_statistics(delta_fm, dose, comp, dur)
                results.append(stats)
                raw_records[(dur, dose, comp)] = delta_fm
                done += 1
                print(f"  [{done:3d}/{total}] {dur}wk | {dose:3d}min/wk | {int(comp*100):2d}% comp "
                      f"→ mean ΔFM = {stats['mean_fm']:.3f} kg")

    return pd.DataFrame(results), raw_records


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)

    df, raw = run_full_simulation()

    # Save scenario-level summary
    df.to_csv('outputs/sim_results.csv', index=False)
    print(f"\nSaved: outputs/sim_results.csv  ({len(df)} rows)")

    # Save raw individual-level data
    with open('outputs/sim_raw.pkl', 'wb') as f:
        pickle.dump(raw, f)
    print("Saved: outputs/sim_raw.pkl")

    print("\n=== 12-week summary ===")
    print(df[df.duration == 12][
        ['dose', 'comp', 'mean_fm', 'sd_fm', 'ci95_l', 'ci95_u',
         'p5', 'p95', 'nr_prop']
    ].to_string(index=False))
