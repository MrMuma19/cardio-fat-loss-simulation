"""
generate_figures_v2_2.py  —  V2.2.1 hotfix

Path fix:
  OUT_FIG  = ROOT / "figures_v2_2"
  OUT_DATA = ROOT / "outputs_v2_2" / "figures_data"

No reference to outputs_v2_1 or figures_v2_1 anywhere.
Reads from outputs_v2_2/*.csv written by the V2.2 scripts.

Run: python code_v2_2/generate_figures_v2_2.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy.interpolate import PchipInterpolator
from simulation_v2_2 import (ModelConfig, HyperParameters, Scenario,
                               generate_population, simulate_scenario)

ROOT     = Path(__file__).resolve().parents[1]
OUT_FIG  = ROOT / "figures_v2_2"
OUT_DATA = ROOT / "outputs_v2_2" / "figures_data"
OUT_FIG.mkdir(exist_ok=True)
OUT_DATA.mkdir(exist_ok=True)

# Load outputs written by V2.2 scripts
df   = pd.read_csv(ROOT / "outputs_v2_2" / "scenario_summary_v2_2.csv")
psa  = pd.read_csv(ROOT / "outputs_v2_2" / "psa_parametric_summary_v2_2.csv")
sens = pd.read_csv(ROOT / "outputs_v2_2" / "sensitivity_oneway_v2_2.csv")

DUR   = 12
DOSES = [0, 75, 150, 225, 300, 375, 450]
COMPS = [0.0, 0.25, 0.50, 0.75]
PAL   = ['#1a9850', '#91CF60', '#FC8D59', '#D73027']
CLABS = ['0% compensation', '25% compensation', '50% compensation', '75% compensation']


def clean(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9, length=0)


# ── Figure 1: Dose-response + PSA bands ──────────────────────────────────────
rows1 = []
fig1, ax1 = plt.subplots(figsize=(11, 7))
fig1.patch.set_facecolor('white'); ax1.set_facecolor('white')
for comp, col, lab in zip(COMPS, PAL, CLABS):
    sub  = df[(df.duration == DUR) & (df.comp == comp)].sort_values('dose')
    psub = psa[(psa.duration == DUR) & (psa.comp == comp)].sort_values('dose')
    ax1.plot(sub.dose, sub.mean_fm, '-o', color=col, lw=2.2, ms=7, label=lab, zorder=4)
    if len(psub):
        ax1.fill_between(psub.dose, psub.psa_l, psub.psa_u, color=col, alpha=0.15, zorder=2)
    for _, r in sub.iterrows():
        pm = psub[psub.dose == r.dose]
        rows1.append({'comp': comp, 'dose': r.dose, 'mean_fm': r.mean_fm,
                      'psa_l': pm['psa_l'].values[0] if len(pm) else np.nan,
                      'psa_u': pm['psa_u'].values[0] if len(pm) else np.nan})
pd.DataFrame(rows1).to_csv(OUT_DATA / 'Figure1_data.csv', index=False)
ax1.set_xlabel('Weekly aerobic exercise volume (min/week)', fontsize=12)
ax1.set_ylabel('Mean predicted fat-mass loss (kg)', fontsize=12)
ax1.set_xticks(DOSES); ax1.legend(fontsize=10, frameon=False); clean(ax1)
ax1.set_title(
    'Figure 1.  Dose-response: mean fat-mass loss at 12 weeks\n'
    '(shaded = 95% parametric PSA uncertainty; n=500 outer × 2,000 inner; exported at 600 dpi)',
    fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(OUT_FIG / 'Figure1_DoseResponse_PSA.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(); print("Figure 1 saved.")

# ── Figure 2: Individual distributions ───────────────────────────────────────
cfg2 = ModelConfig(n=3000); hp2 = HyperParameters()
DCOLS = ['#ABDDA4', '#66C2A5', '#3288BD', '#5E4FA2']
DLABS = ['75 min/wk', '150 min/wk', '300 min/wk', '450 min/wk']
rows2 = []
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9)); fig2.patch.set_facecolor('white')
for ci, (comp, clab) in enumerate(zip(COMPS, CLABS)):
    ax = axes2[ci // 2][ci % 2]; ax.set_facecolor('white')
    for dose, dc, dl in zip([75, 150, 300, 450], DCOLS, DLABS):
        rng = np.random.default_rng(42)
        pop = generate_population(cfg2, hp2, rng)
        out = simulate_scenario(pop, Scenario(dose, DUR, comp), cfg2, hp2, rng)
        fm  = out['delta_fm'].values
        ax.hist(fm, bins=45, color=dc, alpha=0.55, edgecolor='none', density=True, label=dl)
        ax.axvline(np.mean(fm), color=dc, lw=1.8, ls='--', alpha=0.9)
        for v in fm:
            rows2.append({'comp': comp, 'dose': dose, 'delta_fm': float(v)})
    ax.set_title(clab, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('Fat-mass loss (kg)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_xlim(-0.2, 4.0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(labelsize=8, length=0)
pd.DataFrame(rows2).to_csv(OUT_DATA / 'Figure2_data.csv', index=False)
handles = [mpatches.Patch(color=c, alpha=0.7, label=l) for c, l in zip(DCOLS, DLABS)]
fig2.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
            frameon=False, bbox_to_anchor=(0.5, -0.01))
fig2.suptitle('Figure 2.  Individual fat-mass loss distributions at 12 weeks\n'
              '(dashed lines = group means; n=3,000 per scenario)',
              fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT_FIG / 'Figure2_IndividualDistributions.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(); print("Figure 2 saved.")

# ── Figure 3: Low-response probability ───────────────────────────────────────
df12 = df[df.duration == DUR].copy()
active = [75, 150, 225, 300, 375, 450]
x = np.arange(len(active)); width = 0.18; offsets = [-1.5, -0.5, 0.5, 1.5]
rows3 = []
fig3, ax3 = plt.subplots(figsize=(11, 7)); fig3.patch.set_facecolor('white'); ax3.set_facecolor('white')
for ci, (comp, col, lab) in enumerate(zip(COMPS, PAL, CLABS)):
    sub  = df12[df12.comp == comp].set_index('dose')
    vals = [sub.loc[d, 'low_response_prop'] * 100 for d in active]
    ax3.bar(x + offsets[ci]*width, vals, width=width*0.85, color=col, alpha=0.85,
            label=lab, edgecolor='white')
    for d, v in zip(active, vals):
        rows3.append({'comp': comp, 'dose': d, 'low_resp_pct': v})
pd.DataFrame(rows3).to_csv(OUT_DATA / 'Figure3_data.csv', index=False)
ax3.axhline(10, color='#333', lw=1.2, ls='--', alpha=0.6, label='10% reference')
ax3.set_xticks(x); ax3.set_xticklabels([f'{d}\nmin/wk' for d in active], fontsize=10)
ax3.set_ylabel('Low-response probability (%)  [ΔFM < 0.5 kg]', fontsize=11)
ax3.set_ylim(0, 105); ax3.legend(fontsize=9, frameon=False, loc='upper right'); clean(ax3)
ax3.set_title('Figure 3.  Low-response probability at 12 weeks by dose and compensation',
              fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(OUT_FIG / 'Figure3_LowResponse.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(); print("Figure 3 saved.")

# ── Figure 4: Marginal returns ────────────────────────────────────────────────
all_d = [0, 75, 150, 225, 300, 375, 450]
trans = list(zip(all_d[:-1], all_d[1:]))
midpts = np.array([(d1+d2)/2 for d1, d2 in trans], dtype=float)
inc_u  = np.array([d2-d1 for d1, d2 in trans], dtype=float) / 75
rows4 = []
fig4, ax4 = plt.subplots(figsize=(11, 7)); fig4.patch.set_facecolor('white'); ax4.set_facecolor('white')
for comp, col, lab in zip(COMPS, PAL, CLABS):
    sub = df12[df12.comp == comp].set_index('dose')
    nm  = np.array([(sub.loc[d2,'mean_fm']-(sub.loc[d1,'mean_fm'] if d1>0 else 0))/iu
                    for (d1,d2),iu in zip(trans,inc_u)])
    pch = PchipInterpolator(midpts, nm)
    xf  = np.linspace(midpts[0], midpts[-1], 400)
    lw  = 2.8 if comp == 0 else 2.0; ls = '--' if comp == 0 else '-'
    ax4.plot(xf, np.clip(pch(xf), 0, None), color=col, lw=lw, ls=ls, label=lab, zorder=4)
    ax4.scatter(midpts, nm, color=col, s=40, zorder=6, alpha=0.8)
    for m, v in zip(midpts, nm):
        rows4.append({'comp': comp, 'midpoint': m, 'norm_marginal': v})

sub_add = df12[df12.comp == 0.0].set_index('dose')
add_nm  = np.array([(sub_add.loc[d2,'mean_fm']-(sub_add.loc[d1,'mean_fm'] if d1>0 else 0))/iu
                    for (d1,d2),iu in zip(trans,inc_u)])
pch_a   = PchipInterpolator(midpts, add_nm)
xf2     = np.linspace(37, 412, 1000)
ax4.plot(xf2, np.clip(pch_a(xf2), 0, None)*0.5, color='#555', lw=1.3, ls=':',
         label='50% of additive model (reference threshold)')

c50    = df12[df12.comp == 0.50].set_index('dose')
c50_nm = np.array([(c50.loc[d2,'mean_fm']-(c50.loc[d1,'mean_fm'] if d1>0 else 0))/iu
                   for (d1,d2),iu in zip(trans,inc_u)])
pch_c50 = PchipInterpolator(midpts, c50_nm)
y_c50   = np.clip(pch_c50(xf2), 0, None)
y_th    = np.clip(pch_a(xf2)*0.5, 0, None)
sc_idx  = np.where(np.diff(np.sign(y_c50 - y_th)))[0]
if len(sc_idx):
    ix = xf2[sc_idx[0]]
    ax4.axvline(ix, color='#B2182B', lw=1.2, ls=':', alpha=0.6)
    ax4.text(ix+8, 0.01, f'Smoothed\ncrossing\n~{ix:.0f} min/wk\n(exploratory)',
             fontsize=7.5, color='#B2182B', style='italic', va='bottom')

pd.DataFrame(rows4).to_csv(OUT_DATA / 'Figure4_data.csv', index=False)
ax4.set_xlabel('Weekly aerobic exercise volume (min/week)', fontsize=11.5)
ax4.set_ylabel('Marginal fat-mass loss per\nadditional 75 min/week (kg)', fontsize=11.5)
ax4.set_xticks([37.5, 112.5, 187.5, 262.5, 337.5, 412.5])
ax4.set_xticklabels(['0→75','75→150','150→225','225→300','300→375','375→450'], fontsize=9)
ax4.legend(fontsize=9.5, frameon=False, loc='upper right'); clean(ax4)
ax4.set_title(
    'Figure 4.  Marginal fat-mass loss per 75 min/week at 12 weeks\n'
    '(PCHIP-smoothed curves; dots = simulation nodes; vertical line = exploratory smoothed crossing)',
    fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(OUT_FIG / 'Figure4_MarginalReturns.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(); print("Figure 4 saved.")

# ── Figure 5: Tornado ──────────────────────────────────────────────────────────
sp = sens.pivot(index='parameter', columns='level', values='relative_change_pct')
sp['range'] = (sp['high'] - sp['low']).abs()
sp = sp[sp['range'] > 0.01].sort_values('range', ascending=True)
sp.reset_index().to_csv(OUT_DATA / 'Figure5_data.csv', index=False)
fig5, ax5 = plt.subplots(figsize=(11, 6)); fig5.patch.set_facecolor('white'); ax5.set_facecolor('white')
y = np.arange(len(sp))
ax5.barh(y, sp['low'].values,  height=0.55, color='#4575B4', alpha=0.85, label='Low parameter value')
ax5.barh(y, sp['high'].values, height=0.55, color='#D73027', alpha=0.85, label='High parameter value')
ax5.set_yticks(y); ax5.set_yticklabels(sp.index, fontsize=10)
ax5.axvline(0, color='#333', lw=1.5, zorder=5)
ax5.set_xlabel('Change in mean fat-mass loss relative to base case (%)', fontsize=10.5)
ax5.legend(fontsize=9.5, frameon=False, loc='lower right')
for s in ['top', 'right', 'left']:
    ax5.spines[s].set_visible(False)
ax5.tick_params(labelsize=10, length=0)
ax5.set_title(
    'Figure 5.  One-way sensitivity analysis\n'
    'Largest single-parameter effects on mean fat-mass loss predictions\n'
    'Base case: 300 min/week, 50% compensation, 12 weeks — all values from simulation reruns',
    fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(OUT_FIG / 'Figure5_Tornado.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(); print("Figure 5 saved.")
print(f"\nAll figures saved to {OUT_FIG}")
print(f"All source data saved to {OUT_DATA}")
