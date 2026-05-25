"""
Figure generation script.
Produces all manuscript figures (1–8) and Table 1 from simulation outputs.

Run AFTER simulation_main.py has completed and outputs/sim_results.csv exists.

Usage:
    python generate_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.stats import truncnorm, beta as beta_dist
from scipy.interpolate import PchipInterpolator
import os

SEED = 42
np.random.seed(SEED)
N    = 10_000

os.makedirs('outputs/figures', exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────

def tnorm(mu, sd, lo, hi, size=N):
    a, b = (lo - mu) / sd, (hi - mu) / sd
    return truncnorm.rvs(a, b, loc=mu, scale=sd, size=size)

def beta_mv(mu, var, size=N):
    a = mu * (mu*(1-mu)/var - 1); b = (1-mu)*(mu*(1-mu)/var-1)
    return beta_dist.rvs(max(a,0.5), max(b,0.5), size=size)

def clean_ax(ax):
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(labelsize=9, length=0)

# ── Shared population ─────────────────────────────────────────────

sex    = np.random.binomial(1, 0.5, N)
age    = tnorm(46, 12, 25, 65)
bw     = np.where(sex==1, tnorm(90,18,55,160), tnorm(77,18,55,160))
bmi    = tnorm(33, 4.5, 27, 45)
bfp    = np.where(sex==1, tnorm(32,5.5,18,55), tnorm(42,5.5,22,58))
height = np.sqrt(bw/bmi)*100
rmr_b  = np.where(sex==1, 10*bw+6.25*height-5*age+5, 10*bw+6.25*height-5*age-161)
rmr    = np.clip(rmr_b * tnorm(1.0,0.10,0.75,1.30), 1000, 3500)
pal    = tnorm(1.40, 0.08, 1.25, 1.60)
fm0    = bw*bfp/100; ffm0 = bw-fm0
P_r    = np.clip(ffm0/(fm0+10.4), 0.05, 0.50)
ED_eff = P_r*1816 + (1-P_r)*9441
MET_i  = tnorm(4.5, 0.6, 3.0, 6.5)
AT     = {8:0.80, 12:1.00, 16:1.15, 24:1.35}

def sim_fm(dose, comp_mean, dur=12):
    if dose == 0: return np.zeros(N)
    sess_h = (dose/3)/60
    ExEE   = MET_i*bw*sess_h*3
    pen    = max(0,(dose-150)/150)*0.04; mu_a = max(0.60,0.82-pen)
    adh    = beta_mv(mu_a, 0.025)
    comp   = np.zeros(N) if comp_mean==0 else np.clip(beta_mv(comp_mean,0.032),0,0.99)
    comp   = np.clip(comp*AT[dur], 0, 0.98)
    deficit= ExEE*adh*(1-comp)*dur
    return (deficit/ED_eff)*(1-P_r)

# ─────────────────────────────────────────────────────────────────
# FIGURE 1 — Virtual population distributions
# ─────────────────────────────────────────────────────────────────

fig1 = plt.figure(figsize=(15,10)); fig1.patch.set_facecolor('white')
gs   = gridspec.GridSpec(2,3,figure=fig1,hspace=0.52,wspace=0.38)
C    = {'main':'#2166AC','male':'#4393C3','female':'#D6604D',
        'green':'#4DAC26','purple':'#762A83','red':'#B2182B'}

panels = [
    (gs[0,0], age,            C['main'],   'Age (years)',          'Age'),
    (gs[0,2], bmi,            C['green'],  'BMI (kg/m²)',          'BMI'),
    (gs[1,2], pal,            C['purple'], 'PAL (dimensionless)',  'Baseline PAL'),
]
for sp, data, col, xlabel, title in panels:
    ax = fig1.add_subplot(sp); ax.set_facecolor('white')
    ax.hist(data, bins=45, color=col, alpha=0.84, edgecolor='none', linewidth=0)
    ax.axvline(np.mean(data), color=C['red'], lw=2.0, ls='--',
               label=f'Mean={np.mean(data):.1f}')
    ax.set_xlabel(xlabel,fontsize=11); ax.set_ylabel('Count',fontsize=11)
    ax.set_title(title,fontsize=12,fontweight='bold',pad=8)
    ax.legend(fontsize=8.5,frameon=False); clean_ax(ax)

for sp, dm, df_, xlabel, title in [
    (gs[0,1], bw[sex==1],    bw[sex==0],    'Body weight (kg)',    'Body weight'),
    (gs[1,0], bfp[sex==1],   bfp[sex==0],   'Body fat (%)',        'Body fat %'),
    (gs[1,1], rmr[sex==1],   rmr[sex==0],   'RMR (kcal/day)',      'RMR'),
]:
    ax = fig1.add_subplot(sp); ax.set_facecolor('white')
    ax.hist(dm, bins=45, color=C['male'],   alpha=0.75, edgecolor='none', label='Male')
    ax.hist(df_,bins=45, color=C['female'], alpha=0.75, edgecolor='none', label='Female')
    ax.set_xlabel(xlabel,fontsize=11); ax.set_ylabel('Count',fontsize=11)
    ax.set_title(title,fontsize=12,fontweight='bold',pad=8)
    ax.legend(fontsize=8.5,frameon=False); clean_ax(ax)

fig1.suptitle('Figure 1.  Distributions of key parameters in the virtual population (n = 10,000)',
              fontsize=13,fontweight='bold',y=1.01)
plt.savefig('outputs/figures/Figure1_VirtualPopulation.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(); print("Figure 1 saved.")

# ─────────────────────────────────────────────────────────────────
# FIGURES 6, 7, 8 and TABLE 1 — from sim_results.csv
# ─────────────────────────────────────────────────────────────────

df = pd.read_csv('outputs/sim_results.csv')

doses4     = [75,150,300,450]
comps4     = [0.0,0.25,0.50,0.75]
dcols      = ['#ABDDA4','#66C2A5','#3288BD','#5E4FA2']
dlabs      = ['75 min/wk','150 min/wk','300 min/wk','450 min/wk']
comp_labs  = ['0% compensation','25% compensation','50% compensation','75% compensation']
comp_cols7 = ['#1a9850','#fee08b','#f46d43','#d73027']

# Figure 6 — distributions
fig6,axes6 = plt.subplots(2,2,figsize=(14,10)); fig6.patch.set_facecolor('white')
for ci,(comp,clab) in enumerate(zip(comps4,comp_labs)):
    ax = axes6[ci//2][ci%2]; ax.set_facecolor('white')
    for dose,dc,dl in zip(doses4,dcols,dlabs):
        fm = sim_fm(dose, comp)
        ax.hist(fm,bins=60,color=dc,alpha=0.55,edgecolor='none',density=True,label=dl)
        ax.axvline(np.mean(fm),color=dc,lw=2.0,ls='--',alpha=0.9)
    ax.set_title(clab,fontsize=12,fontweight='bold',pad=8)
    ax.set_xlabel('Individual fat-mass loss (kg)',fontsize=10.5)
    ax.set_ylabel('Probability density',fontsize=10.5)
    ax.set_xlim(-0.3,5.5); clean_ax(ax)

handles = [mpatches.Patch(color=c,alpha=0.7,label=l) for c,l in zip(dcols,dlabs)]
fig6.legend(handles=handles,loc='lower center',ncol=4,fontsize=10,
            frameon=False,bbox_to_anchor=(0.5,-0.01))
fig6.suptitle('Figure 6.  Distributions of individual fat-mass loss at 12 weeks\n'
              '(dashed lines = group means; n = 10,000 per scenario)',
              fontsize=12,fontweight='bold',y=1.02)
plt.tight_layout(rect=[0,0.04,1,1])
plt.savefig('outputs/figures/Figure6_Distributions.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(); print("Figure 6 saved.")

# Figure 7 — low responders
df12 = df[df.duration==12].copy()
fig7,ax7 = plt.subplots(figsize=(11,7)); fig7.patch.set_facecolor('white')
ax7.set_facecolor('white')
x = np.arange(len(doses4)); width=0.18; offsets=[-1.5,-0.5,0.5,1.5]

for ci,(comp,ccol,clab) in enumerate(zip(comps4,comp_cols7,comp_labs)):
    sub  = df12[df12.comp==comp].set_index('dose')
    nr   = [sub.loc[d,'nr_prop']*100  for d in doses4]
    ci_l = [sub.loc[d,'nr_ci_l']*100  for d in doses4]
    ci_u = [sub.loc[d,'nr_ci_u']*100  for d in doses4]
    ax7.bar(x+offsets[ci]*width, nr, width=width*0.85,
            color=ccol, alpha=0.85, label=clab, edgecolor='white', linewidth=0.5)
    ax7.errorbar(x+offsets[ci]*width, nr,
                 yerr=[[nr[i]-ci_l[i] for i in range(4)],
                       [ci_u[i]-nr[i] for i in range(4)]],
                 fmt='none', color='#333', capsize=3, lw=1.2)

ax7.axhline(10,color='#333',lw=1.2,ls='--',alpha=0.7,label='10% reference threshold')
ax7.set_xticks(x); ax7.set_xticklabels([f'{d} min/wk' for d in doses4],fontsize=11)
ax7.set_ylabel('Low responders (%)\n[fat-mass loss <0.5 kg]',fontsize=11)
ax7.set_ylim(0,105); ax7.legend(fontsize=9.5,frameon=False,loc='upper right')
for sp in ['top','right']: ax7.spines[sp].set_visible(False)
ax7.tick_params(labelsize=10,length=0)
ax7.set_title('Figure 7.  Proportion of low responders at 12 weeks\n'
              '(error bars: 95% Wilson confidence intervals)',
              fontsize=11,fontweight='bold',pad=12)
plt.tight_layout()
plt.savefig('outputs/figures/Figure7_LowResponders.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(); print("Figure 7 saved.")

# Figure 8 — marginal returns (PCHIP, normalised per 75 min/wk)
transitions = [(0,75),(75,150),(150,300),(300,450)]
midpoints   = np.array([37.5,112.5,225.0,375.0])
inc_units   = [1,1,2,2]
x_fine      = np.linspace(20,420,500)
fig8_cols   = ['#1a9850','#91CF60','#FC8D59','#D73027']
fig8_labs   = ['0% compensation (additive)','25% compensation',
               '50% compensation','75% compensation']

fig8,ax8 = plt.subplots(figsize=(12,7)); fig8.patch.set_facecolor('white'); ax8.set_facecolor('white')
for ci,(comp,ccol,clab) in enumerate(zip(comps4,fig8_cols,fig8_labs)):
    sub = df[(df.duration==12)&(df.comp==comp)].set_index('dose')
    nm  = np.array([(sub.loc[d2,'mean_fm']-(sub.loc[d1,'mean_fm'] if d1>0 else 0))/iu
                    for (d1,d2),iu in zip(transitions,inc_units)])
    pch = PchipInterpolator(midpoints, nm)
    y   = np.clip(pch(x_fine), 0, None)
    lw  = 2.8 if ci==0 else 2.2; ls = '--' if ci==0 else '-'
    ax8.plot(x_fine,y,color=ccol,lw=lw,ls=ls,label=clab,zorder=4+ci)
    ax8.scatter(midpoints,nm,color=ccol,s=45,zorder=6,alpha=0.7)

sub_add = df[(df.duration==12)&(df.comp==0.0)].set_index('dose')
add_nm  = np.array([(sub_add.loc[d2,'mean_fm']-(sub_add.loc[d1,'mean_fm'] if d1>0 else 0))/iu
                    for (d1,d2),iu in zip(transitions,inc_units)])
pch_a   = PchipInterpolator(midpoints, add_nm)
ax8.plot(x_fine, np.clip(pch_a(x_fine),0,None)*0.5, color='#555',
         lw=1.4, ls=':', label='50% of additive model (threshold)', zorder=3)
ax8.axvspan(225,280,alpha=0.10,color='#B2182B',zorder=2)
ax8.text(252,0.005,'Inflection\nzone\n225–280',ha='center',va='bottom',
         fontsize=8,color='#B2182B',style='italic')
ax8.set_xlabel('Weekly aerobic exercise volume (min/week)',fontsize=11.5)
ax8.set_ylabel('Marginal fat-mass loss per\nadditional 75 min/week (kg)',fontsize=11.5)
ax8.set_xlim(0,450); ax8.set_ylim(0,None)
ax8.set_xticks([0,75,150,225,300,375,450])
ax8.legend(fontsize=9.5,frameon=False,loc='upper right')
for sp in ['top','right']: ax8.spines[sp].set_visible(False)
ax8.tick_params(labelsize=10,length=0)
ax8.set_title('Figure 8.  Marginal fat-mass loss per additional 75 min/week at 12 weeks\n'
              '(PCHIP smoothing; dots = simulation nodes; values normalised per 75 min/wk increment)',
              fontsize=11,fontweight='bold',pad=12)
plt.tight_layout()
plt.savefig('outputs/figures/Figure8_MarginalReturns.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(); print("Figure 8 saved.")

print("\nAll figures saved to outputs/figures/")
