# V2.2.1 Final Pre-Manuscript Package
## More Cardio, Less Fat? A Probabilistic Simulation of Exercise-Induced Fat Loss Under Different Energy Compensation Scenarios

**Version:** V2.2.1 — Final pre-manuscript  
**Date:** May 2026  
**OSF Preregistration:** https://doi.org/10.17605/OSF.IO/TD2ZS  
**GitHub:** https://github.com/MrMuma19/cardio-fat-loss-simulation  

---

## Package contents

| Folder | Description |
|--------|-------------|
| `manuscript/` | Final Word manuscript with all 8 figures embedded |
| `figures/png/` | 8 clean figure PNG files (300 dpi) |
| `figures/source_data/` | CSV source data for each figure |
| `code/` | All Python simulation scripts (V2.2.1) |
| `outputs/` | Primary simulation outputs (CSVs, TXT) |
| `validation/` | External validation dataset |
| `parameters/` | Parameter dictionary, PSA hyperparameters, unresolved list |
| `environment/` | requirements.txt, environment.yml, session info |

---

## Key results (12 weeks, 50% compensation)

| Anchor | Mean ΔFM | PSA 95% | Additive | Attenuation |
|--------|----------|---------|----------|-------------|
| Primary (300 min/wk) | 0.629 kg | 0.496–0.760 kg | 1.247 kg | 49.5% |
| Secondary (450 min/wk) | 0.892 kg | 0.700–1.101 kg | 1.788 kg | 50.1% |

## Validation (three-tier)
- **Church et al. (preliminary primary):** RMSE = 0.313 kg
- **STRRIDE (sensitivity only):** RMSE = 0.953 kg  
- **Donnelly 2003/2013 (extended-duration comparison):** RMSE = 1.099 kg

## PRCC global sensitivity (top 5)
1. Compensation mean |PRCC| = 0.934
2. Adherence |PRCC| = 0.920
3. MET mean |PRCC| = 0.919
4. Adherence dose penalty |PRCC| = 0.638
5. MET SD |PRCC| = 0.172

NEAT cap and AT cap: |PRCC| < 0.02 (non-binding at studied doses)

---

## Execution order

```bash
conda env create -f environment/environment.yml
conda activate cardio-fat-loss-v2

python -m pytest code/test_model_v2_2.py -v
python code/simulation_v2_2.py
python code/psa_parametric_v2_2.py
python code/sensitivity_oneway_v2_2.py
python code/prcc_global_sensitivity_v2_2.py
python code/generate_figures_v2_2.py
python code/validation_external_v2_2.py
```

All scripts use `Path(__file__).resolve().parents[1]` — run from any directory.

---

## Key V2.2.1 changes from V2.1

1. **AT_SCALE removed** — duration multipliers (0.80/1.00/1.15/1.35) removed; no verifiable primary source
2. **PRCC global sensitivity** — n=2,000 LHS; comp mean, adherence, MET as dominant parameters
3. **Three-tier validation** — Church (preliminary primary), STRRIDE (sensitivity), Donnelly (extended-duration)
4. **112 scenarios** — 7 dose nodes × 4 comp × 4 durations (225 and 375 added for marginal-return analyses)
5. **Body fat % source** — CDC/NCHS NHANES DXA primary; Flegal 2010 secondary
6. **All figures regenerated** — 8 new clean figures from final outputs
