# More Cardio, Less Fat?
### A Probabilistic Simulation of Exercise-Induced Fat Loss Under Different Energy Compensation Scenarios

**Fit Generation Research Institute — Andorra la Vella, Andorra**  
Corresponding author: José Francisco Tornero-Aguilera — jtornero@fitgeneration.es

---

## Overview

This repository contains the complete simulation code for the preregistered probabilistic modelling study examining the dose-response relationship between aerobic exercise volume and fat-mass loss under different energy compensation scenarios.

**Preregistration:** OSF — [https://doi.org/10.17605/OSF.IO/TD2ZS]  
**Random seed:** 42 (fixed; do not modify)  
**Language:** Python ≥ 3.11

---

## Repository structure

```
├── simulation/
│   ├── simulation_main.py     # Core Monte Carlo simulation (run this first)
│   └── generate_figures.py    # Produces all manuscript figures
├── outputs/                   # Generated automatically on first run
│   ├── sim_results.csv        # Scenario-level statistics (80 scenarios)
│   ├── sim_raw.pkl            # Individual-level raw data
│   └── figures/               # All manuscript figures (PNG, 180 dpi)
├── requirements.txt
└── README.md
```

---

## How to reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the simulation

```bash
cd simulation
python simulation_main.py
```

This generates `outputs/sim_results.csv` and `outputs/sim_raw.pkl`.  
Expected runtime: approximately 3–8 minutes depending on hardware.

### 3. Generate figures

```bash
python generate_figures.py
```

This produces all manuscript figures in `outputs/figures/`.

---

## Simulation design

| Factor | Levels |
|---|---|
| Weekly exercise volume | 0, 75, 150, 300, 450 min/week |
| Energy compensation | 0%, 25%, 50%, 75% of ExEE |
| Intervention duration | 8, 12, 16, 24 weeks |
| **Total scenarios** | **80** |
| Individuals per scenario | 10,000 |

### Key model components

- **Population:** truncated normal distributions calibrated to NHANES 2017–18 and Jayedi et al. (2024)
- **RMR:** Mifflin-St Jeor equation with individual residual error (±10% SD)
- **Exercise EE:** MET × BW × duration; MET ~ TruncNormal(4.5, 0.6)
- **Adherence:** Beta distribution; mean decreases 0.04 per 150 min/wk above 150 min/wk
- **Compensation:** Beta distribution; variance = 0.032 (Riou et al. 2015)
- **Body composition:** Forbes-Hall P-ratio model (Hall 2007)
- **Energy density:** 9,441 kcal/kg (fat fraction); 1,816 kcal/kg (FFM fraction)

---

## Calibration and validation

The model was calibrated against Jayedi et al. (JAMA Network Open, 2024) and validated against:
- STRRIDE programme (Slentz et al. 2004, 2005)
- Church et al. 2009 (PLoS ONE)
- Midwest Exercise Trial

Acceptance criterion: ≥80% of validation trial estimates within the 95% simulation prediction interval.

---

## Citation

> Muñoz López M, Quesada Fernández G, Zabaleta Korta A, Sancho Haro ES, Baz Valle E,  
> Ramírez de la Piscina Viúdez X, López Gil JF, Tornero-Aguilera JF.  
> More Cardio, Less Fat? A Probabilistic Simulation of Exercise-Induced Fat Loss  
> Under Different Energy Compensation Scenarios. [Journal — under review]

---

## Author contributions
Conceptualization, M.M.-L., G.Q.-F. and E.S.S.-H.; methodology, M.M.-L., G.Q.-F., E.S.S.-H. and J.F.L.-G.; validation, G.Q.-F, E.S.S.-H., A.Z.-K. and E.B.-V.; formal analysis, M.M.-L., G.Q.-F., E.S.S.-H. and J.F.T.-A.; investigation, M.M.-L., A.Z.-K., E.B.-V. and X.,R-V.; resources, J.F.L.-G. and J.F.T.-A.; data curation, M.M.-L. G.Q.-F and E.S.S.-H.; writing—original draft preparation, M.M.L. and J.F.T.-A.; writing—review and editing, M.M.-L., G.Q.-F, E.S.S.-H., A.Z.-K., E.B.-V., X.,R-V., J.F.L.-G. and J.F.T.-A.; visualization, M.M.-L., G.Q.-F and J.F.T.-A.; supervision, J.F.L.-G and J.F.T.-A.; project administration, J.F.T.-A. All authors have read and agreed to the published version of the manuscript.

---

## License

This code is made available for scientific reproducibility purposes.  
Contact jtornero@fitgeneration.es for any questions.
