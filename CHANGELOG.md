# CHANGELOG — V2.2.1

## V2.1 → V2.2.1 (May 2026)

### Critical fixes
- AT_SCALE duration multipliers definitively removed (no primary source for exact values)
- PRCC global sensitivity analysis added (n=2,000 LHS)
- Validation restructured into three tiers (preliminary primary / sensitivity / extended-duration)
- 112 scenario combinations (7 doses × 4 comp × 4 durations); 225 and 375 min/wk added
- All figure source data provided alongside clean PNGs

### Parameter updates
- Body fat % source: CDC/NCHS NHANES DXA (primary); Flegal 2010 JAMA (secondary)
- PAL reclassified as technical parameter (NEAT cap construction only)
- Compensation split 50/30/20 confirmed as modelling assumption (split sensitivity: <0.01% effect)

### Validation updates
- Donnelly 2003/2013: reclassified as extended_duration_external_comparison
- STRRIDE: sensitivity_only (partial circularity with adherence parametrisation)
- Church et al.: preliminary_primary_validation; RMSE = 0.313 kg

### Code
- All scripts renamed *_v2_2.py, run from project root
- No outputs_v2_1 or figures_v2_1 directories created
- 6/6 unit tests pass
- Path(__file__).resolve().parents[1] throughout

### Manuscript
- Final Word document with 8 embedded clean figures (300 dpi)
- All V2.2.1 numerical values verified against outputs
- References reordered by first appearance; [37] Flegal 2010 added
- Supplementary Table S3 fully regenerated from scenario_summary_v2_2.csv
