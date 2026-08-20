# Five-fold ensemble subject-level bootstrap (B=2000)

Script: `src/validation/run_ensemble_subject_bootstrap.py`
Seed: 42; B: 2000; method: percentile 95% CI
Signature: `C:\Users\UC\Desktop\my_fyp\models\simulations_constrained_full`

| Cohort | n | r | 95% CI | Cosine | 95% CI |
| ------ | --: | --: | -----: | -----: | -----: |
| TUH | 200 | 0.642081 | 0.617877 to 0.644729 | 0.668654 | 0.631050 to 0.686110 |
| OSF | 92 | 0.625967 | 0.584414 to 0.623890 | 0.654660 | 0.592014 to 0.675637 |
| PADIC | 145 | 0.626693 | 0.595064 to 0.628936 | 0.647266 | 0.598259 to 0.668658 |

Bootstrap median / SD (r):
- **tuh**: median=0.631709; SD=0.006793; invalid=0
- **osf**: median=0.606189; SD=0.010094; invalid=0
- **padic**: median=0.613150; SD=0.008687; invalid=0
