# Linear direction baseline (Phase 1 control; NEW, non-locked)
- Timestamp (UTC): 2026-08-11T20:09:47.340258+00:00
- Script: `C:\Users\UC\Desktop\my_fyp\src\validation\run_linear_direction_baseline.py`
- Targets: locked CVAE simulated Δx from `models/simulations_constrained_full` (not observed post-dose EEG).

## Comparison table (computed values only)
| Model | TUH r | TUH signs | OSF r | OSF signs | P-ADIC r | P-ADIC signs | Scored against |
|---|---:|---:|---:|---:|---:|---:|---|
| Prior-only (existing locked) | — | 10/10 | — | 10/10 | — | 10/10 | CVAE training signature |
| Linear, fold-0, primary | 0.851 | 10/10 | 0.639 | 10/10 | 0.775 | 10/10 | exact locked CVAE signature |
| Linear, fold-0, secondary (fit subjects) | 0.897 | 10/10 | 0.668 | 10/10 | 0.818 | 10/10 | own fitted signature (train_ids) |
| Linear, fold-0, secondary (full 66) | 0.897 | 10/10 | 0.671 | 10/10 | 0.824 | 10/10 | own fitted signature (all 66) |
| CVAE locked fold-0 seed-42 | 0.908 | 10/10 | 0.851 | 10/10 | 0.918 | 10/10 | own training signature |

Protocol-matched secondary comparator in the required table is **own fitted signature**.
The full-66 own-signature row is reported for CVAE-signature-construction analogy; it does not replace the locked-CVAE-signature comparison.

## Five-fold primary summary (locked CVAE signature)
- TUH: mean r = 0.883 (sd 0.027); per-fold r = [0.851, 0.881, 0.919, 0.9, 0.864]; signs = ['10/10', '10/10', '10/10', '10/10', '10/10']
- OSF: mean r = 0.787 (sd 0.110); per-fold r = [0.639, 0.804, 0.913, 0.859, 0.72]; signs = ['10/10', '8/10', '10/10', '10/10', '10/10']
- P-ADIC: mean r = 0.858 (sd 0.060); per-fold r = [0.775, 0.886, 0.919, 0.895, 0.817]; signs = ['10/10', '10/10', '10/10', '10/10', '10/10']

No superiority claim is written here. Compare the numbers. This control does not establish independent pharmacodynamic discovery.
