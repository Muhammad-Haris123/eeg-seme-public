# Multi-seed fold-0 robustness summary

Generated: 2026-08-11T00:03:46.630177

## Design

- Fixed `split_seed=42` (identical fold-0 subject split to locked training).
- Varied `train_seed` in `{7, 21, 123, 2024}`; seed `42` = locked reference (not retrained).
- Checkpoint rule: min internal validation total loss on fold 0.
- External scoring after freeze on TUH / OSF / P-ADIC vs fixed training signature.

## Seed × cohort

| Seed | Cohort | Agree | r | Cosine | Δr vs 42 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 42 | tuh | 10/10 | 0.908024 | 0.927613 | +0.000000 |
| 42 | osf | 10/10 | 0.851003 | 0.884206 | +0.000000 |
| 42 | padic | 10/10 | 0.917640 | 0.942843 | +0.000000 |
| 7 | tuh | 10/10 | 0.407892 | 0.592086 | -0.500131 |
| 7 | osf | 10/10 | 0.407638 | 0.588463 | -0.443365 |
| 7 | padic | 10/10 | 0.427859 | 0.608313 | -0.489781 |
| 21 | tuh | 10/10 | 0.154262 | 0.054284 | -0.753762 |
| 21 | osf | 10/10 | 0.159845 | 0.117509 | -0.691158 |
| 21 | padic | 8/10 | 0.105660 | -0.046439 | -0.811980 |
| 123 | tuh | 9/10 | 0.146906 | 0.025395 | -0.761118 |
| 123 | osf | 9/10 | 0.230114 | 0.209951 | -0.620889 |
| 123 | padic | 9/10 | 0.080451 | -0.124799 | -0.837189 |
| 2024 | tuh | 10/10 | 0.163031 | 0.018633 | -0.744992 |
| 2024 | osf | 10/10 | 0.182212 | 0.088491 | -0.668791 |
| 2024 | padic | 10/10 | 0.146115 | 0.016763 | -0.771525 |

## Aggregate

| Cohort | Mean r | Median r | SD | Min | Max | 10/10 seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tuh | 0.356023 | 0.163031 | 0.327520 | 0.146906 | 0.908024 | 4/5 |
| osf | 0.366162 | 0.230114 | 0.287969 | 0.159845 | 0.851003 | 4/5 |
| padic | 0.335545 | 0.146115 | 0.353963 | 0.080451 | 0.917640 | 3/5 |

## Interpretation

**Evidence suggests substantial training-seed sensitivity**

## Limitations

- Five seeds is a robustness check, not a full training-distribution characterization.
- Does not address fold dependence, prior-only baseline, or post-dose EEG absence.
- Does not prove historical seed-42 pre-registration.

