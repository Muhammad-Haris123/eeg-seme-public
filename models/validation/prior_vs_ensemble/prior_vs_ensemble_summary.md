# Prior-only vs five-fold ensemble incremental-value analysis

Analysis of **existing** ensemble subject-bootstrap distributions vs the established prior-only scalar. No retraining. No manuscript edits.

Prior-only r (exact stored): **0.636017** (reported 0.636)
Bootstrap: B=2000, seed=42, unit=subject

## Comparison table

| Cohort | Ensemble r | Prior r | Difference | Fraction bootstrap > prior | Empirical percentile of prior |
| ------ | ---------: | ------: | ---------: | -------------------------: | ----------------------------: |
| TUH | 0.642081 | 0.636017 | +0.006064 | 539/2000 (0.2695) | 73.05 |
| OSF | 0.625967 | 0.636017 | -0.010050 | 0/2000 (0.0000) | 100.00 |
| PADIC | 0.626693 | 0.636017 | -0.009324 | 1/2000 (0.0005) | 99.95 |

## Paired analysis

**Not performed.** The prior-only baseline is a FIXED scalar: literature prior effect vectors (get_eeg_effect_prior) correlated with the FIXED constrained training signature (models/simulations_constrained_full). It does not produce subject-level prior-only effect vectors for external TUH/OSF/P-ADIC subjects. The ensemble bootstrap resamples external subjects and recomputes cohort-mean ensemble effects. A paired bootstrap of Delta r = r_ensemble - r_prior under the same subject resampling is therefore not statistically supportable from stored artifacts without reconstructing a different prior-only statistic. The available stored artifacts do not support a formally paired bootstrap comparison without reconstructing the subject-level prior-only statistic.

## Verdict

**Evidence does not clearly establish incremental agreement**

- Prior-only r is a fixed literature-vs-training-signature scalar, not a subject-resampled external statistic; no valid paired Delta r bootstrap from stored artifacts.
- Point differences vs prior are small in magnitude (order 10^-2) and mixed in sign: TUH slightly above; OSF and P-ADIC slightly below.
- For all three cohorts, the fraction of ensemble bootstrap replicates with r > prior is low (TUH=0.270, OSF=0.000, P-ADIC=0.001), and prior sits at a high empirical percentile of each bootstrap distribution (near or above the upper tail).
- Existing ensemble 95% percentile CIs describe uncertainty in the ensemble correlation under subject resampling; they are not a formal paired test of ensemble vs prior. Not converting CI exclusion into a p-value.
- Fold-0 locked external r values remain substantially higher than the ensemble; this analysis does not erase five-fold sensitivity variance.

