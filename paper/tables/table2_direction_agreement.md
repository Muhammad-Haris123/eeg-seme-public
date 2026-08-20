# Table 2. Drug-response direction agreement

**Caption (place ABOVE the table, Elsevier):** Checkpoint-specific external signature concordance at the locked fold-0 seed-42 checkpoint. Band-level direction agreement is scored between each external cohort and the training-cohort constrained signature on five feature blocks × two drugs (Donepezil, Memantine; maximum 10/10). Correlation r is the Pearson correlation of mean drug-minus-baseline effect vectors (`effect_magnitude_correlation`). These values are not architecture-wide performance estimates; fold and training-seed sensitivity are reported in the Results. CAUEEG is not included in the locked 10/10 direction claim.

| Dataset | N | Direction agreement (fraction) | Correlation r | Replication status |
|---------|---|-------------------------------|---------------|--------------------|
| TUH | 200 | 10/10 | 0.908 | 1st external assessment |
| OSF | 92 | 10/10 | 0.851 | 2nd assessment (independent cohort) |
| P-ADIC | 145 | 10/10 | 0.918 | 3rd assessment (independent cohort, different acquisition) |

## Source notes

- Primary: `models/validation/complete_validation_report_v3.json`
- TUH: `layer5.cross_dataset`
- OSF: `layer5b_ad_labeled_external.cross_dataset` (mirror: `data/ad_labeled_external/validation/layer5b_results.json`)
- P-ADIC: `layer5d_padic_external.cross_dataset` (mirror: `models/validation/layer5d_padic_results.json`)
