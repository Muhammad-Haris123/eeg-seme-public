# Table 4. Encoding analysis results (CAUEEG)

**Caption (place ABOVE the table, Elsevier):** CAUEEG Dementia vs Normal encoding analysis (n = 727). OOF ROC-AUC and balanced accuracy are out-of-fold aggregates. Permutation p-values are reported where computed (original latent probe: 200 label shuffles of 5-fold CV; best head: 200 shuffles of nested CV). The 2185-D logistic row is a raw feature-space diagnostic reference, not a CVAE result, not a theoretical upper bound, and not a supervised deep 2185-D classifier. Delta is OOF AUC minus the θ/α feature-probe OOF AUC (reference, not a ceiling). Pairs with Fig. 5.

| Model | Input representation | OOF ROC-AUC | Balanced accuracy | Permutation p-value | vs. theta/alpha reference |
|-------|---------------------|-------------|-------------------|---------------------|---------------------------|
| 2185-D logistic (feature-space reference) | packed 2185-D features | 0.699 | 0.642 | -- | +0.026 |
| θ/α feature probe (reference) | Band-power θ/α ratio | 0.673 | 0.625 | -- | +0.000 |
| Untuned logistic regression | μ_base | 0.576 | 0.568 | 0.010 | -0.097 |
| Tuned L2 logistic regression | μ_base | 0.593 | 0.573 | 0.005 | -0.080 |
| MLP | μ_base | 0.543 | 0.543 | -- | -0.130 |
| HistGradientBoosting | μ_base | 0.556 | 0.548 | -- | -0.117 |
| Tuned LogReg (μ+logvar control) | μ_base + logvar | 0.587 | 0.554 | -- | -0.085 |

## Source notes

- `models/validation/layer5e_caueeg_latent_probe_results.json`
- `models/validation/layer5e_caueeg_classifier_head_results.json`
- Reference column = feature_probe OOF ROC-AUC; deltas = row OOF AUC − that reference
- 2185-D row: `layer5e_caueeg_raw2185_probe_results.json` `raw2185_logistic_dementia_vs_normal.cv.oof`
