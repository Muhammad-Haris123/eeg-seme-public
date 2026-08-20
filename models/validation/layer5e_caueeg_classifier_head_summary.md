# Layer 5e — Stronger classifier heads on frozen CVAE latents

Read-only. Twin **not** retrained. Nested CV (outer 5 / inner 3). Does not overwrite prior probe JSON.

## References

- Original latent LogReg CV mean AUC: **0.579** (OOF 0.576)
- Theta/alpha feature probe CV mean AUC: **0.675** (OOF 0.673)

## Results table

| Model | OOF ROC-AUC | Balanced Acc | vs LogReg baseline (0.579) | vs theta/alpha (0.675) |
|-------|-------------|--------------|----------------------------|-------------------------|
| tuned_logreg_l2 | 0.593 | 0.573 | +0.017 | -0.080 |
| mlp | 0.543 | 0.543 | -0.033 | -0.130 |
| hist_gradient_boosting | 0.556 | 0.548 | -0.020 | -0.117 |
| tuned_logreg_l2 + logvar (control) | 0.587 | 0.554 | +0.011 | -0.085 |

**Best head:** `tuned_logreg_l2` — OOF AUC **0.593**

## Permutation (best head)
```json
{
  "head": "tuned_logreg_l2",
  "n_permutations": 200,
  "true_oof_auc": 0.5928544405561336,
  "null_auc_mean": 0.5002915839086983,
  "null_auc_std": 0.02993284500667492,
  "null_auc_percentiles": {
    "p5": 0.4465783520918062,
    "p50": 0.5022384060027114,
    "p95": 0.5463054478388347
  },
  "true_auc_percentile_vs_null": 100.0,
  "permutation_p_value": 0.004975124378109453
}
```

## mu + logvar control (best head family)
```json
{
  "roc_auc": 0.5874160597748984,
  "balanced_accuracy": 0.5541473564740376,
  "f1": 0.5030487804878049
}
```

OOF AUC with `[mu; logvar]`: **0.587** (Δ vs mu-only best: -0.005)

## Verdict

**ceiling reached, move to writing**

Best head OOF ROC-AUC stays near the original logistic regression latent probe (~0.57–0.62), not meaningfully beyond it. The ceiling was already found; information loss is at encoding, not readout. Do not pursue further downstream heads; report the ~0.579 latent probe as-is and move on to writing.

## Artifacts

- `models/validation/layer5e_caueeg_classifier_head_results.json`
- Latents reused: `data/caueeg_external/validation/baseline_latents.npz`
