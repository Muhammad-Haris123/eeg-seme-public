# Layer 5e — CVAE baseline latent linear probe (CAUEEG)

Read-only diagnostic. Twin **not** retrained. Does not modify `layer5e_caueeg_results.json`.

## Setup

- Checkpoint: `C:\Users\UC\Desktop\my_fyp\models\checkpoints_constrained\checkpoint_constrained.pt`
- N subjects: 1122 (Normal 436 / MCI 395 / Dementia 291)
- mu_base: zero-drug embedding; **disease_label fixed to 0.0** for all (no diagnosis leak into fusion)
- CV: 5-fold stratified; LogisticRegression `class_weight='balanced'`; StandardScaler fit per fold
- Permutations: 200

## Headline

- **Latent probe (Dem vs Norm) ROC-AUC:** 0.579 ± 0.047
- **Permutation p-value:** 0.0100 (true AUC percentile vs null: 99.5%)
- **Theta/alpha feature probe ROC-AUC:** 0.675 ± 0.028
- Beats permutation null (p < 0.05)? **YES**

## Dementia vs Normal (primary)

### Latent mu_base
```json
{
  "cv": {
    "roc_auc": {
      "mean": 0.5790672423869888,
      "std": 0.04658681634356796,
      "per_fold": [
        0.5452586206896552,
        0.5230859146697837,
        0.5822433610780817,
        0.6044391597304797,
        0.6403091557669441
      ]
    },
    "balanced_accuracy": {
      "mean": 0.5679874431043339,
      "std": 0.03798553266313474,
      "per_fold": [
        0.5481974921630094,
        0.5158776543931425,
        0.5890804597701149,
        0.5718390804597702,
        0.6149425287356322
      ]
    },
    "f1": {
      "mean": 0.49160755956767377,
      "std": 0.03479365805402931,
      "per_fold": [
        0.4666666666666667,
        0.453125,
        0.5043478260869565,
        0.4915254237288136,
        0.5423728813559322
      ]
    },
    "oof": {
      "roc_auc": 0.5759954601343044,
      "balanced_accuracy": 0.5679442920646931,
      "f1": 0.49081803005008345
    },
    "n": 727,
    "n_pos": 291,
    "n_neg": 436
  },
  "in_sample_not_held_out": {
    "roc_auc": 0.778484504555629,
    "balanced_accuracy": 0.7174761184148302,
    "f1": 0.670926517571885,
    "note": "in-sample, not held-out"
  }
}
```

### Theta/alpha ratio (feature reference)
```json
{
  "cv": {
    "roc_auc": {
      "mean": 0.6752999217675048,
      "std": 0.02754639392404103,
      "per_fold": [
        0.661833855799373,
        0.6643288525228912,
        0.6462544589774079,
        0.6868806975822435,
        0.7172017439556084
      ]
    },
    "balanced_accuracy": {
      "mean": 0.6251981386040416,
      "std": 0.025923749977448913,
      "per_fold": [
        0.602076802507837,
        0.6474771089031754,
        0.6149425287356322,
        0.603448275862069,
        0.6580459770114943
      ]
    },
    "f1": {
      "mean": 0.5351193090323524,
      "std": 0.036086206741015335,
      "per_fold": [
        0.5,
        0.5739130434782609,
        0.5185185185185185,
        0.509090909090909,
        0.5740740740740741
      ]
    },
    "oof": {
      "roc_auc": 0.6728932185756172,
      "balanced_accuracy": 0.6252837416059775,
      "f1": 0.5355191256830601
    },
    "n": 727,
    "n_pos": 291,
    "n_neg": 436
  },
  "in_sample_not_held_out": {
    "roc_auc": 0.6750922160219427,
    "balanced_accuracy": 0.6264305305968032,
    "f1": 0.5364963503649635,
    "note": "in-sample, not held-out"
  }
}
```

## 3-class (Normal / MCI / Dementia)
```json
{
  "balanced_accuracy": {
    "mean": 0.3712026773666277,
    "std": 0.029858960990135908,
    "per_fold": [
      0.37134604446913483,
      0.3195283928514181,
      0.38593045249527136,
      0.3886463941025268,
      0.3905621029147874
    ]
  },
  "macro_f1": {
    "mean": 0.37060937360066076,
    "std": 0.029276591575733577,
    "per_fold": [
      0.3716296078623437,
      0.3200032064230302,
      0.3819069290941268,
      0.3880483438714785,
      0.3914587807523244
    ]
  },
  "oof_confusion_matrix": {
    "labels": [
      "Normal",
      "MCI",
      "Dementia"
    ],
    "matrix": [
      [
        196,
        124,
        116
      ],
      [
        137,
        136,
        122
      ],
      [
        92,
        106,
        93
      ]
    ]
  },
  "n": 1122,
  "n_by_class": {
    "Normal": 436,
    "MCI": 395,
    "Dementia": 291
  }
}
```

## Permutation null (latent Dem vs Norm)
```json
{
  "n_permutations": 200,
  "true_auc": 0.5790672423869888,
  "null_auc_mean": 0.5005520468622418,
  "null_auc_std": 0.0307285587277489,
  "null_auc_percentiles": {
    "p5": 0.44935496704585914,
    "p50": 0.4994870343838229,
    "p95": 0.5586371976430093
  },
  "true_auc_percentile_vs_null": 99.5,
  "permutation_p_value": 0.009950248756218905
}
```

## Interpretation (do not overclaim)

Latent probe beats the permutation null but **falls short** of the theta/alpha feature probe. Some signal survives encoding, but less than the raw spectral marker; both readout changes and milder encoder/constraint revisits remain open.

## Artifacts

- Latents: `C:\Users\UC\Desktop\my_fyp\data\caueeg_external\validation\baseline_latents.npz`
- Results JSON: `models/validation/layer5e_caueeg_latent_probe_results.json`
