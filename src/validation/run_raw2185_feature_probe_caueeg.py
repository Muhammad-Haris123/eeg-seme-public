"""
CAUEEG Dementia vs Normal probe on full 2185-D features.

Same 5-fold stratified CV protocol as the theta/alpha and latent probes.
Writes models/validation/layer5e_caueeg_raw2185_probe_results.json
without overwriting prior probe JSON.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tuh.run_layer5e_caueeg import load_processed_features
from src.tuh.run_latent_probe_caueeg import (
    LABEL_TO_INT,
    probe_binary_cv,
    probe_binary_insample,
    theta_alpha_ratio,
)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "caueeg_external"
    flats, metas = load_processed_features(data_dir)
    labels = np.asarray([LABEL_TO_INT[m["label"]] for m in metas], dtype=np.int64)
    mask = (labels == 0) | (labels == 2)
    y = (labels[mask] == 2).astype(np.int64)
    X = flats[mask].astype(np.float64)
    ta = theta_alpha_ratio(flats)[mask].reshape(-1, 1)

    print(f"n={len(y)} pos={y.sum()} neg={(y==0).sum()} feat_dim={X.shape[1]}")
    raw_cv = probe_binary_cv(X, y)
    raw_in = probe_binary_insample(X, y)
    ta_cv = probe_binary_cv(ta, y)
    print(
        f"2185-D CV AUC={raw_cv['roc_auc']['mean']:.3f}±{raw_cv['roc_auc']['std']:.3f} "
        f"OOF={raw_cv['oof']['roc_auc']:.3f}"
    )
    print(
        f"theta/alpha CV AUC={ta_cv['roc_auc']['mean']:.3f}±{ta_cv['roc_auc']['std']:.3f} "
        f"OOF={ta_cv['oof']['roc_auc']:.3f}"
    )

    out = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "caueeg_raw2185_feature_logistic_probe",
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "feature_dim": int(X.shape[1]),
        "protocol": "StratifiedKFold(5), StandardScaler per fold, LogisticRegression balanced",
        "raw2185_logistic_dementia_vs_normal": {
            "cv": raw_cv,
            "in_sample_not_held_out": raw_in,
        },
        "theta_alpha_recheck": {"cv": ta_cv},
        "comparison_note": (
            "Compare raw2185 CV mean AUC to latent probe 0.579 and theta/alpha 0.675 "
            "from layer5e_caueeg_latent_probe_results.json."
        ),
    }
    path = root / "models" / "validation" / "layer5e_caueeg_raw2185_probe_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = root / "models" / "validation" / "layer5e_caueeg_raw2185_probe_summary.md"
    md.write_text(
        "# CAUEEG raw 2185-D feature probe (Dementia vs Normal)\n\n"
        f"- n = {out['n']} (Dem {out['n_pos']} / Norm {out['n_neg']})\n"
        f"- **2185-D logistic CV mean ROC-AUC:** {raw_cv['roc_auc']['mean']:.3f} "
        f"± {raw_cv['roc_auc']['std']:.3f} (OOF {raw_cv['oof']['roc_auc']:.3f})\n"
        f"- **theta/alpha recheck CV mean ROC-AUC:** {ta_cv['roc_auc']['mean']:.3f} "
        f"(OOF {ta_cv['oof']['roc_auc']:.3f})\n",
        encoding="utf-8",
    )
    print(f"[wrote] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
