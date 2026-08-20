"""
Phase 1: re-extract subject-level OOF probe scores from frozen artifacts.

Does NOT retrain the CVAE. Does NOT overwrite locked probe JSON.

Reproduction gate: CV-mean AUCs must match locked artifacts within tolerance
before OOF vectors are stored.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tuh.run_latent_classifier_head_caueeg import INNER_FOLDS, _head_specs
from src.tuh.run_latent_probe_caueeg import LABEL_TO_INT, theta_alpha_ratio
from src.tuh.run_layer5e_caueeg import load_processed_features

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "models" / "validation" / "oof_probe_scores_phase1.json"
OUT_NPZ = ROOT / "models" / "validation" / "oof_probe_scores_phase1.npz"
SCRIPT_PATH = Path(__file__).resolve()

N_FOLDS = 5
RANDOM_STATE = 42
# Material disagreement threshold on CV-mean AUC.
TOL = 0.01

EXPECTED = {
    "latent_cv_mean": 0.5790672423869888,
    "theta_alpha_cv_mean": 0.6752999217675048,
    "packed2185_cv_mean": 0.6990767069310571,
    "latent_oof": 0.5759954601343044,
    "theta_alpha_oof": 0.6728932185756172,
    "packed2185_oof": None,  # filled from locked raw2185 JSON at runtime
}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    return str(obj)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def probe_cv_with_oof(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_metrics = []
    oof_prob = np.zeros(len(y), dtype=np.float64)
    oof_pred = np.zeros(len(y), dtype=np.int64)
    fold_id = np.full(len(y), -1, dtype=np.int64)

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        m = _binary_metrics(y[te], prob, pred)
        m["fold"] = fold
        fold_metrics.append(m)
        oof_prob[te] = prob
        oof_pred[te] = pred
        fold_id[te] = fold

    keys = ("roc_auc", "balanced_accuracy", "f1")
    summary = {
        k: {
            "mean": float(np.mean([f[k] for f in fold_metrics])),
            "std": float(np.std([f[k] for f in fold_metrics], ddof=1)),
            "per_fold": [float(f[k]) for f in fold_metrics],
        }
        for k in keys
    }
    summary["oof"] = _binary_metrics(y, oof_prob, oof_pred)
    summary["n"] = int(len(y))
    summary["n_pos"] = int(np.sum(y == 1))
    summary["n_neg"] = int(np.sum(y == 0))
    summary["oof_prob"] = oof_prob
    summary["oof_pred"] = oof_pred
    summary["fold_id"] = fold_id
    return summary


def nested_best_head_oof(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """Reproduce tuned_logreg_l2 nested CV and return OOF probabilities."""
    spec = _head_specs()["tuned_logreg_l2"]
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = np.zeros(len(y), dtype=np.float64)
    oof_pred = np.zeros(len(y), dtype=np.int64)
    fold_id = np.full(len(y), -1, dtype=np.int64)
    fold_metrics: List[Dict[str, Any]] = []
    best_params_per_fold: List[Dict] = []

    for fold, (tr, te) in enumerate(outer.split(X, y)):
        gs = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            scoring="roc_auc",
            cv=inner,
            n_jobs=1,
            refit=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X[tr], y[tr])
        proba = gs.predict_proba(X[te])[:, 1]
        pred = gs.predict(X[te])
        m = _binary_metrics(y[te], proba, pred)
        m["fold"] = fold
        m["best_params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()}
        fold_metrics.append(m)
        best_params_per_fold.append(m["best_params"])
        oof_prob[te] = proba
        oof_pred[te] = pred
        fold_id[te] = fold
        print(f"  [tuned_logreg_l2] fold {fold+1}/{N_FOLDS} AUC={m['roc_auc']:.3f} params={m['best_params']}")

    keys = ("roc_auc", "balanced_accuracy", "f1")
    summary: Dict[str, Any] = {
        "name": "tuned_logreg_l2",
        **{
            k: {
                "mean": float(np.mean([f[k] for f in fold_metrics])),
                "std": float(np.std([f[k] for f in fold_metrics], ddof=1)),
                "per_fold": [float(f[k]) for f in fold_metrics],
            }
            for k in keys
        },
    }
    summary["oof"] = _binary_metrics(y, oof_prob, oof_pred)
    summary["best_params_per_fold"] = best_params_per_fold
    summary["oof_prob"] = oof_prob
    summary["oof_pred"] = oof_pred
    summary["fold_id"] = fold_id
    return summary


def _check(name: str, expected: float, got: float) -> Dict[str, Any]:
    diff = float(got - expected)
    ok = abs(diff) <= TOL
    return {
        "name": name,
        "expected": expected,
        "reproduced": got,
        "difference": diff,
        "tolerance": TOL,
        "pass": ok,
    }


def main() -> int:
    raw2185 = json.loads(
        (ROOT / "models" / "validation" / "layer5e_caueeg_raw2185_probe_results.json").read_text(
            encoding="utf-8"
        )
    )
    EXPECTED["packed2185_oof"] = float(
        raw2185["raw2185_logistic_dementia_vs_normal"]["cv"]["oof"]["roc_auc"]
    )

    lat_path = ROOT / "data" / "caueeg_external" / "validation" / "baseline_latents.npz"
    if not lat_path.exists():
        print("STOP: missing frozen baseline_latents.npz")
        print(f"expected path: {lat_path}")
        return 2

    z = np.load(lat_path, allow_pickle=True)
    labels_all = np.asarray(z["labels"]).astype(np.int64)
    mask = np.asarray(z["dementia_vs_normal_mask"]).astype(bool)
    mu = np.asarray(z["mu_base"], dtype=np.float64)[mask]
    y = (labels_all[mask] == 2).astype(np.int64)
    sids_all = np.asarray(z["subject_ids"])
    sids = sids_all[mask]

    data_dir = ROOT / "data" / "caueeg_external"
    flats, metas = load_processed_features(data_dir)
    labels_feat = np.asarray([LABEL_TO_INT[m["label"]] for m in metas], dtype=np.int64)
    mask_feat = (labels_feat == 0) | (labels_feat == 2)
    y_feat = (labels_feat[mask_feat] == 2).astype(np.int64)
    X2185 = flats[mask_feat].astype(np.float64)
    ta = theta_alpha_ratio(flats)[mask_feat].reshape(-1, 1)
    sids_feat = np.asarray([m["recording_id"] for m in metas], dtype=object)[mask_feat]

    if len(y) != 727 or len(y_feat) != 727:
        print(f"STOP: expected n=727 Dementia vs Normal; got latent {len(y)} features {len(y_feat)}")
        return 2
    sids_str = np.asarray(sids, dtype=str)
    sids_feat_str = np.asarray(sids_feat, dtype=str)
    if set(sids_str) != set(sids_feat_str):
        print("STOP: subject ID sets differ between latents and packed features")
        return 2
    if not np.array_equal(sids_str, sids_feat_str):
        feat_index = {sid: i for i, sid in enumerate(sids_feat_str)}
        order = np.array([feat_index[sid] for sid in sids_str], dtype=np.int64)
        X2185 = X2185[order]
        ta = ta[order]
        y_feat = y_feat[order]
        sids_feat = sids_feat[order]
        print("[oof] aligned packed features to latent subject_id order")
    if not np.array_equal(y, y_feat):
        print("STOP: latent mask labels do not match packed-feature labels after ID alignment")
        return 2

    print(f"[oof] n={len(y)} pos={int(y.sum())} neg={int((y == 0).sum())}")
    print("[oof] reproducing latent / theta-alpha / packed-2185 probes")
    latent = probe_cv_with_oof(mu, y)
    theta = probe_cv_with_oof(ta, y)
    packed = probe_cv_with_oof(X2185, y)

    checks = [
        _check("latent_cv_mean", EXPECTED["latent_cv_mean"], latent["roc_auc"]["mean"]),
        _check("theta_alpha_cv_mean", EXPECTED["theta_alpha_cv_mean"], theta["roc_auc"]["mean"]),
        _check("packed2185_cv_mean", EXPECTED["packed2185_cv_mean"], packed["roc_auc"]["mean"]),
        _check("latent_oof", EXPECTED["latent_oof"], latent["oof"]["roc_auc"]),
        _check("theta_alpha_oof", EXPECTED["theta_alpha_oof"], theta["oof"]["roc_auc"]),
        _check("packed2185_oof", EXPECTED["packed2185_oof"], packed["oof"]["roc_auc"]),
    ]
    for c in checks:
        flag = "PASS" if c["pass"] else "FAIL"
        print(
            f"  [{flag}] {c['name']}: expected {c['expected']:.6f} "
            f"reproduced {c['reproduced']:.6f} diff={c['difference']:+.6f}"
        )

    failed = [c for c in checks if not c["pass"]]
    if failed:
        print("STOP: OOF reproduction disagrees materially with locked artifacts.")
        print("No OOF file written. Locked probe JSON untouched.")
        fail_path = ROOT / "models" / "validation" / "oof_probe_scores_phase1_REPRODUCTION_FAIL.json"
        fail_path.write_text(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "REPRODUCTION_FAILED_NO_OOF_STORED",
                    "checks": checks,
                    "failed": failed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[wrote failure report] {fail_path}")
        return 1

    print("[oof] reproduction OK; extracting nested tuned_logreg_l2 OOF (best stored head)")
    head = nested_best_head_oof(mu, y)
    print(
        f"  nested head OOF AUC={head['oof']['roc_auc']:.6f} "
        f"(locked best_head oof was 0.593)"
    )

    fold_id = latent["fold_id"]
    if not np.array_equal(fold_id, theta["fold_id"]) or not np.array_equal(fold_id, packed["fold_id"]):
        print("STOP: fold assignments differ across probes despite identical StratifiedKFold args")
        return 2

    subjects = []
    for i in range(len(y)):
        subjects.append(
            {
                "subject_id": str(sids[i]),
                "index": i,
                "fold": int(fold_id[i]),
                "true_label": int(y[i]),
                "label_name": "Dementia" if int(y[i]) == 1 else "Normal",
                "latent_probe_oof_probability": float(latent["oof_prob"][i]),
                "theta_alpha_oof_probability": float(theta["oof_prob"][i]),
                "packed_2185_oof_probability": float(packed["oof_prob"][i]),
                "best_nested_head_oof_probability": float(head["oof_prob"][i]),
                "best_nested_head_name": "tuned_logreg_l2",
            }
        )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "phase1_oof_probe_scores",
        "status": "NEW_NON_LOCKED",
        "delong_computed": False,
        "delong_note": (
            "OOF subject-level prediction vectors were successfully stored. "
            "DeLong was not computed in this session and is ready for a follow-up paired analysis."
        ),
        "n": int(len(y)),
        "n_pos_dementia": int(y.sum()),
        "n_neg_normal": int((y == 0).sum()),
        "cv_protocol": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "seed": RANDOM_STATE,
        "logistic_probe": {
            "class_weight": "balanced",
            "max_iter": 2000,
            "scaler": "StandardScaler fit on training fold only",
        },
        "source_artifacts": {
            "latents": str(lat_path),
            "caueeg_features": str(data_dir),
            "locked_latent_probe": "models/validation/layer5e_caueeg_latent_probe_results.json",
            "locked_raw2185_probe": "models/validation/layer5e_caueeg_raw2185_probe_results.json",
            "locked_classifier_head": "models/validation/layer5e_caueeg_classifier_head_results.json",
        },
        "script": str(SCRIPT_PATH),
        "reproduction_check": {
            "tolerance_abs_auc": TOL,
            "all_pass": True,
            "checks": checks,
        },
        "reproduced_summaries": {
            "latent": {
                "cv_mean_auc": latent["roc_auc"]["mean"],
                "oof_auc": latent["oof"]["roc_auc"],
                "per_fold_auc": latent["roc_auc"]["per_fold"],
            },
            "theta_alpha": {
                "cv_mean_auc": theta["roc_auc"]["mean"],
                "oof_auc": theta["oof"]["roc_auc"],
                "per_fold_auc": theta["roc_auc"]["per_fold"],
            },
            "packed_2185": {
                "cv_mean_auc": packed["roc_auc"]["mean"],
                "oof_auc": packed["oof"]["roc_auc"],
                "per_fold_auc": packed["roc_auc"]["per_fold"],
            },
            "best_nested_head_tuned_logreg_l2": {
                "cv_mean_auc": head["roc_auc"]["mean"],
                "oof_auc": head["oof"]["roc_auc"],
                "per_fold_auc": head["roc_auc"]["per_fold"],
                "best_params_per_fold": head["best_params_per_fold"],
            },
        },
        "best_nested_head_extracted": True,
        "best_nested_head_name": "tuned_logreg_l2",
        "npz_path": str(OUT_NPZ),
        "subjects": subjects,
    }

    np.savez_compressed(
        OUT_NPZ,
        subject_id=sids.astype(str),
        fold=fold_id,
        true_label=y,
        latent_probe_oof_probability=latent["oof_prob"],
        theta_alpha_oof_probability=theta["oof_prob"],
        packed_2185_oof_probability=packed["oof_prob"],
        best_nested_head_oof_probability=head["oof_prob"],
    )
    OUT_JSON.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"[wrote] {OUT_JSON}")
    print(f"[wrote] {OUT_NPZ}")
    print(payload["delong_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
