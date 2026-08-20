"""
Train-vs-external domain-shift logistic classifier (diagnostic only).

Primary representation: subject-level 2185-D flattened EEG features, the same
representation used as twin model input and in Layer 5* PCA / external loaders.

Writes ONLY under models/validation/domain_shift/.
Does NOT modify ensemble, bootstrap, checkpoints, or manuscript.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.utils.feature_processor import flatten_features
from src.tuh.run_latent_probe_caueeg import theta_alpha_ratio
from src.validation.run_unconstrained_external_battery import (
    load_osf,
    load_padic,
    load_tuh,
)

SEED = 42
N_FOLDS = 5
N_BOOT = 2000
N_PERM_SANITY = 20
RANDOM_STATE = SEED


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_training_flats(root: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """66 training subjects → 2185-D flats (same construction as Layer 5 PCA)."""
    flats: List[np.ndarray] = []
    sids: List[str] = []
    groups: List[str] = []
    for g in ("AD", "HC"):
        psd = np.load(root / "data" / "eeg_features" / f"{g}_psd.npy")
        bp = np.load(root / "data" / "eeg_features" / f"{g}_band_powers.npy")
        coh = np.load(root / "data" / "eeg_features" / f"{g}_coherence.npy")
        plv = np.load(root / "data" / "eeg_features" / f"{g}_plv.npy")
        for i in range(psd.shape[0]):
            flats.append(
                flatten_features(psd[i], bp[i], coh[i], plv[i]).astype(np.float32)
            )
            sids.append(f"{g}_{i:03d}")
            groups.append(f"{g}_{i:03d}")  # one subject = one group
    return np.stack(flats), sids, np.asarray(groups)


def tuh_patient_id(sid: str) -> str:
    if "_s" in sid:
        return sid.rsplit("_s", 1)[0]
    return sid


def domain_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict:
    """
    Subject/group-level stratified CV. Scaler + LR fit only on train partition.
    Primary metric: ROC-AUC (fold mean/SD + pooled OOF).
    """
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n_unique_groups = int(len(np.unique(groups)))
    # Prefer StratifiedGroupKFold when groups matter (TUH multi-session).
    use_group = n_unique_groups < len(y)
    if use_group:
        splitter = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
        splits = list(splitter.split(X, y, groups))
        split_mode = "StratifiedGroupKFold"
    else:
        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
        splits = list(splitter.split(X, y))
        split_mode = "StratifiedKFold"

    fold_aucs: List[float] = []
    fold_bal: List[float] = []
    oof_prob = np.full(len(y), np.nan, dtype=np.float64)
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    leakage_flags: List[str] = []

    for fold, (tr, te) in enumerate(splits):
        g_tr = set(groups[tr].tolist())
        g_te = set(groups[te].tolist())
        overlap = g_tr & g_te
        if overlap:
            leakage_flags.append(f"fold{fold}:{sorted(overlap)[:5]}")
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
        oof_prob[te] = prob
        oof_pred[te] = pred
        fold_aucs.append(float(roc_auc_score(y[te], prob)))
        fold_bal.append(float(balanced_accuracy_score(y[te], pred)))

    if leakage_flags:
        raise RuntimeError(f"Subject/group leakage detected: {leakage_flags}")
    if np.any(~np.isfinite(oof_prob)):
        raise RuntimeError("Incomplete OOF coverage")

    pooled_auc = float(roc_auc_score(y, oof_prob))
    pooled_bal = float(balanced_accuracy_score(y, oof_pred))
    cm = confusion_matrix(y, oof_pred, labels=[0, 1]).tolist()

    # Subject-level bootstrap CI on OOF AUC (resample rows with replacement)
    rng = np.random.default_rng(random_state)
    boot_aucs = np.empty(N_BOOT, dtype=float)
    n = len(y)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], oof_prob[idx]
        if len(np.unique(yb)) < 2:
            boot_aucs[b] = np.nan
        else:
            boot_aucs[b] = float(roc_auc_score(yb, pb))
    n_invalid = int(np.sum(~np.isfinite(boot_aucs)))
    ci_low = float(np.nanpercentile(boot_aucs, 2.5))
    ci_high = float(np.nanpercentile(boot_aucs, 97.5))

    return {
        "split_mode": split_mode,
        "n_folds": n_folds,
        "n_rows": int(len(y)),
        "n_unique_groups": n_unique_groups,
        "n_train_class": int(np.sum(y == 0)),
        "n_external_class": int(np.sum(y == 1)),
        "mean_cv_auc": float(np.mean(fold_aucs)),
        "std_cv_auc": float(np.std(fold_aucs, ddof=1)),
        "per_fold_auc": fold_aucs,
        "mean_cv_balanced_accuracy": float(np.mean(fold_bal)),
        "std_cv_balanced_accuracy": float(np.std(fold_bal, ddof=1)),
        "pooled_oof_auc": pooled_auc,
        "pooled_oof_balanced_accuracy": pooled_bal,
        "oof_confusion_matrix_labels_0_1": cm,
        "bootstrap_oof_auc": {
            "n_boot": N_BOOT,
            "seed": random_state,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "boot_mean": float(np.nanmean(boot_aucs)),
            "boot_std": float(np.nanstd(boot_aucs, ddof=1)),
            "boot_median": float(np.nanmedian(boot_aucs)),
            "n_invalid": n_invalid,
        },
        "oof_prob": oof_prob,
        "oof_pred": oof_pred,
        "y": y,
        "groups": groups,
    }


def label_permutation_sanity(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_perm: int = N_PERM_SANITY,
    random_state: int = RANDOM_STATE,
) -> Dict:
    """
    Software sanity: permute domain labels at the group level (keeps within-group
    label consistency required by StratifiedGroupKFold); expect AUCs near 0.5.
    """
    rng = np.random.default_rng(random_state + 7)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    # One label per group from the original labels (groups are domain-pure)
    uniq, inv = np.unique(groups, return_inverse=True)
    group_y = np.zeros(len(uniq), dtype=int)
    for gi, g in enumerate(uniq):
        labs = np.unique(y[groups == g])
        if len(labs) != 1:
            raise RuntimeError(f"Group {g} has mixed domain labels {labs}")
        group_y[gi] = int(labs[0])

    aucs = []
    for i in range(n_perm):
        gy_perm = rng.permutation(group_y)
        y_perm = gy_perm[inv]
        use_group = len(uniq) < len(y)
        if use_group:
            splitter = StratifiedGroupKFold(
                n_splits=N_FOLDS, shuffle=True, random_state=random_state + i
            )
            splits = list(splitter.split(X, y_perm, groups))
        else:
            splitter = StratifiedKFold(
                n_splits=N_FOLDS, shuffle=True, random_state=random_state + i
            )
            splits = list(splitter.split(X, y_perm))
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            clf = LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=random_state
            )
            clf.fit(Xtr, y_perm[tr])
            oof[te] = clf.predict_proba(Xte)[:, 1]
        if len(np.unique(y_perm)) < 2 or np.any(~np.isfinite(oof)):
            continue
        aucs.append(float(roc_auc_score(y_perm, oof)))
    return {
        "n_perm": n_perm,
        "n_completed": len(aucs),
        "mean_pooled_oof_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "std_pooled_oof_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "per_perm_auc": aucs,
        "note": (
            "Group-level label-permutation software sanity check; "
            "not a formal null hypothesis test."
        ),
    }


def run_comparison(
    name: str,
    X_train: np.ndarray,
    train_sids: List[str],
    train_groups: np.ndarray,
    X_ext: np.ndarray,
    ext_sids: List[str],
    ext_groups: np.ndarray,
    representation: str,
) -> Dict:
    X = np.vstack([X_train, X_ext]).astype(np.float64)
    y = np.concatenate(
        [np.zeros(len(X_train), dtype=int), np.ones(len(X_ext), dtype=int)]
    )
    groups = np.concatenate([train_groups, ext_groups])
    sids = list(train_sids) + list(ext_sids)

    # Deduplicate group strings across domains with prefix to avoid accidental collision
    groups = np.asarray(
        [f"train::{g}" for g in train_groups]
        + [f"ext::{g}" for g in ext_groups]
    )

    cv = domain_cv(X, y, groups)
    sanity = label_permutation_sanity(X, y, groups)

    out = {
        "comparison": name,
        "representation": representation,
        "primary": True,
        "n_training": int(len(X_train)),
        "n_external": int(len(X_ext)),
        "n_unique_train_groups": int(len(np.unique(train_groups))),
        "n_unique_external_groups": int(len(np.unique(ext_groups))),
        "classifier": {
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 2000,
            "C": 1.0,
            "penalty": "l2",
            "preprocessing": "StandardScaler fit on train fold only",
            "hyperparameter_tuning": "none",
        },
        "cv": {
            k: v
            for k, v in cv.items()
            if k not in {"oof_prob", "oof_pred", "y", "groups"}
        },
        "label_permutation_sanity": sanity,
        "subject_ids": sids,
        "domain_labels": y.tolist(),
        "group_ids": groups.tolist(),
        "_arrays": {
            "oof_prob": cv["oof_prob"],
            "oof_pred": cv["oof_pred"],
            "y": cv["y"],
        },
    }
    return out


def make_figure(primary_rows: List[Dict], out_path: Path) -> None:
    labels = [r["comparison"].replace("Training vs ", "") for r in primary_rows]
    means = [r["cv"]["mean_cv_auc"] for r in primary_rows]
    sds = [r["cv"]["std_cv_auc"] for r in primary_rows]
    pooled = [r["cv"]["pooled_oof_auc"] for r in primary_rows]

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    x = np.arange(len(labels))
    ax.errorbar(
        x,
        means,
        yerr=sds,
        fmt="o",
        color="#2c5f8a",
        ecolor="#2c5f8a",
        capsize=4,
        markersize=7,
        label="Mean 5-fold CV AUC ± SD",
    )
    ax.scatter(
        x,
        pooled,
        marker="s",
        s=36,
        color="#c44e52",
        zorder=3,
        label="Pooled OOF AUC",
    )
    ax.axhline(0.5, color="gray", ls="--", lw=1.0, label="Chance (AUC = 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Domain-classification ROC-AUC")
    ax.set_xlabel("External cohort (vs training)")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Train vs external domain separability\n(2185-D features, logistic regression)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    root = _root()
    out_dir = root / "models" / "validation" / "domain_shift"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [
        out_dir / "domain_shift_results.json",
        out_dir / "domain_shift_summary.md",
        out_dir / "domain_shift_predictions.npz",
        out_dir / "domain_shift_auc_figure.png",
    ]:
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite {p}")

    X_train, train_sids, train_groups = load_training_flats(root)
    assert X_train.shape[1] == 2185
    assert len(X_train) == 66

    loaders = {
        "TUH": load_tuh,
        "OSF": load_osf,
        "P-ADIC": load_padic,
    }
    primary_rep = "raw_2185D_flattened_eeg_features"
    primary_why = (
        "Same 2185-D flattened PSD/band-power/coherence/PLV vector used as twin "
        "model input and by Layer 5* PCA / external loaders (load_tuh/osf/padic). "
        "Most directly relevant to the external-validation feature pipeline."
    )

    comparisons = []
    pred_arrays = {}

    for ext_name, loader in loaders.items():
        flats, _dis, sids, _labs = loader(root)
        flats = np.asarray(flats, dtype=np.float32)
        if flats.shape[1] != 2185:
            raise RuntimeError(f"{ext_name} feature dim {flats.shape[1]} != 2185")
        if ext_name == "TUH":
            groups = np.asarray([tuh_patient_id(s) for s in sids])
        else:
            groups = np.asarray(sids)
        comp_name = f"Training vs {ext_name}"
        print(f"=== {comp_name} ({primary_rep}) ===")
        row = run_comparison(
            comp_name,
            X_train,
            train_sids,
            train_groups,
            flats,
            sids,
            groups,
            primary_rep,
        )
        print(
            f"  mean CV AUC={row['cv']['mean_cv_auc']:.3f}±{row['cv']['std_cv_auc']:.3f} "
            f"OOF={row['cv']['pooled_oof_auc']:.3f} "
            f"CI[{row['cv']['bootstrap_oof_auc']['ci_low']:.3f},"
            f"{row['cv']['bootstrap_oof_auc']['ci_high']:.3f}] "
            f"perm_mean={row['label_permutation_sanity']['mean_pooled_oof_auc']:.3f}"
        )
        key = ext_name.lower().replace("-", "")
        pred_arrays[f"{key}_oof_prob"] = row["_arrays"]["oof_prob"]
        pred_arrays[f"{key}_oof_pred"] = row["_arrays"]["oof_pred"]
        pred_arrays[f"{key}_y"] = row["_arrays"]["y"]
        del row["_arrays"]
        comparisons.append(row)

        # Exploratory sensitivity: theta/alpha scalar (existing probe feature)
        print(f"=== {comp_name} (exploratory theta/alpha) ===")
        Xtr_ta = theta_alpha_ratio(X_train).reshape(-1, 1)
        Xex_ta = theta_alpha_ratio(flats).reshape(-1, 1)
        row_ta = run_comparison(
            comp_name,
            Xtr_ta,
            train_sids,
            train_groups,
            Xex_ta,
            sids,
            groups,
            "theta_alpha_ratio_1D_exploratory",
        )
        row_ta["primary"] = False
        print(
            f"  mean CV AUC={row_ta['cv']['mean_cv_auc']:.3f}±{row_ta['cv']['std_cv_auc']:.3f} "
            f"OOF={row_ta['cv']['pooled_oof_auc']:.3f}"
        )
        pred_arrays[f"{key}_theta_alpha_oof_prob"] = row_ta["_arrays"]["oof_prob"]
        del row_ta["_arrays"]
        comparisons.append(row_ta)

    versions = {}
    try:
        import sklearn
        import scipy

        versions = {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        }
    except Exception:
        pass

    primary_rows = [c for c in comparisons if c.get("primary")]
    fig_path = out_dir / "domain_shift_auc_figure.png"
    make_figure(primary_rows, fig_path)

    results = {
        "timestamp": datetime.now().isoformat(),
        "script": "src/validation/run_domain_shift_classifier.py",
        "experiment": "train_vs_external_domain_logistic",
        "diagnostic_only": True,
        "does_not_modify_primary_ensemble": True,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_boot_auc_ci": N_BOOT,
        "n_perm_sanity": N_PERM_SANITY,
        "primary_representation": primary_rep,
        "primary_representation_rationale": primary_why,
        "exploratory_representations": [
            "theta_alpha_ratio_1D_exploratory (same scalar used in CAUEEG feature probe)"
        ],
        "not_run": [
            "latent mu: requires encoding all cohorts through frozen twin; deferred to keep this run cheap and representation provenance clear",
        ],
        "library_versions": versions,
        "training_feature_source": "data/eeg_features/{AD,HC}_*.npy via flatten_features",
        "external_loaders": {
            "TUH": "src.validation.run_unconstrained_external_battery.load_tuh",
            "OSF": "load_osf (features_calibrated)",
            "P-ADIC": "load_padic",
        },
        "leakage_controls": [
            "StratifiedGroupKFold when multi-row subjects exist (TUH patient id)",
            "StandardScaler fit on training fold only",
            "No pharmacodynamic outcomes or disease labels as features",
            "No use of domain AUC for model/fold/ensemble selection",
        ],
        "comparisons": comparisons,
        "audit": {
            "A_no_subject_leakage": True,
            "B_preprocess_train_fold_only": True,
            "C_no_external_labels_in_features": True,
            "D_no_pharmacodynamic_outcome": True,
            "E_no_fold_selection_by_domain_auc": True,
            "F_no_primary_ensemble_modification": True,
            "G_no_training_signature_modification": True,
            "H_no_manuscript_modification": True,
            "I_all_tested_representations_reported": True,
            "J_no_cherry_picking": True,
        },
    }

    # Strip large subject id lists from JSON? Keep for auditability - user asked for
    # subject IDs or auditable mapping. Keep them.

    json_path = out_dir / "domain_shift_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "domain_shift_predictions.npz", **pred_arrays)

    lines = [
        "# Domain-shift classifier (train vs external)",
        "",
        f"**Primary representation:** `{primary_rep}`",
        "",
        primary_why,
        "",
        "Classifier: LogisticRegression (balanced, L2, C=1), StandardScaler per fold, "
        f"{N_FOLDS}-fold subject/group-stratified CV. Seed={SEED}.",
        "",
        "| Comparison | Representation | Mean CV AUC | SD | Pooled OOF AUC | "
        "OOF AUC 95% CI | n training | n external |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in comparisons:
        boot = c["cv"]["bootstrap_oof_auc"]
        lines.append(
            f"| {c['comparison']} | {c['representation']} | "
            f"{c['cv']['mean_cv_auc']:.3f} | {c['cv']['std_cv_auc']:.3f} | "
            f"{c['cv']['pooled_oof_auc']:.3f} | "
            f"{boot['ci_low']:.3f}–{boot['ci_high']:.3f} | "
            f"{c['n_training']} | {c['n_external']} |"
        )
    lines.extend(
        [
            "",
            f"Figure: `{fig_path.name}`",
            "",
            "## Label-permutation sanity (primary 2185-D)",
            "",
        ]
    )
    for c in primary_rows:
        s = c["label_permutation_sanity"]
        lines.append(
            f"- {c['comparison']}: mean perm OOF AUC = "
            f"{s['mean_pooled_oof_auc']:.3f} ± {s['std_pooled_oof_auc']:.3f} "
            f"(n_perm={s['n_completed']})"
        )
    lines.extend(
        [
            "",
            "## Interpretation note",
            "",
            "Diagnostic only. Domain AUC does not establish that domain shift "
            "caused differences in twin direction or diagnostic performance.",
            "",
        ]
    )
    (out_dir / "domain_shift_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"[wrote] {json_path}")
    print(f"[wrote] {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
