"""
Domain-shift forensic characterization (diagnostic only).

Characterizes WHY train-vs-external separability is strong in 2185-D space:
feature-family AUCs, within-fold PCA sensitivity, coefficient concentration.

Writes ONLY under models/validation/domain_shift_forensics/.
Does NOT modify domain_shift/, ensemble, bootstrap, checkpoints, or manuscript.
Does NOT modify src/validation/run_domain_shift_classifier.py.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.utils.feature_processor import (
    BAND_SIZE,
    EEG_FLAT_SIZE,
    N_BANDS,
    N_CHANNELS,
    PSD_SIZE,
    TRIU_SIZE,
)
from src.validation.run_domain_shift_classifier import (
    load_training_flats,
    tuh_patient_id,
)
from src.validation.run_unconstrained_external_battery import (
    load_osf,
    load_padic,
    load_tuh,
)

SEED = 42
N_FOLDS = 5
RANDOM_STATE = SEED
EXPECTED_PRIMARY = {
    "TUH": 0.984,
    "OSF": 1.000,
    "P-ADIC": 1.000,
}
PRIMARY_ATOL = 0.015  # allow small float / fold-path noise vs locked OOF


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_feature_family_indices() -> Dict[str, np.ndarray]:
    """
    Index sets from flatten_features / prepare_target_features / unflatten_features.

    Connectivity is interleaved per band: [coh_triu(171), plv_triu(171)] × 5.
    Contiguous 'coherence_block' / 'plv_block' in feature_block_ranges() are NOT used.
    """
    families: Dict[str, List[int]] = {
        "psd": list(range(0, PSD_SIZE)),
        "band_powers": list(range(PSD_SIZE, PSD_SIZE + BAND_SIZE)),
        "connectivity_all": list(range(PSD_SIZE + BAND_SIZE, EEG_FLAT_SIZE)),
        "spectral_psd_plus_band_powers": list(range(0, PSD_SIZE + BAND_SIZE)),
        "coherence": [],
        "plv": [],
        "full_2185": list(range(EEG_FLAT_SIZE)),
    }
    i = PSD_SIZE + BAND_SIZE
    for _b in range(N_BANDS):
        families["coherence"].extend(range(i, i + TRIU_SIZE))
        i += TRIU_SIZE
        families["plv"].extend(range(i, i + TRIU_SIZE))
        i += TRIU_SIZE
    assert i == EEG_FLAT_SIZE
    assert len(families["coherence"]) == N_BANDS * TRIU_SIZE
    assert len(families["plv"]) == N_BANDS * TRIU_SIZE
    return {k: np.asarray(v, dtype=int) for k, v in families.items()}


def write_feature_mapping_md(path: Path, families: Dict[str, np.ndarray]) -> None:
    # Document interleaved connectivity ranges per band
    band_rows = []
    i = PSD_SIZE + BAND_SIZE
    for b in range(N_BANDS):
        coh = (i, i + TRIU_SIZE)
        i += TRIU_SIZE
        plv = (i, i + TRIU_SIZE)
        i += TRIU_SIZE
        band_rows.append((b, coh, plv))

    lines = [
        "# 2185-D feature dimension mapping",
        "",
        f"Total dimensionality: **{EEG_FLAT_SIZE}**",
        "",
        "## Source of truth",
        "",
        "Construction order is defined by:",
        "",
        "1. `api/utils/feature_processor.py` → `flatten_features`",
        "2. `src/models/train_phase2.py` → `prepare_target_features`",
        "3. `src/tuh/process_tuh_eeg.py` → `unflatten_features` (inverse)",
        "",
        "These three agree. Domain-shift forensics uses this ordering.",
        "",
        "## Contiguous major blocks (reliable)",
        "",
        "| Family | Dims | Half-open index range |",
        "| ------ | ---: | --------------------- |",
        f"| PSD (19×20) | {PSD_SIZE} | `[0, {PSD_SIZE})` |",
        f"| Band powers (19×5, channel-major C-order) | {BAND_SIZE} | `[{PSD_SIZE}, {PSD_SIZE + BAND_SIZE})` |",
        f"| Connectivity (coherence+PLV interleaved) | {N_BANDS * TRIU_SIZE * 2} | `[{PSD_SIZE + BAND_SIZE}, {EEG_FLAT_SIZE})` |",
        f"| **Total** | **{EEG_FLAT_SIZE}** | |",
        "",
        "## Connectivity interleaving (reliable)",
        "",
        "For each of 5 bands, flatten appends **coherence upper-triangle (171)** then "
        "**PLV upper-triangle (171)** (diagonal excluded; "
        f"171 = {N_CHANNELS}×{N_CHANNELS - 1}/2).",
        "",
        "| Band index | Coherence range | PLV range |",
        "| ----------: | --------------- | --------- |",
    ]
    for b, coh, plv in band_rows:
        lines.append(f"| {b} | `[{coh[0]}, {coh[1]})` | `[{plv[0]}, {plv[1]})` |")

    lines.extend(
        [
            "",
            f"| All coherence indices | {len(families['coherence'])} dims (non-contiguous union) |",
            f"| All PLV indices | {len(families['plv'])} dims (non-contiguous union) |",
            "",
            "## Documented but unreliable / unused here",
            "",
            "1. `feature_block_ranges()` in `api/utils/feature_processor.py` claims "
            "`coherence_block=[475,1330)` and `plv_block=[1330,2185)`. "
            "**This does not match** `flatten_features` interleaving (verified with "
            "synthetic markers: claimed coherence block mixes coh and PLV values).",
            "2. `BAND_INDICES` / contiguous delta–gamma slices in `src/models/losses.py` "
            "assume band-major contiguous channel packs; actual band_powers flatten is "
            "**channel-major** (`reshape(-1)` on `(19, 5)`). Those slices are **not** "
            "used for family classifiers in this forensic run.",
            "3. θ/α is a **derived scalar** (mean theta / mean alpha over channels), "
            "not a contiguous 2185 slice (already tested in the primary domain-shift run).",
            "",
            "## Families used in forensics",
            "",
        ]
    )
    for name, idx in families.items():
        lines.append(f"- `{name}`: {len(idx)} dimensions")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _splits(X, y, groups, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    use_group = len(np.unique(groups)) < len(y)
    if use_group:
        splitter = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
        return list(splitter.split(X, y, groups)), "StratifiedGroupKFold"
    splitter = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )
    return list(splitter.split(X, y)), "StratifiedKFold"


def domain_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    pca_dims: int | None = None,
    return_coefs: bool = False,
) -> Dict:
    """Scaler (+ optional PCA) + LR fit only on each training partition."""
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    splits, split_mode = _splits(X, y, groups)
    fold_aucs: List[float] = []
    oof_prob = np.full(len(y), np.nan, dtype=np.float64)
    coef_abs_sum = None
    n_coef_folds = 0
    leakage = []

    for fold, (tr, te) in enumerate(splits):
        g_tr, g_te = set(groups[tr].tolist()), set(groups[te].tolist())
        if g_tr & g_te:
            leakage.append(f"fold{fold}")
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        if pca_dims is not None:
            n_comp = min(int(pca_dims), Xtr.shape[0], Xtr.shape[1])
            pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            C=1.0,
            penalty="l2",
        )
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        oof_prob[te] = prob
        fold_aucs.append(float(roc_auc_score(y[te], prob)))
        if return_coefs and pca_dims is None:
            # Map coefficients back to original feature space via scaler scale
            coef = np.asarray(clf.coef_).ravel()  # on standardized features
            abs_c = np.abs(coef)
            if coef_abs_sum is None:
                coef_abs_sum = np.zeros_like(abs_c)
            coef_abs_sum = coef_abs_sum + abs_c
            n_coef_folds += 1

    if leakage:
        raise RuntimeError(f"Subject/group leakage: {leakage}")
    if np.any(~np.isfinite(oof_prob)):
        raise RuntimeError("Incomplete OOF coverage")

    out: Dict = {
        "split_mode": split_mode,
        "n_folds": N_FOLDS,
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "pca_dims": pca_dims,
        "mean_cv_auc": float(np.mean(fold_aucs)),
        "std_cv_auc": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "per_fold_auc": fold_aucs,
        "pooled_oof_auc": float(roc_auc_score(y, oof_prob)),
        "n_train_class": int(np.sum(y == 0)),
        "n_external_class": int(np.sum(y == 1)),
        "classifier": {
            "model": "LogisticRegression",
            "C": 1.0,
            "penalty": "l2",
            "class_weight": "balanced",
            "max_iter": 2000,
            "preprocessing": "StandardScaler on train fold"
            + (f" then PCA({pca_dims}) on train fold" if pca_dims else ""),
        },
        "seed": SEED,
    }
    if return_coefs and coef_abs_sum is not None:
        mean_abs = coef_abs_sum / max(n_coef_folds, 1)
        out["mean_abs_coef_across_folds"] = mean_abs
    return out


def concentration_stats(abs_coef: np.ndarray, ks=(10, 25, 50)) -> Dict:
    total = float(np.sum(abs_coef))
    order = np.argsort(-abs_coef)
    out = {"total_abs_coef": total, "n_features": int(len(abs_coef))}
    for k in ks:
        kk = min(k, len(abs_coef))
        top_idx = order[:kk]
        top_mag = float(np.sum(abs_coef[top_idx]))
        out[f"top_{k}"] = {
            "k": kk,
            "indices": top_idx.tolist(),
            "abs_coefs": abs_coef[top_idx].tolist(),
            "fraction_of_total_abs": top_mag / total if total > 0 else float("nan"),
        }
    return out


def family_of_index(i: int, families: Dict[str, np.ndarray]) -> str:
    for name in ("psd", "band_powers", "coherence", "plv"):
        if i in set(families[name].tolist()):
            return name
    return "unknown"


def make_family_figure(rows: List[Dict], out_path: Path) -> None:
    comps = ["TUH", "OSF", "P-ADIC"]
    fams = [
        "psd",
        "band_powers",
        "coherence",
        "plv",
        "connectivity_all",
        "spectral_psd_plus_band_powers",
        "full_2185",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), dpi=300, sharey=True)
    for ax, comp in zip(axes, comps):
        vals = []
        for f in fams:
            hit = [
                r
                for r in rows
                if r["external_cohort"] == comp and r["feature_family"] == f
            ]
            vals.append(hit[0]["pooled_oof_auc"] if hit else np.nan)
        x = np.arange(len(fams))
        ax.bar(x, vals, color="#4c72b0", alpha=0.85)
        ax.axhline(0.5, color="gray", ls="--", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(fams, rotation=55, ha="right", fontsize=7)
        ax.set_title(f"Train vs {comp}")
        ax.set_ylim(0.45, 1.05)
        if comp == "TUH":
            ax.set_ylabel("Pooled OOF ROC-AUC")
    fig.suptitle("Domain AUC by feature family (2185 layout)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_pca_figure(rows: List[Dict], out_path: Path) -> None:
    comps = ["TUH", "OSF", "P-ADIC"]
    reps = ["full_2185", "PCA-10", "PCA-25", "PCA-50"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=300)
    width = 0.22
    x = np.arange(len(reps))
    colors = ["#2c5f8a", "#c44e52", "#55a868"]
    for i, comp in enumerate(comps):
        vals = []
        for rep in reps:
            hit = [
                r
                for r in rows
                if r["external_cohort"] == comp and r["representation"] == rep
            ]
            vals.append(hit[0]["pooled_oof_auc"] if hit else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=comp, color=colors[i])
    ax.axhline(0.5, color="gray", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(reps)
    ax.set_ylabel("Pooled OOF ROC-AUC")
    ax.set_ylim(0.45, 1.05)
    ax.set_title("Domain AUC: raw 2185-D vs within-fold PCA")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_coef_figure(coef_blocks: List[Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=300)
    comps = [b["external_cohort"] for b in coef_blocks]
    f10 = [b["concentration"]["top_10"]["fraction_of_total_abs"] for b in coef_blocks]
    f25 = [b["concentration"]["top_25"]["fraction_of_total_abs"] for b in coef_blocks]
    f50 = [b["concentration"]["top_50"]["fraction_of_total_abs"] for b in coef_blocks]
    x = np.arange(len(comps))
    w = 0.25
    ax.bar(x - w, f10, width=w, label="Top 10", color="#4c72b0")
    ax.bar(x, f25, width=w, label="Top 25", color="#dd8452")
    ax.bar(x + w, f50, width=w, label="Top 50", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.set_ylabel("Fraction of Σ|coef|")
    ax.set_ylim(0, 1.0)
    ax.set_title("Coefficient concentration (mean |coef| across CV folds)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def assemble_xy(
    X_train, train_groups, X_ext, ext_groups
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack([X_train, X_ext]).astype(np.float64)
    y = np.concatenate(
        [np.zeros(len(X_train), dtype=int), np.ones(len(X_ext), dtype=int)]
    )
    groups = np.asarray(
        [f"train::{g}" for g in train_groups] + [f"ext::{g}" for g in ext_groups]
    )
    return X, y, groups


def main() -> int:
    root = _root()
    out_dir = root / "models" / "validation" / "domain_shift_forensics"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        out_dir / "feature_dimension_mapping.md",
        out_dir / "domain_shift_feature_family_results.json",
        out_dir / "domain_shift_pca_results.json",
        out_dir / "domain_shift_coefficient_forensics.json",
        out_dir / "domain_shift_forensic_summary.md",
        out_dir / "domain_shift_forensic_tables.csv",
        out_dir / "fig_feature_family_auc.png",
        out_dir / "fig_pca_auc.png",
        out_dir / "fig_coefficient_concentration.png",
    ]
    for p in targets:
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite {p}")

    families = build_feature_family_indices()
    write_feature_mapping_md(out_dir / "feature_dimension_mapping.md", families)

    X_train, train_sids, train_groups = load_training_flats(root)
    assert X_train.shape == (66, 2185)

    loaders = {"TUH": load_tuh, "OSF": load_osf, "P-ADIC": load_padic}
    family_order = [
        "full_2185",
        "psd",
        "band_powers",
        "coherence",
        "plv",
        "connectivity_all",
        "spectral_psd_plus_band_powers",
    ]

    family_rows: List[Dict] = []
    pca_rows: List[Dict] = []
    coef_blocks: List[Dict] = []
    check_a_ok = True

    for ext_name, loader in loaders.items():
        flats, _dis, sids, _labs = loader(root)
        flats = np.asarray(flats, dtype=np.float32)
        if flats.shape[1] != 2185:
            raise RuntimeError(f"{ext_name} dim mismatch")
        if ext_name == "TUH":
            ext_groups = np.asarray([tuh_patient_id(s) for s in sids])
        else:
            ext_groups = np.asarray(sids)
        X, y, groups = assemble_xy(X_train, train_groups, flats, ext_groups)

        print(f"\n=== {ext_name}: feature families ===")
        for fam in family_order:
            idx = families[fam]
            Xf = X[:, idx]
            res = domain_cv_auc(
                Xf, y, groups, return_coefs=(fam == "full_2185")
            )
            row = {
                "comparison": f"Training vs {ext_name}",
                "external_cohort": ext_name,
                "feature_family": fam,
                "dimensions": int(len(idx)),
                "mean_cv_auc": res["mean_cv_auc"],
                "std_cv_auc": res["std_cv_auc"],
                "pooled_oof_auc": res["pooled_oof_auc"],
                "per_fold_auc": res["per_fold_auc"],
                "n_training": int(np.sum(y == 0)),
                "n_external": int(np.sum(y == 1)),
                "n_unique_external_groups": int(len(np.unique(ext_groups))),
                "split_mode": res["split_mode"],
                "classifier": res["classifier"],
                "seed": SEED,
            }
            family_rows.append(row)
            print(
                f"  {fam:32s} d={len(idx):4d} "
                f"mean={res['mean_cv_auc']:.3f}±{res['std_cv_auc']:.3f} "
                f"OOF={res['pooled_oof_auc']:.3f}"
            )

            if fam == "full_2185":
                expected = EXPECTED_PRIMARY[ext_name]
                if abs(res["pooled_oof_auc"] - expected) > PRIMARY_ATOL:
                    check_a_ok = False
                    raise RuntimeError(
                        f"CHECK A FAIL {ext_name}: OOF={res['pooled_oof_auc']:.3f} "
                        f"vs expected ~{expected}"
                    )
                abs_coef = res["mean_abs_coef_across_folds"]
                # abs_coef is in the selected feature space (= full 2185)
                conc = concentration_stats(abs_coef)
                # family membership of top-25
                top25 = conc["top_25"]["indices"]
                fam_counts: Dict[str, int] = {}
                for ii in top25:
                    fn = family_of_index(int(ii), families)
                    fam_counts[fn] = fam_counts.get(fn, 0) + 1
                coef_blocks.append(
                    {
                        "external_cohort": ext_name,
                        "comparison": f"Training vs {ext_name}",
                        "representation": "full_2185_standardized_within_fold",
                        "note": (
                            "Mean absolute logistic coefficients across 5 CV folds; "
                            "coefficients are on StandardScaler-transformed features "
                            "within each fold. Not biological biomarkers."
                        ),
                        "concentration": {
                            k: v
                            for k, v in conc.items()
                            if k != "n_features" or True
                        },
                        "top25_family_counts": fam_counts,
                        "top10_family_counts": {
                            family_of_index(int(ii), families): None
                            for ii in conc["top_10"]["indices"]
                        },
                    }
                )
                # fix top10 counts properly
                t10c: Dict[str, int] = {}
                for ii in conc["top_10"]["indices"]:
                    fn = family_of_index(int(ii), families)
                    t10c[fn] = t10c.get(fn, 0) + 1
                coef_blocks[-1]["top10_family_counts"] = t10c

        print(f"=== {ext_name}: PCA sensitivity ===")
        # include raw full as reference
        full = next(
            r
            for r in family_rows
            if r["external_cohort"] == ext_name and r["feature_family"] == "full_2185"
        )
        pca_rows.append(
            {
                "comparison": f"Training vs {ext_name}",
                "external_cohort": ext_name,
                "representation": "full_2185",
                "dimensions": 2185,
                "mean_cv_auc": full["mean_cv_auc"],
                "std_cv_auc": full["std_cv_auc"],
                "pooled_oof_auc": full["pooled_oof_auc"],
                "source": "reproduced_in_forensics_run",
            }
        )
        for d in (10, 25, 50):
            res = domain_cv_auc(X, y, groups, pca_dims=d)
            row = {
                "comparison": f"Training vs {ext_name}",
                "external_cohort": ext_name,
                "representation": f"PCA-{d}",
                "dimensions": d,
                "mean_cv_auc": res["mean_cv_auc"],
                "std_cv_auc": res["std_cv_auc"],
                "pooled_oof_auc": res["pooled_oof_auc"],
                "per_fold_auc": res["per_fold_auc"],
                "split_mode": res["split_mode"],
                "classifier": res["classifier"],
                "seed": SEED,
                "pca_fit": "within each CV training fold only (after StandardScaler)",
            }
            pca_rows.append(row)
            print(
                f"  PCA-{d}: mean={res['mean_cv_auc']:.3f}±{res['std_cv_auc']:.3f} "
                f"OOF={res['pooled_oof_auc']:.3f}"
            )

    # Figures
    make_family_figure(family_rows, out_dir / "fig_feature_family_auc.png")
    make_pca_figure(pca_rows, out_dir / "fig_pca_auc.png")
    make_coef_figure(coef_blocks, out_dir / "fig_coefficient_concentration.png")

    # CSV tables
    csv_path = out_dir / "domain_shift_forensic_tables.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "table",
                "comparison",
                "representation_or_family",
                "dimensions",
                "mean_cv_auc",
                "std_cv_auc",
                "pooled_oof_auc",
            ]
        )
        for r in family_rows:
            w.writerow(
                [
                    "feature_family",
                    r["comparison"],
                    r["feature_family"],
                    r["dimensions"],
                    f"{r['mean_cv_auc']:.6f}",
                    f"{r['std_cv_auc']:.6f}",
                    f"{r['pooled_oof_auc']:.6f}",
                ]
            )
        for r in pca_rows:
            w.writerow(
                [
                    "pca",
                    r["comparison"],
                    r["representation"],
                    r["dimensions"],
                    f"{r['mean_cv_auc']:.6f}",
                    f"{r['std_cv_auc']:.6f}",
                    f"{r['pooled_oof_auc']:.6f}",
                ]
            )

    family_json = {
        "timestamp": datetime.now().isoformat(),
        "script": "src/validation/run_domain_shift_forensics.py",
        "seed": SEED,
        "n_folds": N_FOLDS,
        "check_A_primary_auc_reproduced": check_a_ok,
        "expected_primary_oof": EXPECTED_PRIMARY,
        "feature_mapping_note": (
            "Families isolated via flatten_features interleaved layout; "
            "feature_block_ranges contiguous coh/plv blocks not used."
        ),
        "acquisition_site_check": (
            "Acquisition/site metadata were not available as multi-site labels. "
            "TUH processing_report includes load_meta fields "
            "(original_sfreq, notch_freq, n_epochs_used, epoch_power_calibration) "
            "but no site/acquisition-site classifier was run."
        ),
        "rows": family_rows,
    }
    pca_json = {
        "timestamp": datetime.now().isoformat(),
        "script": "src/validation/run_domain_shift_forensics.py",
        "seed": SEED,
        "pca_protocol": (
            "Within each CV fold: StandardScaler.fit(train) → PCA.fit(scaled train) "
            "→ LogisticRegression; transform/held-out with train-fitted objects only."
        ),
        "rows": pca_rows,
    }
    # Serialize coef blocks without huge float arrays beyond tops
    coef_json = {
        "timestamp": datetime.now().isoformat(),
        "script": "src/validation/run_domain_shift_forensics.py",
        "seed": SEED,
        "interpretation_guardrail": (
            "High |coef| dimensions indicate contribution to domain discrimination "
            "in standardized feature space; they are not biological biomarkers."
        ),
        "comparisons": coef_blocks,
    }

    (out_dir / "domain_shift_feature_family_results.json").write_text(
        json.dumps(family_json, indent=2), encoding="utf-8"
    )
    (out_dir / "domain_shift_pca_results.json").write_text(
        json.dumps(pca_json, indent=2), encoding="utf-8"
    )
    (out_dir / "domain_shift_coefficient_forensics.json").write_text(
        json.dumps(coef_json, indent=2), encoding="utf-8"
    )

    # Summary markdown
    lines = [
        "# Domain-shift forensic summary",
        "",
        f"Script: `src/validation/run_domain_shift_forensics.py`  ",
        f"Seed={SEED}; folds={N_FOLDS}; LR L2 C=1 balanced; scaler (+PCA) train-fold only.",
        "",
        "## CHECK A (primary 2185-D OOF AUC reproduction)",
        "",
    ]
    for ext in ("TUH", "OSF", "P-ADIC"):
        r = next(
            x
            for x in family_rows
            if x["external_cohort"] == ext and x["feature_family"] == "full_2185"
        )
        lines.append(
            f"- {ext}: OOF={r['pooled_oof_auc']:.3f} "
            f"(expected ~{EXPECTED_PRIMARY[ext]:.3f})"
        )
    lines.extend(["", "## Feature-family results", "", "| Comparison | Feature family | Dimensions | Mean CV AUC | SD | Pooled OOF AUC |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for r in family_rows:
        lines.append(
            f"| {r['comparison']} | {r['feature_family']} | {r['dimensions']} | "
            f"{r['mean_cv_auc']:.3f} | {r['std_cv_auc']:.3f} | {r['pooled_oof_auc']:.3f} |"
        )
    lines.extend(["", "## PCA sensitivity", "", "| Comparison | Representation | Mean CV AUC | SD | Pooled OOF AUC |", "| --- | --- | ---: | ---: | ---: |"])
    for r in pca_rows:
        lines.append(
            f"| {r['comparison']} | {r['representation']} | "
            f"{r['mean_cv_auc']:.3f} | {r['std_cv_auc']:.3f} | {r['pooled_oof_auc']:.3f} |"
        )
    lines.extend(["", "## Coefficient concentration (Σ|coef| fractions)", ""])
    for b in coef_blocks:
        c = b["concentration"]
        lines.append(
            f"- **{b['external_cohort']}**: top10={c['top_10']['fraction_of_total_abs']:.3f}; "
            f"top25={c['top_25']['fraction_of_total_abs']:.3f}; "
            f"top50={c['top_50']['fraction_of_total_abs']:.3f}; "
            f"top10 families={b['top10_family_counts']}; "
            f"top25 families={b['top25_family_counts']}"
        )
    lines.extend(
        [
            "",
            "## Acquisition/site",
            "",
            family_json["acquisition_site_check"],
            "",
            "## Guardrails",
            "",
            "Results describe distributional / representation-level separability only. "
            "They do not establish biological mechanism, causal domain effects, "
            "or that domain shift caused twin performance differences.",
            "",
        ]
    )
    (out_dir / "domain_shift_forensic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"\n[wrote] outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
