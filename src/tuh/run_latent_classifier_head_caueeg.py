"""
Stronger classifier heads on frozen CVAE mu_base (CAUEEG Dementia vs Normal).

Read-only. Does NOT retrain the twin.
Does NOT overwrite layer5e_caueeg_latent_probe_results.json.
Uses same folds / scaler-per-fold discipline as run_latent_probe_caueeg.py.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Keep constants local — avoid importing run_latent_probe_caueeg (pulls torch;
# joblib workers then OOM on Windows when n_jobs>1).
N_FOLDS = 5
N_PERM = 200
RANDOM_STATE = 42
REF_LOGREG_AUC = 0.579
REF_THETA_ALPHA_AUC = 0.675
INNER_FOLDS = 3


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dem_vs_norm_latents(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reuse baseline_latents.npz; Dementia vs Normal only (drop MCI)."""
    path = root / "data" / "caueeg_external" / "validation" / "baseline_latents.npz"
    z = np.load(path, allow_pickle=True)
    labels = np.asarray(z["labels"]).astype(np.int64)
    mask = np.asarray(z["dementia_vs_normal_mask"]).astype(bool)
    mu = np.asarray(z["mu_base"], dtype=np.float64)[mask]
    logvar = np.asarray(z["logvar_base"], dtype=np.float64)[mask]
    y = (labels[mask] == 2).astype(np.int64)  # Dementia=1, Normal=0
    sids = np.asarray(z["subject_ids"])[mask]
    return mu, logvar, y, sids


def load_prior_probe_refs(root: Path) -> Dict[str, float]:
    prior = root / "models" / "validation" / "layer5e_caueeg_latent_probe_results.json"
    out = {
        "logreg_oof_auc": REF_LOGREG_AUC,
        "theta_alpha_oof_auc": REF_THETA_ALPHA_AUC,
        "logreg_cv_mean_auc": REF_LOGREG_AUC,
        "theta_alpha_cv_mean_auc": REF_THETA_ALPHA_AUC,
    }
    if not prior.exists():
        return out
    j = json.loads(prior.read_text(encoding="utf-8"))
    lat = j["latent_probe_dementia_vs_normal"]["cv"]
    feat = j["feature_probe_theta_alpha_dementia_vs_normal"]["cv"]
    out["logreg_cv_mean_auc"] = float(lat["roc_auc"]["mean"])
    out["logreg_oof_auc"] = float(lat["oof"]["roc_auc"])
    out["theta_alpha_cv_mean_auc"] = float(feat["roc_auc"]["mean"])
    out["theta_alpha_oof_auc"] = float(feat["oof"]["roc_auc"])
    return out


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _head_specs() -> Dict[str, Dict[str, Any]]:
    """Pipelines + inner GridSearch param grids (prefix clf__)."""
    return {
        "tuned_logreg_l2": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=5000,
                            solver="lbfgs",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
            "needs_sample_weight": False,
        },
        "mlp": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            activation="relu",
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=15,
                            max_iter=400,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "clf__hidden_layer_sizes": [(32,), (64,)],
                "clf__alpha": [0.0001, 0.001, 0.01],
            },
            "needs_sample_weight": True,  # MLP has no class_weight
        },
        "hist_gradient_boosting": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_depth=3,
                            learning_rate=0.05,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=15,
                            random_state=RANDOM_STATE,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "clf__max_iter": [50, 100],
                "clf__l2_regularization": [0.0, 1.0],
                "clf__max_leaf_nodes": [15, 31],
            },
            "needs_sample_weight": False,
        },
    }


def nested_cv_head(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    spec: Dict[str, Any],
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Outer 5-fold; inner 3-fold GridSearchCV. Scaler inside pipeline (no leak)."""
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=random_state)

    fold_metrics: List[Dict[str, Any]] = []
    oof_prob = np.zeros(len(y), dtype=np.float64)
    oof_pred = np.zeros(len(y), dtype=np.int64)
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
        fit_params = {}
        if spec.get("needs_sample_weight"):
            sw = compute_sample_weight("balanced", y[tr])
            fit_params["clf__sample_weight"] = sw

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X[tr], y[tr], **fit_params)

        # Predict on outer test (pipeline includes scaler)
        proba = gs.predict_proba(X[te])[:, 1]
        pred = gs.predict(X[te])
        m = _binary_metrics(y[te], proba, pred)
        m["fold"] = fold
        m["best_params"] = {k: _jsonable(v) for k, v in gs.best_params_.items()}
        fold_metrics.append(m)
        best_params_per_fold.append(m["best_params"])
        oof_prob[te] = proba
        oof_pred[te] = pred
        if verbose:
            print(
                f"  [{name}] fold {fold+1}/{N_FOLDS} "
                f"AUC={m['roc_auc']:.3f} params={m['best_params']}"
            )

    keys = ("roc_auc", "balanced_accuracy", "f1")
    summary: Dict[str, Any] = {
        "name": name,
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
    summary["n"] = int(len(y))
    summary["n_pos"] = int(np.sum(y == 1))
    summary["n_neg"] = int(np.sum(y == 0))
    return summary


def _jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def permutation_best_head(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    spec: Dict[str, Any],
    true_oof_auc: float,
    n_perm: int = N_PERM,
) -> Dict[str, Any]:
    """Shuffle labels; rerun same nested CV; collect OOF ROC-AUC null."""
    rng = np.random.RandomState(RANDOM_STATE)
    null_aucs = []
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        # New outer fold seed per perm (same protocol as original probe)
        res = nested_cv_head(
            X,
            y_shuf,
            name=f"{name}_perm{i}",
            spec=spec,
            random_state=RANDOM_STATE + 1 + i,
            verbose=False,
        )
        null_aucs.append(res["oof"]["roc_auc"])
        if (i + 1) % 10 == 0:
            print(f"  perm [{i+1}/{n_perm}] null_oof_auc_mean={np.mean(null_aucs):.3f}")

    null_aucs = np.asarray(null_aucs, dtype=np.float64)
    p_value = float((np.sum(null_aucs >= true_oof_auc) + 1) / (n_perm + 1))
    percentile = float(100.0 * np.mean(null_aucs < true_oof_auc))
    return {
        "head": name,
        "n_permutations": n_perm,
        "true_oof_auc": float(true_oof_auc),
        "null_auc_mean": float(np.mean(null_aucs)),
        "null_auc_std": float(np.std(null_aucs, ddof=1)),
        "null_auc_percentiles": {
            "p5": float(np.percentile(null_aucs, 5)),
            "p50": float(np.percentile(null_aucs, 50)),
            "p95": float(np.percentile(null_aucs, 95)),
        },
        "true_auc_percentile_vs_null": percentile,
        "permutation_p_value": p_value,
        "null_aucs": null_aucs.tolist(),
    }


def verdict_from_auc(best_oof: float) -> Tuple[str, str]:
    """
    Returns (short_tag, long_text) per user decision rule.
    """
    if best_oof >= 0.65:
        return (
            "pursue downstream head",
            "Best head OOF ROC-AUC is close to or above the theta/alpha "
            "reference (~0.65+). A lightweight downstream head is a real fix "
            "worth adding as an auxiliary diagnostic output and recommending "
            "for the paper.",
        )
    if 0.57 <= best_oof <= 0.62:
        return (
            "ceiling reached, move to writing",
            "Best head OOF ROC-AUC stays near the original logistic regression "
            "latent probe (~0.57–0.62), not meaningfully beyond it. The ceiling "
            "was already found; information loss is at encoding, not readout. "
            "Do not pursue further downstream heads; report the ~0.579 latent "
            "probe as-is and move on to writing.",
        )
    if best_oof > 0.62 and best_oof < 0.65:
        return (
            "ceiling reached, move to writing",
            "Best head improves modestly over logistic regression but remains "
            "well below theta/alpha (~0.675). Downstream nonlinear readout is "
            "not enough to close the gap; treat encoding loss as the bottleneck "
            "and move on to writing with the latent-probe result.",
        )
    # below 0.57
    return (
        "ceiling reached, move to writing",
        "Best head did not improve on the original logistic probe. Downstream "
        "readout is not the lever; move on to writing.",
    )


def write_summary_md(path: Path, results: Dict) -> None:
    refs = results["references"]
    rows = results["comparison_table"]
    best = results["best_head"]
    tag, long_v = results["verdict_tag"], results["verdict_text"]

    lines = [
        "# Layer 5e — Stronger classifier heads on frozen CVAE latents",
        "",
        "Read-only. Twin **not** retrained. Nested CV (outer 5 / inner 3). "
        "Does not overwrite prior probe JSON.",
        "",
        "## References",
        "",
        f"- Original latent LogReg CV mean AUC: **{refs['logreg_cv_mean_auc']:.3f}** "
        f"(OOF {refs['logreg_oof_auc']:.3f})",
        f"- Theta/alpha feature probe CV mean AUC: **{refs['theta_alpha_cv_mean_auc']:.3f}** "
        f"(OOF {refs['theta_alpha_oof_auc']:.3f})",
        "",
        "## Results table",
        "",
        "| Model | OOF ROC-AUC | Balanced Acc | vs LogReg baseline (0.579) | vs theta/alpha (0.675) |",
        "|-------|-------------|--------------|----------------------------|-------------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['oof_roc_auc']:.3f} | {r['oof_balanced_accuracy']:.3f} | "
            f"{r['delta_vs_logreg']:+.3f} | {r['delta_vs_theta_alpha']:+.3f} |"
        )

    lines.extend(
        [
            "",
            f"**Best head:** `{best['name']}` — OOF AUC **{best['oof_roc_auc']:.3f}**",
            "",
            "## Permutation (best head)",
            "```json",
            json.dumps(
                {k: v for k, v in results["permutation_best_head"].items() if k != "null_aucs"},
                indent=2,
            ),
            "```",
            "",
            "## mu + logvar control (best head family)",
            "```json",
            json.dumps(results["mu_logvar_control"]["oof"], indent=2),
            "```",
            "",
            f"OOF AUC with `[mu; logvar]`: "
            f"**{results['mu_logvar_control']['oof']['roc_auc']:.3f}** "
            f"(Δ vs mu-only best: "
            f"{results['mu_logvar_control']['oof']['roc_auc'] - best['oof_roc_auc']:+.3f})",
            "",
            "## Verdict",
            "",
            f"**{tag}**",
            "",
            long_v,
            "",
            "## Artifacts",
            "",
            "- `models/validation/layer5e_caueeg_classifier_head_results.json`",
            "- Latents reused: `data/caueeg_external/validation/baseline_latents.npz`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[wrote] {path}")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument(
        "--fast-perm",
        action="store_true",
        help="Use fewer perms for smoke tests (default full 200)",
    )
    args = ap.parse_args()
    n_perm = 20 if args.fast_perm else args.n_perm

    root = _root()
    out_json = root / "models" / "validation" / "layer5e_caueeg_classifier_head_results.json"
    out_md = root / "models" / "validation" / "layer5e_caueeg_classifier_head_summary.md"

    print("=== load frozen latents (Dem vs Norm) ===")
    mu, logvar, y, sids = load_dem_vs_norm_latents(root)
    print(f"n={len(y)} pos={y.sum()} neg={(y==0).sum()} dim={mu.shape[1]}")
    refs = load_prior_probe_refs(root)
    print(
        f"refs: logreg_oof={refs['logreg_oof_auc']:.3f} "
        f"theta_alpha_oof={refs['theta_alpha_oof_auc']:.3f}"
    )

    specs = _head_specs()
    head_results = {}
    for name, spec in specs.items():
        print(f"=== nested CV: {name} ===")
        head_results[name] = nested_cv_head(mu, y, name, spec)

    # Pick best by OOF ROC-AUC
    best_name = max(head_results, key=lambda k: head_results[k]["oof"]["roc_auc"])
    best = head_results[best_name]
    print(f"=== best head: {best_name} OOF AUC={best['oof']['roc_auc']:.3f} ===")

    print(f"=== permutation ({n_perm}x) on best head ===")
    perm = permutation_best_head(
        mu, y, best_name, specs[best_name], best["oof"]["roc_auc"], n_perm=n_perm
    )
    print(f"  perm p={perm['permutation_p_value']:.4f}")

    print("=== mu+logvar control with best head family ===")
    X_both = np.concatenate([mu, logvar], axis=1)
    control = nested_cv_head(X_both, y, f"{best_name}_mu_logvar", specs[best_name])
    print(f"  control OOF AUC={control['oof']['roc_auc']:.3f}")

    logreg_ref = float(refs["logreg_oof_auc"])
    theta_ref = float(refs["theta_alpha_oof_auc"])
    # Also keep the user-facing 0.579 / 0.675 in table labels
    table = []
    for name, res in head_results.items():
        oof_auc = res["oof"]["roc_auc"]
        table.append(
            {
                "model": name,
                "oof_roc_auc": oof_auc,
                "oof_balanced_accuracy": res["oof"]["balanced_accuracy"],
                "oof_f1": res["oof"]["f1"],
                "cv_roc_auc_mean": res["roc_auc"]["mean"],
                "cv_roc_auc_std": res["roc_auc"]["std"],
                "delta_vs_logreg": oof_auc - logreg_ref,
                "delta_vs_theta_alpha": oof_auc - theta_ref,
                "delta_vs_logreg_0.579": oof_auc - REF_LOGREG_AUC,
                "delta_vs_theta_alpha_0.675": oof_auc - REF_THETA_ALPHA_AUC,
            }
        )
    table.append(
        {
            "model": f"{best_name} + logvar (control)",
            "oof_roc_auc": control["oof"]["roc_auc"],
            "oof_balanced_accuracy": control["oof"]["balanced_accuracy"],
            "oof_f1": control["oof"]["f1"],
            "cv_roc_auc_mean": control["roc_auc"]["mean"],
            "cv_roc_auc_std": control["roc_auc"]["std"],
            "delta_vs_logreg": control["oof"]["roc_auc"] - logreg_ref,
            "delta_vs_theta_alpha": control["oof"]["roc_auc"] - theta_ref,
            "delta_vs_logreg_0.579": control["oof"]["roc_auc"] - REF_LOGREG_AUC,
            "delta_vs_theta_alpha_0.675": control["oof"]["roc_auc"] - REF_THETA_ALPHA_AUC,
        }
    )

    best_oof = float(best["oof"]["roc_auc"])
    tag, long_v = verdict_from_auc(best_oof)

    results = {
        "experiment": "caueeg_frozen_latent_classifier_heads",
        "read_only": True,
        "twin_retrained": False,
        "n": int(len(y)),
        "n_pos_dementia": int(y.sum()),
        "n_neg_normal": int((y == 0).sum()),
        "latent_dim": int(mu.shape[1]),
        "outer_folds": N_FOLDS,
        "inner_folds": INNER_FOLDS,
        "random_state": RANDOM_STATE,
        "references": refs,
        "heads": head_results,
        "best_head": {
            "name": best_name,
            "oof_roc_auc": best_oof,
            "oof_balanced_accuracy": best["oof"]["balanced_accuracy"],
            "oof_f1": best["oof"]["f1"],
        },
        "permutation_best_head": perm,
        "mu_logvar_control": control,
        "comparison_table": table,
        "verdict_tag": tag,
        "verdict_text": long_v,
        "does_not_overwrite": [
            "models/validation/layer5e_caueeg_latent_probe_results.json",
            "models/validation/layer5e_caueeg_results.json",
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[wrote] {out_json}")
    write_summary_md(out_md, results)

    print(
        f"BEST HEAD: {best_name} | OOF AUC = {best_oof:.3f} | "
        f"vs LogReg baseline 0.579 | vs theta/alpha 0.675 | "
        f"VERDICT: {tag}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
