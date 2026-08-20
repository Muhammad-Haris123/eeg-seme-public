"""
Layer 5e diagnostic: linear probe on CVAE baseline latent means (CAUEEG).

Read-only. Does NOT retrain or modify the twin.
Does NOT overwrite layer5e_caueeg_results.json.

Question: does Dementia vs Normal clinical signal survive encoding into
mu_base (zero-drug), or is it discarded before drug-response magnitude?

Leakage control: disease_label is FIXED to 0 for all subjects when
extracting mu_base so the probe cannot read the diagnosis bit from fusion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.inference import load_trained_model
from src.tuh.process_tuh_eeg import unflatten_features
from src.tuh.run_layer5e_caueeg import _root, load_processed_features

LABEL_TO_INT = {"normal": 0, "mci": 1, "dementia": 2}
INT_TO_LABEL = {0: "Normal", 1: "MCI", 2: "Dementia"}
N_PERM = 200
N_FOLDS = 5
RANDOM_STATE = 42
# Fixed disease bit — prevents diagnosis leakage into fusion → CVAE encode
PROBE_DISEASE_LABEL = 0.0


def extract_baseline_latents(
    flats: np.ndarray,
    metas: List[Dict],
    root: Path,
    out_npz: Path,
) -> Dict[str, np.ndarray]:
    """Single forward pass per subject: EEG → fusion(drug=0, disease=0) → mu, logvar."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = root / "models" / "checkpoints_constrained" / "checkpoint_constrained.pt"
    model = load_trained_model(ckpt, device)
    model.eval()

    zero_drug = torch.zeros(1, 384, dtype=torch.float32, device=device)
    disease_t = torch.tensor([PROBE_DISEASE_LABEL], dtype=torch.float32, device=device)

    mus, logvars, sids, labels = [], [], [], []
    with torch.no_grad():
        drug_lat = model.drug_encoder(zero_drug)
        for i, (flat, m) in enumerate(zip(flats, metas)):
            eeg = unflatten_features(flat)
            psd = torch.FloatTensor(eeg["psd"]).unsqueeze(0).to(device)
            bp = torch.FloatTensor(eeg["band_powers"]).unsqueeze(0).to(device)
            coh = torch.FloatTensor(eeg["coherence"]).unsqueeze(0).to(device)
            plv = torch.FloatTensor(eeg["plv"]).unsqueeze(0).to(device)
            eeg_lat = model.eeg_encoder(psd, bp, coh, plv)
            fused = model.fusion(eeg_lat, drug_lat, disease_t)
            mu, logvar = model.cvae.encode(fused)
            mus.append(mu.cpu().numpy().reshape(-1))
            logvars.append(logvar.cpu().numpy().reshape(-1))
            sids.append(m["recording_id"])
            labels.append(LABEL_TO_INT[m["label"]])
            if (i + 1) % 100 == 0:
                print(f"  encode [{i+1}/{len(metas)}]")

    mu_base = np.stack(mus, axis=0).astype(np.float32)
    logvar_base = np.stack(logvars, axis=0).astype(np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(sids)
    # Binary mask: Normal or Dementia only (exclude MCI)
    dem_vs_norm_mask = (labels_arr == 0) | (labels_arr == 2)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        subject_ids=subject_ids,
        mu_base=mu_base,
        logvar_base=logvar_base,
        labels=labels_arr,
        dementia_vs_normal_mask=dem_vs_norm_mask.astype(np.bool_),
        disease_label_used=np.float32(PROBE_DISEASE_LABEL),
        checkpoint=str(ckpt),
    )
    print(f"[saved] {out_npz} mu_base={mu_base.shape}")
    return {
        "subject_ids": subject_ids,
        "mu_base": mu_base,
        "logvar_base": logvar_base,
        "labels": labels_arr,
        "dementia_vs_normal_mask": dem_vs_norm_mask,
    }


def theta_alpha_ratio(flats: np.ndarray) -> np.ndarray:
    """Same spectral summary as Layer 5e feature-space check (19ch × 5 bands at [380:475])."""
    bp = flats[:, 380:475].reshape(-1, 19, 5)
    theta = bp[:, :, 1].mean(axis=1)
    alpha = bp[:, :, 2].mean(axis=1)
    return (theta / np.maximum(alpha, 1e-12)).astype(np.float64)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def probe_binary_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict:
    """5-fold stratified LR; scaler fit inside each fold."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_metrics = []
    oof_prob = np.zeros(len(y), dtype=np.float64)
    oof_pred = np.zeros(len(y), dtype=np.int64)

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
    return summary


def probe_binary_insample(X: np.ndarray, y: np.ndarray) -> Dict:
    """Fit on all data — sanity upper bound, NOT held-out."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(Xs, y)
    prob = clf.predict_proba(Xs)[:, 1]
    pred = clf.predict(Xs)
    m = _binary_metrics(y, prob, pred)
    m["note"] = "in-sample, not held-out"
    return m


def probe_multiclass_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    bal_accs, macro_f1s = [], []
    oof_pred = np.zeros(len(y), dtype=np.int64)

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
        pred = clf.predict(Xte)
        bal_accs.append(float(balanced_accuracy_score(y[te], pred)))
        macro_f1s.append(float(f1_score(y[te], pred, average="macro", zero_division=0)))
        oof_pred[te] = pred

    cm = confusion_matrix(y, oof_pred, labels=[0, 1, 2])
    return {
        "balanced_accuracy": {
            "mean": float(np.mean(bal_accs)),
            "std": float(np.std(bal_accs, ddof=1)),
            "per_fold": bal_accs,
        },
        "macro_f1": {
            "mean": float(np.mean(macro_f1s)),
            "std": float(np.std(macro_f1s, ddof=1)),
            "per_fold": macro_f1s,
        },
        "oof_confusion_matrix": {
            "labels": ["Normal", "MCI", "Dementia"],
            "matrix": cm.tolist(),
        },
        "n": int(len(y)),
        "n_by_class": {INT_TO_LABEL[i]: int(np.sum(y == i)) for i in (0, 1, 2)},
    }


def permutation_test_binary(
    X: np.ndarray,
    y: np.ndarray,
    true_auc: float,
    n_perm: int = N_PERM,
    random_state: int = RANDOM_STATE,
) -> Dict:
    rng = np.random.RandomState(random_state)
    null_aucs = []
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        # Different fold seed per perm to avoid identical splits; still reproducible
        res = probe_binary_cv(X, y_shuf, random_state=random_state + 1 + i)
        null_aucs.append(res["roc_auc"]["mean"])
        if (i + 1) % 25 == 0:
            print(f"  perm [{i+1}/{n_perm}] null_auc_mean={np.mean(null_aucs):.3f}")

    null_aucs = np.asarray(null_aucs, dtype=np.float64)
    # One-sided: how often null >= true
    p_value = float((np.sum(null_aucs >= true_auc) + 1) / (n_perm + 1))
    percentile = float(100.0 * np.mean(null_aucs < true_auc))
    return {
        "n_permutations": n_perm,
        "true_auc": float(true_auc),
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


def write_summary_md(path: Path, results: Dict) -> None:
    lat = results["latent_probe_dementia_vs_normal"]["cv"]
    feat = results["feature_probe_theta_alpha_dementia_vs_normal"]["cv"]
    perm = results["permutation_test_latent"]
    lat_auc = lat["roc_auc"]["mean"]
    feat_auc = feat["roc_auc"]["mean"]
    p = perm["permutation_p_value"]

    beats_null = p < 0.05
    # Comparable if within 0.05 AUC of feature probe
    comparable = abs(lat_auc - feat_auc) <= 0.05 or lat_auc >= feat_auc - 0.05

    if beats_null and comparable and lat_auc >= 0.55:
        reading = (
            "Clinical signal appears to **SURVIVE** encoding into mu_base "
            "(latent probe above permutation null and in the ballpark of the "
            "theta/alpha feature probe). The Layer 5e null on drug-response "
            "**magnitude** is more likely a **downstream endpoint / readout** "
            "issue than total loss of diagnosis at encode time. A lightweight "
            "classifier head on latents may be worth trying; full twin retrain "
            "is not required to test that hypothesis."
        )
    elif not beats_null or lat_auc < 0.55:
        reading = (
            "Clinical signal appears **DISCARDED or heavily attenuated** at "
            "encoding into mu_base (latent probe not clearly above the "
            "permutation null / chance). Fixing Layer 5e would likely need "
            "**architecture or constraint changes and retraining**, not only "
            "a downstream classifier on current latents."
        )
    else:
        reading = (
            "Latent probe beats the permutation null but **falls short** of the "
            "theta/alpha feature probe. Some signal survives encoding, but "
            "less than the raw spectral marker; both readout changes and "
            "milder encoder/constraint revisits remain open."
        )

    lines = [
        "# Layer 5e — CVAE baseline latent linear probe (CAUEEG)",
        "",
        "Read-only diagnostic. Twin **not** retrained. Does not modify "
        "`layer5e_caueeg_results.json`.",
        "",
        "## Setup",
        "",
        f"- Checkpoint: `{results['checkpoint']}`",
        f"- N subjects: {results['n_subjects']} "
        f"(Normal {results['n_by_class']['Normal']} / "
        f"MCI {results['n_by_class']['MCI']} / "
        f"Dementia {results['n_by_class']['Dementia']})",
        f"- mu_base: zero-drug embedding; **disease_label fixed to "
        f"{PROBE_DISEASE_LABEL}** for all (no diagnosis leak into fusion)",
        f"- CV: {N_FOLDS}-fold stratified; LogisticRegression "
        f"`class_weight='balanced'`; StandardScaler fit per fold",
        f"- Permutations: {N_PERM}",
        "",
        "## Headline",
        "",
        f"- **Latent probe (Dem vs Norm) ROC-AUC:** "
        f"{lat_auc:.3f} ± {lat['roc_auc']['std']:.3f}",
        f"- **Permutation p-value:** {p:.4f} "
        f"(true AUC percentile vs null: {perm['true_auc_percentile_vs_null']:.1f}%)",
        f"- **Theta/alpha feature probe ROC-AUC:** "
        f"{feat_auc:.3f} ± {feat['roc_auc']['std']:.3f}",
        f"- Beats permutation null (p < 0.05)? **{'YES' if beats_null else 'NO'}**",
        "",
        "## Dementia vs Normal (primary)",
        "",
        "### Latent mu_base",
        "```json",
        json.dumps(
            {
                "cv": lat,
                "in_sample_not_held_out": results["latent_probe_dementia_vs_normal"][
                    "in_sample_not_held_out"
                ],
            },
            indent=2,
        ),
        "```",
        "",
        "### Theta/alpha ratio (feature reference)",
        "```json",
        json.dumps(
            {
                "cv": feat,
                "in_sample_not_held_out": results[
                    "feature_probe_theta_alpha_dementia_vs_normal"
                ]["in_sample_not_held_out"],
            },
            indent=2,
        ),
        "```",
        "",
        "## 3-class (Normal / MCI / Dementia)",
        "```json",
        json.dumps(results["latent_probe_3class"], indent=2),
        "```",
        "",
        "## Permutation null (latent Dem vs Norm)",
        "```json",
        json.dumps({k: v for k, v in perm.items() if k != "null_aucs"}, indent=2),
        "```",
        "",
        "## Interpretation (do not overclaim)",
        "",
        reading,
        "",
        "## Artifacts",
        "",
        f"- Latents: `{results['latents_path']}`",
        f"- Results JSON: `models/validation/layer5e_caueeg_latent_probe_results.json`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[wrote] {path}")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force-encode", action="store_true")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    root = _root()
    data_dir = root / "data" / "caueeg_external"
    val_dir = data_dir / "validation"
    latents_path = val_dir / "baseline_latents.npz"
    out_json = root / "models" / "validation" / "layer5e_caueeg_latent_probe_results.json"
    out_md = root / "models" / "validation" / "layer5e_caueeg_latent_probe_summary.md"
    ckpt = root / "models" / "checkpoints_constrained" / "checkpoint_constrained.pt"

    print("=== load CAUEEG features ===")
    flats, metas = load_processed_features(data_dir)
    print(f"loaded {flats.shape[0]} x {flats.shape[1]}")

    if args.force_encode or (not latents_path.exists()):
        print("=== extract baseline latents (drug=0, disease=0) ===")
        pack = extract_baseline_latents(flats, metas, root, latents_path)
    else:
        print(f"[skip encode] {latents_path}")
        z = np.load(latents_path, allow_pickle=True)
        pack = {k: z[k] for k in z.files if k in (
            "subject_ids", "mu_base", "logvar_base", "labels", "dementia_vs_normal_mask"
        )}

    mu = pack["mu_base"]
    labels = np.asarray(pack["labels"]).astype(np.int64)
    mask = np.asarray(pack["dementia_vs_normal_mask"]).astype(bool)

    # Align theta/alpha with same subject order as flats/metas
    ta = theta_alpha_ratio(flats)
    # Primary binary: Dementia=1, Normal=0 (drop MCI)
    y_bin = (labels[mask] == 2).astype(np.int64)
    X_lat = mu[mask]
    X_feat = ta[mask].reshape(-1, 1)

    print(f"=== binary probe Dem vs Norm n={len(y_bin)} "
          f"(pos={y_bin.sum()}, neg={(y_bin==0).sum()}) ===")
    lat_cv = probe_binary_cv(X_lat, y_bin)
    lat_in = probe_binary_insample(X_lat, y_bin)
    print(f"  latent CV AUC={lat_cv['roc_auc']['mean']:.3f}±{lat_cv['roc_auc']['std']:.3f}")

    feat_cv = probe_binary_cv(X_feat, y_bin)
    feat_in = probe_binary_insample(X_feat, y_bin)
    print(f"  feature CV AUC={feat_cv['roc_auc']['mean']:.3f}±{feat_cv['roc_auc']['std']:.3f}")

    print("=== 3-class probe ===")
    multi = probe_multiclass_cv(mu, labels)
    print(
        f"  bal_acc={multi['balanced_accuracy']['mean']:.3f}±"
        f"{multi['balanced_accuracy']['std']:.3f} "
        f"macro_f1={multi['macro_f1']['mean']:.3f}±{multi['macro_f1']['std']:.3f}"
    )

    print(f"=== permutation test ({args.n_perm}x) ===")
    perm = permutation_test_binary(
        X_lat, y_bin, true_auc=lat_cv["roc_auc"]["mean"], n_perm=args.n_perm
    )
    print(f"  perm p={perm['permutation_p_value']:.4f}")

    n_by = {INT_TO_LABEL[i]: int(np.sum(labels == i)) for i in (0, 1, 2)}
    results = {
        "experiment": "caueeg_cvae_baseline_latent_linear_probe",
        "read_only": True,
        "twin_retrained": False,
        "checkpoint": str(ckpt),
        "latents_path": str(latents_path),
        "disease_label_for_encode": PROBE_DISEASE_LABEL,
        "disease_label_note": (
            "Fixed to 0 for all subjects so the linear probe cannot read "
            "diagnosis from the fusion disease bit."
        ),
        "n_subjects": int(len(labels)),
        "n_by_class": n_by,
        "latent_dim": int(mu.shape[1]),
        "latent_probe_dementia_vs_normal": {
            "cv": lat_cv,
            "in_sample_not_held_out": lat_in,
        },
        "feature_probe_theta_alpha_dementia_vs_normal": {
            "cv": feat_cv,
            "in_sample_not_held_out": feat_in,
        },
        "latent_probe_3class": multi,
        "permutation_test_latent": perm,
        "does_not_overwrite": "models/validation/layer5e_caueeg_results.json",
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # Compact JSON without full null list duplication in md; keep nulls in JSON
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[wrote] {out_json}")
    write_summary_md(out_md, results)

    print(
        f"LATENT PROBE: AUC = {lat_cv['roc_auc']['mean']:.2f} "
        f"(perm p = {perm['permutation_p_value']:.3f}) | "
        f"FEATURE PROBE: AUC = {feat_cv['roc_auc']['mean']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
