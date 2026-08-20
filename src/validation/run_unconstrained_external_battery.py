"""
Matched unconstrained external battery (Plan-to-8.3 B1).

Uses fold_0 unconstrained checkpoint (same selection rule as constrained
`checkpoint_constrained.pt` = fold_0_best). Direction is scored against the
**constrained** training mean effect signature (same reference as Layer 5*).

Does NOT modify `complete_validation_report_v3.json`.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tuh.process_tuh_eeg import TARGET_N_EPOCHS_FOR_PSD, calibrate_flat_features_for_epoch_count
from src.validation.direction_metrics import (
    extract_baseline_mu,
    load_constrained_training_effects,
    mw_groups,
    score_effects_vs_signature,
    simulate_flat_cohort,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_ckpt(root: Path) -> Path:
    return root / "models" / "checkpoints_unconstrained" / "checkpoint_unconstrained.pt"


# ---- loaders ----

def load_tuh(root: Path) -> Tuple[np.ndarray, List[int], List[str], List[str]]:
    report = json.loads((root / "data" / "tuh_features" / "processing_report.json").read_text(encoding="utf-8"))
    flats, diseases, sids, labels = [], [], [], []
    for rec in report["results"]:
        if not rec.get("ok"):
            continue
        path = rec.get("feature_path")
        if not path or not Path(path).exists():
            # fall back to local tuh_features naming
            pid, sid = rec.get("patient_id"), rec.get("session_id")
            alt = root / "data" / "tuh_features" / f"{pid}_{sid}_features.npy"
            if not alt.exists():
                continue
            path = str(alt)
        flat = np.load(path).astype(np.float32)
        lab = rec.get("label", "unknown")
        disease = 1 if lab == "abnormal" else 0
        flats.append(flat)
        diseases.append(disease)
        sids.append(f"{rec.get('patient_id')}_{rec.get('session_id')}")
        labels.append(lab)
    return np.stack(flats), diseases, sids, labels


def load_osf(root: Path) -> Tuple[np.ndarray, List[int], List[str], List[str]]:
    """Load OSF AD/HC with same calibration as Layer 5b when needed."""
    cal_dir = root / "data" / "ad_labeled_external" / "features_calibrated"
    flats, diseases, sids, labels = [], [], [], []

    if cal_dir.exists() and any(cal_dir.glob("*_features.npy")):
        for p in sorted(cal_dir.glob("*_features.npy")):
            stem = p.stem  # AD_000_features -> wait stem is AD_000_features
            name = p.name
            if name.startswith("AD_"):
                lab, disease = "ad", 1
            elif name.startswith("HC_"):
                lab, disease = "hc", 0
            else:
                continue
            flats.append(np.load(p).astype(np.float32))
            diseases.append(disease)
            sids.append(p.stem.replace("_features", ""))
            labels.append(lab)
        print(f"[osf] loaded {len(flats)} calibrated flats from {cal_dir}")
        return np.stack(flats), diseases, sids, labels

    # Fallback: rebuild from group arrays + calibrate
    from api.utils.feature_processor import flatten_features

    feat_dir = root / "data" / "ad_labeled_external" / "features"
    for group, lab, disease in (("AD", "ad", 1), ("HC", "hc", 0)):
        psd = np.load(feat_dir / f"{group}_psd.npy")
        bp = np.load(feat_dir / f"{group}_band_powers.npy")
        coh = np.load(feat_dir / f"{group}_coherence.npy")
        plv = np.load(feat_dir / f"{group}_plv.npy")
        for i in range(psd.shape[0]):
            flat = flatten_features(psd[i], bp[i], coh[i], plv[i]).astype(np.float32)
            flat, _ = calibrate_flat_features_for_epoch_count(
                flat, n_epochs_used=1, target_n_epochs=TARGET_N_EPOCHS_FOR_PSD
            )
            flats.append(flat)
            diseases.append(disease)
            sids.append(f"{group}_{i:03d}")
            labels.append(lab)
    print(f"[osf] loaded {len(flats)} from group arrays + calibration")
    return np.stack(flats), diseases, sids, labels


def load_padic(root: Path) -> Tuple[np.ndarray, List[int], List[str], List[str]]:
    report = json.loads((root / "data" / "padic_external" / "processing_report.json").read_text(encoding="utf-8"))
    flats, diseases, sids, labels = [], [], [], []
    for rec in report["results"]:
        if not rec.get("ok"):
            continue
        flat = np.load(rec["feature_path"]).astype(np.float32)
        group = rec["group"]
        disease = int(rec.get("disease_label", 1 if group == "AD" else 0))
        flats.append(flat)
        diseases.append(disease)
        sids.append(rec["recording_id"])
        labels.append("ad" if group == "AD" else "hc")
    return np.stack(flats), diseases, sids, labels


def load_caueeg(root: Path) -> Tuple[np.ndarray, List[int], List[str], List[str]]:
    report = json.loads((root / "data" / "caueeg_external" / "processing_report.json").read_text(encoding="utf-8"))
    flats, diseases, sids, labels = [], [], [], []
    for rec in report["results"]:
        if not rec.get("ok"):
            continue
        flat = np.load(rec["feature_path"]).astype(np.float32)
        lab = rec.get("label") or rec.get("group") or "unknown"
        # disease for twin: dementia/mci -> 1, normal -> 0 (match existing runners when present)
        lab_l = str(lab).lower()
        if lab_l in ("dementia", "dem", "ad"):
            disease = 1
            lab_std = "dementia"
        elif lab_l in ("mci",):
            disease = 1
            lab_std = "mci"
        elif lab_l in ("normal", "nor", "hc", "healthy"):
            disease = 0
            lab_std = "normal"
        else:
            disease = int(rec.get("disease_label", 0))
            lab_std = lab_l
        flats.append(flat)
        diseases.append(disease)
        sid = rec.get("recording_id") or rec.get("subject_id")
        if not sid:
            sid = Path(rec["feature_path"]).stem.replace("_features", "")
        sids.append(str(sid))
        labels.append(lab_std)
    return np.stack(flats), diseases, sids, labels


def theta_alpha(flats: np.ndarray) -> np.ndarray:
    bp = flats[:, 380:475].reshape(-1, 19, 5)
    theta = bp[:, :, 1].mean(axis=1)
    alpha = bp[:, :, 2].mean(axis=1)
    return (theta / np.maximum(alpha, 1e-12)).astype(np.float64)


def latent_probe_auc(mu: np.ndarray, y: np.ndarray) -> Dict:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if len(np.unique(y)) < 2 or len(y) < 20:
        return {"roc_auc_mean": float("nan"), "n": int(len(y))}
    proba = cross_val_predict(pipe, mu, y, cv=cv, method="predict_proba")[:, 1]
    # also per-fold mean
    aucs = []
    for tr, te in cv.split(mu, y):
        pipe.fit(mu[tr], y[tr])
        p = pipe.predict_proba(mu[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return {
        "roc_auc_oof": float(roc_auc_score(y, proba)),
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
    }


def run_cohort(
    name: str,
    flats: np.ndarray,
    diseases: List[int],
    sids: List[str],
    labels: List[str],
    ckpt: Path,
    sim_dir: Path,
    signature: Dict,
    magnitude_contrast: str,
) -> Dict:
    print(f"\n=== {name}: twin ({flats.shape[0]} subjects) ===")
    sim = simulate_flat_cohort(flats, diseases, sids, ckpt, sim_dir)
    direction = score_effects_vs_signature(sim["don_effects"], sim["mem_effects"], signature)

    # attach labels onto patients
    lab_map = {s: lab for s, lab in zip(sids, labels)}
    for p in sim["patients"]:
        p["label"] = lab_map.get(p["subject_id"], "unknown")

    mag = {}
    if magnitude_contrast == "abnormal_vs_normal":
        a = [p["mean_drug_response"] for p in sim["patients"] if p["label"] == "abnormal"]
        b = [p["mean_drug_response"] for p in sim["patients"] if p["label"] == "normal"]
        mag = mw_groups(a, b)
        mag["contrast"] = "abnormal_vs_normal"
    elif magnitude_contrast == "ad_vs_hc":
        a = [p["mean_drug_response"] for p in sim["patients"] if p["label"] in ("ad", "AD")]
        b = [p["mean_drug_response"] for p in sim["patients"] if p["label"] in ("hc", "HC", "healthy")]
        mag = mw_groups(a, b)
        mag["contrast"] = "ad_vs_hc"
    elif magnitude_contrast == "dementia_vs_normal":
        a = [p["mean_drug_response"] for p in sim["patients"] if p["label"] == "dementia"]
        b = [p["mean_drug_response"] for p in sim["patients"] if p["label"] == "normal"]
        mag = mw_groups(a, b)
        mag["contrast"] = "dementia_vs_normal"

    # drop large arrays from saved JSON
    slim = {k: v for k, v in sim.items() if k not in ("don_effects", "mem_effects")}
    return {
        "simulation": slim,
        "direction": direction,
        "magnitude": mag,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--skip-caueeg-probe", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="Comma list: tuh,osf,padic,caueeg")
    args = ap.parse_args()

    root = _root()
    ckpt = Path(args.checkpoint) if args.checkpoint else _default_ckpt(root)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    only = {x.strip() for x in args.only.split(",")} if args.only else {"tuh", "osf", "padic", "caueeg"}
    signature = load_constrained_training_effects(root)
    out_root = root / "models" / "validation" / "unconstrained_sims"
    out_root.mkdir(parents=True, exist_ok=True)

    pre_registered = {
        "before_looking_at_results": (
            "If constrained >> unconstrained on direction and unconstrained >= "
            "constrained on latent AUC: keep trade-off as main result. "
            "If both ~10/10 direction: soften constraint-causal direction; "
            "emphasize domain-shift preservation + encoding. "
            "If both weak latent AUC: soften constraint-compresses-diagnosis."
        )
    }

    results: Dict = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(ckpt),
        "checkpoint_selection": str(
            root / "models" / "checkpoints_unconstrained" / "CHECKPOINT_SELECTION.md"
        ),
        "direction_reference_signature": signature["source"],
        "pre_registered_interpretation": pre_registered,
        "cohorts": {},
    }

    if "tuh" in only:
        flats, dis, sids, labs = load_tuh(root)
        results["cohorts"]["tuh"] = run_cohort(
            "TUH", flats, dis, sids, labs, ckpt, out_root / "tuh", signature, "abnormal_vs_normal"
        )
        print("TUH direction", results["cohorts"]["tuh"]["direction"]["direction_agreement_total"])

    if "osf" in only:
        flats, dis, sids, labs = load_osf(root)
        results["cohorts"]["osf"] = run_cohort(
            "OSF", flats, dis, sids, labs, ckpt, out_root / "osf", signature, "ad_vs_hc"
        )
        print("OSF direction", results["cohorts"]["osf"]["direction"]["direction_agreement_total"])

    if "padic" in only:
        flats, dis, sids, labs = load_padic(root)
        results["cohorts"]["padic"] = run_cohort(
            "P-ADIC", flats, dis, sids, labs, ckpt, out_root / "padic", signature, "ad_vs_hc"
        )
        print("P-ADIC direction", results["cohorts"]["padic"]["direction"]["direction_agreement_total"])

    if "caueeg" in only:
        flats, dis, sids, labs = load_caueeg(root)
        results["cohorts"]["caueeg"] = run_cohort(
            "CAUEEG",
            flats,
            dis,
            sids,
            labs,
            ckpt,
            out_root / "caueeg",
            signature,
            "dementia_vs_normal",
        )
        print(
            "CAUEEG direction",
            results["cohorts"]["caueeg"]["direction"]["direction_agreement_total"],
        )
        if not args.skip_caueeg_probe:
            print("=== CAUEEG latent probe (unconstrained encoder) ===")
            mu = extract_baseline_mu(flats, ckpt, disease_label=0.0)
            y_all = np.asarray(labs)
            mask = np.isin(y_all, ["dementia", "normal"])
            y = (y_all[mask] == "dementia").astype(np.int64)
            X = mu[mask]
            lat = latent_probe_auc(X, y)
            feat = latent_probe_auc(theta_alpha(flats)[mask].reshape(-1, 1), y)
            results["cohorts"]["caueeg"]["encoding"] = {
                "latent_probe": lat,
                "theta_alpha_probe": feat,
            }
            print("latent AUC", lat, "theta/alpha", feat)

    out_json = root / "models" / "validation" / "unconstrained_external_battery.json"
    out_md = root / "models" / "validation" / "unconstrained_external_battery.md"
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Unconstrained external battery",
        "",
        f"- Checkpoint: `{ckpt}`",
        f"- Direction reference: `{signature['source']}`",
        "",
    ]
    for name, block in results["cohorts"].items():
        d = block.get("direction", {})
        m = block.get("magnitude", {})
        lines.append(
            f"- **{name}**: direction {d.get('direction_agreement_total')} "
            f"(r={d.get('effect_magnitude_correlation')}); "
            f"magnitude p={m.get('mannwhitney_p')} d={m.get('cohens_d')}"
        )
        enc = block.get("encoding")
        if enc:
            lines.append(
                f"  - latent probe AUC={enc['latent_probe'].get('roc_auc_mean')}; "
                f"theta/alpha AUC={enc['theta_alpha_probe'].get('roc_auc_mean')}"
            )
    lines.append("")
    lines.append(pre_registered["before_looking_at_results"])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[wrote] {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
