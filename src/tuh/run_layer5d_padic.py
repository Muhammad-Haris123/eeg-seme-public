"""
Layer 5d: P-ADIC AD vs HC external validation (Dryad 10.5061/dryad.8gtht76pw).

Addition only — does not replace TUH Layer 5 or OSF Layer 5b.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.utils.feature_processor import flatten_features
from src.models.data_loader import load_drug_embeddings
from src.models.inference import load_trained_model, simulate_post_drug_eeg
from src.tuh.process_padic_eeg import montage_report, process_padic_cohort
from src.tuh.process_tuh_eeg import TARGET_N_EPOCHS_FOR_PSD, unflatten_features
from src.tuh.tuh_validation import _band_change_vector, _cohens_d


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_processed_features(out_dir: Path) -> Tuple[np.ndarray, List[Dict]]:
    report = json.loads((out_dir / "processing_report.json").read_text(encoding="utf-8"))
    flats, metas = [], []
    for rec in report["results"]:
        if not rec.get("ok"):
            continue
        flat = np.load(rec["feature_path"]).astype(np.float32)
        metas.append(
            {
                "group": rec["group"],
                "index": rec["subject_col"],
                "recording_id": rec["recording_id"],
                "label": "ad" if rec["group"] == "AD" else "hc",
                "disease_label": int(rec.get("disease_label", 1 if rec["group"] == "AD" else 0)),
                "n_epochs_used": int(rec.get("n_epochs_used") or 0),
                "age": rec.get("age"),
                "sfreq": rec.get("sfreq"),
                "epoch_power_calibration": rec.get("epoch_power_calibration"),
            }
        )
        flats.append(flat)
    return np.stack(flats, axis=0), metas


def scale_check(flats: np.ndarray, metas: List[Dict], root: Path) -> Dict:
    train_bp = []
    for g in ("AD", "HC"):
        bp = np.load(root / "data" / "eeg_features" / f"{g}_band_powers.npy")
        train_bp.append(bp.reshape(bp.shape[0], -1))
    train_bp = np.concatenate(train_bp, axis=0)
    train_mean = float(np.mean(np.abs(train_bp)))

    # band powers live at flat[380:475]
    ext_bp = flats[:, 380:475]
    ext_mean = float(np.mean(np.abs(ext_bp)))
    ratio = ext_mean / max(train_mean, 1e-12)
    epochs = [m["n_epochs_used"] for m in metas]
    return {
        "train_abs_bandpower_mean": train_mean,
        "padic_abs_bandpower_mean": ext_mean,
        "bandpower_ratio_padic_over_train": ratio,
        "target_n_epochs": TARGET_N_EPOCHS_FOR_PSD,
        "epoch_count_summary": {
            "min": int(min(epochs)) if epochs else None,
            "median": float(np.median(epochs)) if epochs else None,
            "max": int(max(epochs)) if epochs else None,
            "mean": float(np.mean(epochs)) if epochs else None,
        },
    }


def run_twin(flats: np.ndarray, metas: List[Dict], root: Path, out_dir: Path) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = root / "models" / "checkpoints_constrained" / "checkpoint_constrained.pt"
    model = load_trained_model(ckpt, device)
    drug_emb = load_drug_embeddings()
    patients = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for flat, m in zip(flats, metas):
        eeg = unflatten_features(flat)
        disease = int(m["disease_label"])
        sims = simulate_post_drug_eeg(
            model, eeg, drug_emb, disease_label=disease, device=device, num_samples=5
        )
        base = sims["baseline"].mean(0)
        don = sims["donepezil"].mean(0)
        mem = sims["memantine"].mean(0)

        with torch.no_grad():
            psd = torch.FloatTensor(eeg["psd"]).unsqueeze(0).to(device)
            bp = torch.FloatTensor(eeg["band_powers"]).unsqueeze(0).to(device)
            coh = torch.FloatTensor(eeg["coherence"]).unsqueeze(0).to(device)
            plv = torch.FloatTensor(eeg["plv"]).unsqueeze(0).to(device)
            disease_t = torch.tensor([disease], dtype=torch.float32).to(device)
            eeg_lat = model.eeg_encoder(psd, bp, coh, plv)

            def _mu(drug_vec: np.ndarray) -> np.ndarray:
                de = torch.FloatTensor(drug_vec).unsqueeze(0).to(device)
                dl = model.drug_encoder(de)
                fused = model.fusion(eeg_lat, dl, disease_t)
                mu, _ = model.cvae.encode(fused)
                return mu.cpu().numpy().reshape(-1)

            mu_b = _mu(np.zeros(384, dtype=np.float32))
            mu_d = _mu(drug_emb["donepezil"])
            mu_m = _mu(drug_emb["memantine"])

        don_shift = float(np.linalg.norm(mu_d - mu_b))
        mem_shift = float(np.linalg.norm(mu_m - mu_b))
        mean_resp = 0.5 * (don_shift + mem_shift)
        sid = m["recording_id"]
        np.savez_compressed(
            out_dir / f"{sid}_sims.npz",
            baseline=base.astype(np.float32),
            donepezil=don.astype(np.float32),
            memantine=mem.astype(np.float32),
            mu_baseline=mu_b.astype(np.float32),
            mu_donepezil=mu_d.astype(np.float32),
            mu_memantine=mu_m.astype(np.float32),
        )
        patients.append(
            {
                "subject_id": sid,
                "label": m["label"],
                "group": m["group"],
                "disease_label": disease,
                "age": m.get("age"),
                "donepezil_latent_shift": don_shift,
                "memantine_latent_shift": mem_shift,
                "mean_drug_response": mean_resp,
            }
        )
    return {"patients": patients, "n_ok": len(patients), "checkpoint": str(ckpt)}


def discriminate(sim: Dict) -> Dict:
    ad = np.array([p["mean_drug_response"] for p in sim["patients"] if p["label"] == "ad"])
    hc = np.array([p["mean_drug_response"] for p in sim["patients"] if p["label"] == "hc"])
    u, p = stats.mannwhitneyu(ad, hc, alternative="two-sided")
    d = _cohens_d(ad, hc)
    direction_ok = float(np.mean(ad)) > float(np.mean(hc))
    passed = bool((p < 0.05 and direction_ok) or (abs(d) >= 0.3 and direction_ok))
    marginal = bool(direction_ok and (p < 0.1 or abs(d) >= 0.2))
    status = "PASS" if passed else ("MARGINAL" if marginal else "FAIL")
    return {
        "n_ad": int(len(ad)),
        "n_hc": int(len(hc)),
        "mean_ad": float(np.mean(ad)),
        "std_ad": float(np.std(ad, ddof=1)) if len(ad) > 1 else 0.0,
        "mean_hc": float(np.mean(hc)),
        "std_hc": float(np.std(hc, ddof=1)) if len(hc) > 1 else 0.0,
        "mannwhitney_u": float(u),
        "mannwhitney_p": float(p),
        "cohens_d": float(d),
        "direction_ad_gt_hc": direction_ok,
        "status": status,
        "pass": status == "PASS",
        "caveats": [
            "Recording-level units (AD multi-row cells) — not fully independent subjects",
            "Age imbalance (AD older than HC) may confound",
            "Eyes open/closed mixed; no EO/EC labels in mats",
            "Channel order assumed (not embedded in .mat)",
        ],
    }


def disease_label_ablation(flats, metas, root, out_dir) -> Dict:
    """Re-run twin with disease_label fixed to 0 or 1 for all."""
    out = {}
    for fixed in (0, 1):
        metas_f = [dict(m, disease_label=fixed) for m in metas]
        sim = run_twin(flats, metas_f, root, out_dir / f"ablation_disease_{fixed}")
        out[f"disease_fixed_{fixed}"] = discriminate(sim)
    return out


def pca_overlap(flats: np.ndarray, root: Path, fig_path: Path) -> Dict:
    train, train_y = [], []
    for g, lab in (("AD", 1), ("HC", 0)):
        psd = np.load(root / "data" / "eeg_features" / f"{g}_psd.npy")
        bp = np.load(root / "data" / "eeg_features" / f"{g}_band_powers.npy")
        coh = np.load(root / "data" / "eeg_features" / f"{g}_coherence.npy")
        plv = np.load(root / "data" / "eeg_features" / f"{g}_plv.npy")
        for i in range(psd.shape[0]):
            train.append(flatten_features(psd[i], bp[i], coh[i], plv[i]))
            train_y.append(lab)
    train = np.stack(train)
    train_y = np.asarray(train_y)

    pca = PCA(2, random_state=42)
    t2 = pca.fit_transform(train)
    x2 = pca.transform(flats)
    c = t2.mean(0)
    r95 = float(np.percentile(np.linalg.norm(t2 - c, axis=1), 95))
    overlap = float(np.mean(np.linalg.norm(x2 - c, axis=1) <= r95))

    train_s, x_s = train[:, :475], flats[:, :475]
    pca_s = PCA(2, random_state=42)
    ts = pca_s.fit_transform(train_s)
    xs = pca_s.transform(x_s)
    cs = ts.mean(0)
    r95s = float(np.percentile(np.linalg.norm(ts - cs, axis=1), 95))
    overlap_s = float(np.mean(np.linalg.norm(xs - cs, axis=1) <= r95s))

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    ax.scatter(t2[train_y == 1, 0], t2[train_y == 1, 1], c="#c44e52", s=28, alpha=0.7, label="Train AD")
    ax.scatter(t2[train_y == 0, 0], t2[train_y == 0, 1], c="#4c72b0", s=28, alpha=0.7, label="Train HC")
    ax.scatter(
        x2[:, 0],
        x2[:, 1],
        c="#55a868",
        s=36,
        marker="^",
        alpha=0.85,
        edgecolors="k",
        linewidths=0.3,
        label="P-ADIC external",
    )
    ax.add_patch(plt.Circle(c, r95, fill=False, ls="--", color="gray"))
    ax.set_title("Layer 5d: P-ADIC vs Training PCA")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    if overlap >= 0.6:
        grade = "good"
    elif overlap >= 0.3:
        grade = "moderate"
    else:
        grade = "poor"
    return {
        "pca_overlap": grade,
        "overlap_fraction_within_train_r95": overlap,
        "spectral_overlap_fraction": overlap_s,
        "figure": str(fig_path),
    }


def cross_dataset(sim: Dict, root: Path, val_dir: Path) -> Dict:
    cohort_dir = root / "models" / "simulations_constrained_full"
    base = np.load(cohort_dir / "simulated_baseline.npy").mean(1)
    don = np.load(cohort_dir / "simulated_donepezil.npy").mean(1)
    mem = np.load(cohort_dir / "simulated_memantine.npy").mean(1)
    cohort_don = (don - base).mean(0)
    cohort_mem = (mem - base).mean(0)

    don_e, mem_e = [], []
    for p in sim["patients"]:
        z = np.load(val_dir / f"{p['subject_id']}_sims.npz")
        don_e.append(z["donepezil"] - z["baseline"])
        mem_e.append(z["memantine"] - z["baseline"])
    ext_don = np.mean(np.stack(don_e), 0)
    ext_mem = np.mean(np.stack(mem_e), 0)

    slices = {
        "delta": slice(380, 399),
        "theta": slice(399, 418),
        "alpha": slice(418, 437),
        "beta": slice(437, 456),
        "connectivity": slice(475, 1330),
    }

    def agree(a, b):
        ok = tot = 0
        for sl in slices.values():
            tot += 1
            if np.sign(np.mean(a[sl])) == np.sign(np.mean(b[sl])) or (
                abs(np.mean(a[sl])) < 1e-8 and abs(np.mean(b[sl])) < 1e-8
            ):
                ok += 1
        return ok, tot

    d_a, d_t = agree(ext_don, cohort_don)
    m_a, m_t = agree(ext_mem, cohort_mem)
    r_d = float(np.corrcoef(ext_don, cohort_don)[0, 1])
    r_m = float(np.corrcoef(ext_mem, cohort_mem)[0, 1])
    return {
        "direction_agreement_total": f"{d_a + m_a}/{d_t + m_t}",
        "donepezil_direction_agreement": f"{d_a}/{d_t}",
        "memantine_direction_agreement": f"{m_a}/{m_t}",
        "effect_magnitude_correlation": float(np.nanmean([r_d, r_m])),
        "n_used": len(don_e),
    }


def confirm_prior_untouched(root: Path) -> Dict:
    v3_path = root / "models" / "validation" / "complete_validation_report_v3.json"
    out = {"v3_exists": v3_path.exists()}
    if v3_path.exists():
        v3 = json.loads(v3_path.read_text(encoding="utf-8"))
        out["layer5_status"] = (v3.get("layer5") or {}).get("status")
        out["layer5_p"] = (v3.get("layer5") or {}).get("mannwhitney_p") or (
            (v3.get("layer5") or {}).get("discrimination") or {}
        ).get("mannwhitney_p")
        out["layer5b_present"] = "layer5b_ad_labeled_external" in v3 or "layer5b" in str(v3.keys())
    return out


def annotate_v3(root: Path, layer5d: Dict) -> None:
    v3_path = root / "models" / "validation" / "complete_validation_report_v3.json"
    if not v3_path.exists():
        return
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    # fingerprint priors before write
    prior_l5 = json.dumps(v3.get("layer5"), sort_keys=True)
    prior_l5b = json.dumps(v3.get("layer5b_ad_labeled_external"), sort_keys=True)
    v3["layer5d_padic_external"] = layer5d
    v3_path.write_text(json.dumps(v3, indent=2), encoding="utf-8")
    v3_after = json.loads(v3_path.read_text(encoding="utf-8"))
    assert json.dumps(v3_after.get("layer5"), sort_keys=True) == prior_l5
    assert json.dumps(v3_after.get("layer5b_ad_labeled_external"), sort_keys=True) == prior_l5b


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true", help="Run disease_label fixed 0/1 twin re-sims")
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()

    root = _root()
    padic_raw = Path(r"E:\padic_external")
    data_dir = root / "data" / "padic_external"
    val_dir = data_dir / "validation"
    fig_dir = root / "models" / "validation" / "figures"
    val_dir.mkdir(parents=True, exist_ok=True)

    print("=== montage report (before processing) ===")
    print(json.dumps(montage_report(), indent=2))

    report_path = data_dir / "processing_report.json"
    if (not report_path.exists()) or args.force_extract:
        print("=== extracting features ===")
        process_padic_cohort(padic_raw, data_dir)
    else:
        print(f"[skip extract] existing {report_path}")

    flats, metas = load_processed_features(data_dir)
    print(f"Loaded features: {flats.shape[0]} recordings x {flats.shape[1]}D")

    scale = scale_check(flats, metas, root)
    print("=== scale check ===")
    print(json.dumps(scale, indent=2))

    print("=== twin ===")
    sim = run_twin(flats, metas, root, val_dir)
    disc = discriminate(sim)
    print("=== discrimination ===")
    print(json.dumps(disc, indent=2))

    print("=== PCA ===")
    pca = pca_overlap(flats, root, fig_dir / "layer5d_padic_pca.png")
    print(json.dumps(pca, indent=2))

    print("=== cross-dataset ===")
    cross = cross_dataset(sim, root, val_dir)
    print(json.dumps(cross, indent=2))

    ablate = {}
    if args.ablation:
        print("=== disease-label ablation ===")
        ablate = disease_label_ablation(flats, metas, root, val_dir)
    else:
        print("[skip] disease-label ablation (pass --ablation to enable)")

    prior = confirm_prior_untouched(root)

    layer5d = {
        "dataset": "P-ADIC Dryad 10.5061/dryad.8gtht76pw",
        "status": disc["status"],
        "discrimination": disc,
        "scale_check": scale,
        "pca": pca,
        "cross_dataset": cross,
        "disease_label_ablation": ablate,
        "montage": montage_report(),
        "n_features": int(flats.shape[0]),
        "prior_layers_untouched": prior,
        "addition_only": True,
    }
    out_json = root / "models" / "validation" / "layer5d_padic_results.json"
    out_json.write_text(json.dumps(layer5d, indent=2), encoding="utf-8")
    annotate_v3(root, layer5d)
    print(f"[wrote] {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
