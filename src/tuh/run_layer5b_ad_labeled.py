"""
Layer 5b: AD-labeled external validation (OSF Eyes_closed AD vs HC).

Uses existing OSF features (clinically labeled AD/Healthy, proper 10-20 montage).
Applies TUH-style 1/N spectral calibration (TARGET_N_EPOCHS_FOR_PSD=864) because
OSF Phase-1 stored only 1 epoch/subject → absolute PSD/BP inflated.

Does NOT modify shared src/eeg/* or TUH Layer-5 artifacts.
"""

from __future__ import annotations

import json
import os
import shutil
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
from src.tuh.process_tuh_eeg import TARGET_N_EPOCHS_FOR_PSD, calibrate_flat_features_for_epoch_count
from src.tuh.tuh_validation import _cohens_d, _band_change_vector, analyze_cross_dataset


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prepare_ad_labeled_external(root: Path) -> Path:
    """Copy OSF features into data/ad_labeled_external/ with provenance README."""
    src = root / "data" / "external_osf" / "features"
    dest = root / "data" / "ad_labeled_external"
    feat_dest = dest / "features"
    feat_dest.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*"):
        shutil.copy2(f, feat_dest / f.name)

    readme = f"""# AD-labeled external dataset (Layer 5b)

## Chosen source: OSF Eyes_closed (already imported as `data/external_osf`)

- **Labels:** AD vs Healthy (mapped to HC) — diagnosis groups from OSF testing_data folders
- **Montage:** Standard 10–20, 19 channels (matches training)
- **Format:** Per-channel `.txt` → Phase-1 features (2185-D layout)
- **N:** AD=80, HC=12 (imbalanced; HC small — power/limitation flagged)
- **Sample rate:** 256 Hz native → resampled to 500 Hz in Phase-1 import
- **Why not Miltiadous ds004504?** That dataset *is* the training cohort (AD≈36–37, HC=29). Using it as \"external\" would be circular.
- **Why not CAUEEG?** Requires ethics-form gated download; not freely fetchable in this session.
- **Why not BrainLAT as primary?** AD/HC clinical labels exist (35/31) but 128-ch A1–A19 has **no 10–20 mapping**; prior external eval was ~chance. Kept as secondary caveat in diagnosis MD.

## Epoch / scale note

`processed/{{AD,HC}}_preprocessed.npy` shapes are `(N, 1, 19, 2000)` → **1 epoch** of 4 s.
Phase-1 spectral path averages z-scored epochs before Welch ⇒ power ∝ 1/N.
Calibration uses `TARGET_N_EPOCHS_FOR_PSD={TARGET_N_EPOCHS_FOR_PSD}` (same as TUH fix).
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")
    return dest


def load_group_features(feat_dir: Path, group: str) -> Tuple[np.ndarray, List[Dict]]:
    psd = np.load(feat_dir / f"{group}_psd.npy")
    bp = np.load(feat_dir / f"{group}_band_powers.npy")
    coh = np.load(feat_dir / f"{group}_coherence.npy")
    plv = np.load(feat_dir / f"{group}_plv.npy")
    flats = []
    meta = []
    for i in range(psd.shape[0]):
        flat = flatten_features(psd[i], bp[i], coh[i], plv[i]).astype(np.float32)
        flats.append(flat)
        meta.append(
            {
                "group": group,
                "index": i,
                "label": "ad" if group == "AD" else "hc",
                "disease_label": 1 if group == "AD" else 0,
                "n_epochs_used": 1,  # from processed shapes (N, 1, 19, 2000)
            }
        )
    return np.stack(flats, axis=0), meta


def calibrate_cohort(flats: np.ndarray, metas: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    calibrated = []
    new_metas = []
    for flat, m in zip(flats, metas):
        cal, scale = calibrate_flat_features_for_epoch_count(
            flat, m["n_epochs_used"], target_n_epochs=TARGET_N_EPOCHS_FOR_PSD
        )
        nm = dict(m)
        nm["epoch_power_calibration"] = {
            "applied": True,
            "n_epochs_used": m["n_epochs_used"],
            "target_n_epochs": TARGET_N_EPOCHS_FOR_PSD,
            "scale": float(scale),
        }
        calibrated.append(cal)
        new_metas.append(nm)
    return np.stack(calibrated, axis=0), new_metas


def unflatten(flat: np.ndarray) -> Dict[str, np.ndarray]:
    from src.tuh.process_tuh_eeg import unflatten_features

    return unflatten_features(flat)


def run_twin(flats: np.ndarray, metas: List[Dict], root: Path, out_dir: Path) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = root / "models" / "checkpoints_constrained" / "checkpoint_constrained.pt"
    model = load_trained_model(ckpt, device)
    drug_emb = load_drug_embeddings()
    patients = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for flat, m in zip(flats, metas):
        eeg = unflatten(flat)
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
        sid = f"{m['group']}_{m['index']:03d}"
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
                "donepezil_latent_shift": don_shift,
                "memantine_latent_shift": mem_shift,
                "mean_drug_response": mean_resp,
                "donepezil_band_changes": _band_change_vector(don, base),
                "memantine_band_changes": _band_change_vector(mem, base),
                "patient_id": sid,
                "session_id": "s001",
            }
        )

    return {"patients": patients, "n_ok": len(patients), "checkpoint": str(ckpt)}


def discriminate_ad_vs_hc(sim: Dict) -> Dict:
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
        "caveat": "HC n=12 limits statistical power; interpret p/d cautiously.",
    }


def pca_overlap(flats: np.ndarray, root: Path, fig_path: Path) -> Dict:
    # training flats
    train = []
    train_y = []
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
    dist = float(np.mean(np.linalg.norm(flats - train.mean(0), axis=1)))

    # spectral-only
    train_s = train[:, :475]
    x_s = flats[:, :475]
    pca_s = PCA(2, random_state=42)
    ts = pca_s.fit_transform(train_s)
    xs = pca_s.transform(x_s)
    cs = ts.mean(0)
    r95s = float(np.percentile(np.linalg.norm(ts - cs, axis=1), 95))
    overlap_s = float(np.mean(np.linalg.norm(xs - cs, axis=1) <= r95s))

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    ax.scatter(t2[train_y == 1, 0], t2[train_y == 1, 1], c="#c44e52", s=28, alpha=0.7, label="Train AD")
    ax.scatter(t2[train_y == 0, 0], t2[train_y == 0, 1], c="#4c72b0", s=28, alpha=0.7, label="Train HC")
    ax.scatter(x2[:, 0], x2[:, 1], c="#55a868", s=36, marker="^", alpha=0.85, edgecolors="k", linewidths=0.3, label="OSF external")
    ax.add_patch(plt.Circle(c, r95, fill=False, ls="--", color="gray"))
    ax.set_title("Layer 5b: OSF vs Training PCA")
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
        "mean_feature_distance": dist,
        "spectral_overlap_fraction": overlap_s,
        "figure": str(fig_path),
    }


def plot_disc(disc: Dict, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    ax.bar(
        ["AD", "HC"],
        [disc["mean_ad"], disc["mean_hc"]],
        yerr=[disc["std_ad"], disc["std_hc"]],
        color=["#c44e52", "#4c72b0"],
        alpha=0.85,
        capsize=4,
        width=0.55,
    )
    ax.set_ylabel("Mean drug response (latent ‖Δμ‖)")
    ax.set_title(f"Layer 5b OSF AD vs HC\np={disc['mannwhitney_p']:.4f}, d={disc['cohens_d']:.2f}")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def cross_dataset_osf(sim: Dict, root: Path, val_dir: Path) -> Dict:
    # reuse analyze_cross_dataset by pointing patient sims into expected naming
    # It looks under data/tuh_validation — instead compute here similarly
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
    tuh_don = np.mean(np.stack(don_e), 0)
    tuh_mem = np.mean(np.stack(mem_e), 0)

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

    d_a, d_t = agree(tuh_don, cohort_don)
    m_a, m_t = agree(tuh_mem, cohort_mem)
    r_d = float(np.corrcoef(tuh_don, cohort_don)[0, 1])
    r_m = float(np.corrcoef(tuh_mem, cohort_mem)[0, 1])
    return {
        "direction_agreement_total": f"{d_a + m_a}/{d_t + m_t}",
        "donepezil_direction_agreement": f"{d_a}/{d_t}",
        "memantine_direction_agreement": f"{m_a}/{m_t}",
        "effect_magnitude_correlation": float(np.nanmean([r_d, r_m])),
        "donepezil_effect_corr": r_d,
        "memantine_effect_corr": r_m,
        "n_used": len(don_e),
    }


def layer14_fingerprint(root: Path) -> Dict:
    v2 = json.loads((root / "models" / "validation" / "complete_validation_report_v2.json").read_text())
    return {
        "layer1_mse": v2["layer1"]["constrained"]["mse_mean"],
        "layer1_pearson": v2["layer1"]["constrained"]["pearson_mean"],
        "layer2_match": v2["layer2"]["constrained"]["n_match"],
        "layer3": v2["layer3"].get("status"),
        "layer4": v2["layer4"].get("status"),
    }


def main() -> int:
    root = _project_root()
    dest = prepare_ad_labeled_external(root)
    feat_dir = dest / "features"
    val_dir = root / "data" / "ad_labeled_external" / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    ad_x, ad_m = load_group_features(feat_dir, "AD")
    hc_x, hc_m = load_group_features(feat_dir, "HC")
    flats = np.concatenate([ad_x, hc_x], 0)
    metas = ad_m + hc_m

    train_bp = float(
        np.concatenate(
            [
                np.load(root / "data" / "eeg_features" / "AD_band_powers.npy"),
                np.load(root / "data" / "eeg_features" / "HC_band_powers.npy"),
            ]
        ).mean()
    )
    before_bp = float(np.mean([f[380:475].mean() for f in flats]))
    flats_c, metas_c = calibrate_cohort(flats, metas)
    after_bp = float(np.mean([f[380:475].mean() for f in flats_c]))

    # save calibrated flats
    cal_dir = dest / "features_calibrated"
    cal_dir.mkdir(exist_ok=True)
    for flat, m in zip(flats_c, metas_c):
        np.save(cal_dir / f"{m['group']}_{m['index']:03d}_features.npy", flat)

    print("Scale: before_bp", before_bp, "after_bp", after_bp, "train_bp", train_bp)
    print("ratio before", before_bp / train_bp, "after", after_bp / train_bp)

    print("Running twin...")
    sim = run_twin(flats_c, metas_c, root, val_dir)
    disc = discriminate_ad_vs_hc(sim)
    plot_disc(disc, val_dir / "fig_layer5b_ad_vs_hc.png")
    pca = pca_overlap(flats_c, root, val_dir / "fig_layer5b_pca_overlap.png")
    cross = cross_dataset_osf(sim, root, val_dir)

    # TUH baseline from v3 / diagnosis
    tuh = json.loads(
        (root / "models" / "validation" / "complete_validation_report_v3.json").read_text(encoding="utf-8")
    )["layer5"]["abnormal_vs_normal"]

    l14 = layer14_fingerprint(root)
    # confirm TUH files untouched (mtime / hash light check: report still has same p)
    tuh_p_unchanged = abs(float(tuh["mannwhitney_p"]) - 0.272471740367322) < 1e-9

    results = {
        "dataset": {
            "name": "OSF Eyes_closed (data/external_osf → data/ad_labeled_external)",
            "n_ad": disc["n_ad"],
            "n_hc": disc["n_hc"],
            "montage": "10-20 19ch",
            "label_quality": "Folder labels AD vs Healthy from OSF testing_data; clinical confirmation depth not fully documented in local metadata — treat as research AD/HC partitions.",
            "why_not_miltiadous": "ds004504 is the training cohort",
            "why_not_caueeg": "ethics-gated download",
            "why_not_brainlat_primary": "no A1–A19→10-20 map",
        },
        "calibration": {
            "n_epochs_used": 1,
            "target_n_epochs": TARGET_N_EPOCHS_FOR_PSD,
            "band_power_mean_before": before_bp,
            "band_power_mean_after": after_bp,
            "train_band_power_mean": train_bp,
            "ratio_before": before_bp / train_bp,
            "ratio_after": after_bp / train_bp,
        },
        "discrimination": disc,
        "feature_distribution": pca,
        "cross_dataset": cross,
        "comparison_vs_tuh_layer5": {
            "tuh_n_abnormal": tuh.get("n_abnormal"),
            "tuh_n_normal": tuh.get("n_normal"),
            "tuh_p": tuh.get("mannwhitney_p"),
            "tuh_d": tuh.get("cohens_d"),
            "tuh_status": tuh.get("status"),
            "osf_p": disc["mannwhitney_p"],
            "osf_d": disc["cohens_d"],
            "osf_status": disc["status"],
        },
        "layers_1_4_fingerprint": l14,
        "tuh_layer5_untouched": tuh_p_unchanged,
        "shared_eeg_modules_modified": False,
    }

    out_json = val_dir / "layer5b_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps(results["discrimination"], indent=2))
    print("cross", results["cross_dataset"])
    print("pca", results["feature_distribution"])
    print("Saved", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
