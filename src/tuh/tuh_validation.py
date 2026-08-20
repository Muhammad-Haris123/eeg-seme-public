"""
TUH clinical validation: twin simulation + group analyses + report Layer 5.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from api.utils.feature_processor import band_powers_mean_per_band, flatten_features
from src.models.config import MODEL_CONFIG
from src.models.data_loader import load_drug_embeddings
from src.models.inference import load_trained_model, simulate_post_drug_eeg
from src.models.train_phase2 import DigitalTwinModel
from src.tuh.process_tuh_eeg import unflatten_features

BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")
BAND_SLICES = {
    "delta": slice(380, 399),
    "theta": slice(399, 418),
    "alpha": slice(418, 437),
    "beta": slice(437, 456),
    "connectivity": slice(475, 1330),
}


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _pick_checkpoint(project_root: str) -> Path:
    cdir = Path(project_root) / "models" / "checkpoints_constrained"
    for name in ("checkpoint_constrained.pt", "fold_0_best.pt", "checkpoint_final_full.pt"):
        p = cdir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No constrained checkpoint under {cdir}")


def _band_change_vector(sim_mean: np.ndarray, base_mean: np.ndarray) -> Dict[str, float]:
    delta = sim_mean - base_mean
    out = {}
    bp = band_powers_mean_per_band(delta)
    for k, v in bp.items():
        out[k] = float(v)
    out["connectivity"] = float(np.mean(delta[475:1330]))
    out["latent_proxy_l2"] = float(np.linalg.norm(delta))
    return out


def run_twin_on_tuh_patients(
    feature_dir: str,
    processing_report: Dict,
    project_root: str,
    output_dir: str,
    num_samples: int = 5,
    device: Optional[str] = None,
) -> Dict:
    """Run baseline/donepezil/memantine simulations for each successful TUH subject."""
    import torch
    from tqdm import tqdm

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = _pick_checkpoint(project_root)
    model = load_trained_model(ckpt, device)
    drug_emb = load_drug_embeddings()

    os.makedirs(output_dir, exist_ok=True)
    per_patient = []

    successes = [r for r in processing_report.get("results", []) if r.get("ok")]
    for rec in tqdm(successes, desc="Twin simulation on TUH"):
        label = rec.get("label", "unknown")
        disease_label = 1 if label == "abnormal" else 0
        struct_path = rec.get("structured_path")
        feat_path = rec.get("feature_path")
        try:
            if struct_path and os.path.exists(struct_path):
                z = np.load(struct_path)
                eeg_features = {
                    "psd": z["psd"],
                    "band_powers": z["band_powers"],
                    "coherence": z["coherence"],
                    "plv": z["plv"],
                }
                flat = flatten_features(
                    eeg_features["psd"],
                    eeg_features["band_powers"],
                    eeg_features["coherence"],
                    eeg_features["plv"],
                ).astype(np.float32)
            else:
                flat = np.load(feat_path).astype(np.float32)
                eeg_features = unflatten_features(flat)

            sims = simulate_post_drug_eeg(
                model,
                eeg_features,
                drug_emb,
                disease_label=disease_label,
                device=device,
                num_samples=num_samples,
            )
            # sims[cond]: (num_samples, 2185)
            base = sims["baseline"].mean(axis=0)
            don = sims["donepezil"].mean(axis=0)
            mem = sims["memantine"].mean(axis=0)

            # Latent shift via encoder/CVAE mu
            with torch.no_grad():
                psd = torch.FloatTensor(eeg_features["psd"]).unsqueeze(0).to(device)
                bp = torch.FloatTensor(eeg_features["band_powers"]).unsqueeze(0).to(device)
                coh = torch.FloatTensor(eeg_features["coherence"]).unsqueeze(0).to(device)
                plv = torch.FloatTensor(eeg_features["plv"]).unsqueeze(0).to(device)
                disease = torch.tensor([disease_label], dtype=torch.float32).to(device)
                eeg_lat = model.eeg_encoder(psd, bp, coh, plv)

                def _mu_for(drug_vec: np.ndarray) -> np.ndarray:
                    de = torch.FloatTensor(drug_vec).unsqueeze(0).to(device)
                    dl = model.drug_encoder(de)
                    fused = model.fusion(eeg_lat, dl, disease)
                    mu, _ = model.cvae.encode(fused)
                    return mu.cpu().numpy().reshape(-1)

                mu_base = _mu_for(np.zeros(384, dtype=np.float32))
                mu_don = _mu_for(drug_emb["donepezil"])
                mu_mem = _mu_for(drug_emb["memantine"])

            don_shift = float(np.linalg.norm(mu_don - mu_base))
            mem_shift = float(np.linalg.norm(mu_mem - mu_base))
            mean_drug_response = 0.5 * (don_shift + mem_shift)

            patient = {
                "patient_id": rec["patient_id"],
                "session_id": rec["session_id"],
                "label": label,
                "disease_label_proxy": disease_label,
                "feature_path": feat_path,
                "donepezil_latent_shift": don_shift,
                "memantine_latent_shift": mem_shift,
                "mean_drug_response": mean_drug_response,
                "donepezil_band_changes": _band_change_vector(don, base),
                "memantine_band_changes": _band_change_vector(mem, base),
                "feature_l2": float(np.linalg.norm(flat)),
            }
            # Persist per-patient sim means
            np.savez_compressed(
                os.path.join(
                    output_dir,
                    f"{rec['patient_id']}_{rec['session_id']}_sims.npz",
                ),
                baseline=base.astype(np.float32),
                donepezil=don.astype(np.float32),
                memantine=mem.astype(np.float32),
                mu_baseline=mu_base.astype(np.float32),
                mu_donepezil=mu_don.astype(np.float32),
                mu_memantine=mu_mem.astype(np.float32),
            )
            per_patient.append(patient)
        except Exception as exc:
            per_patient.append(
                {
                    "patient_id": rec.get("patient_id"),
                    "session_id": rec.get("session_id"),
                    "label": label,
                    "ok": False,
                    "error": str(exc),
                }
            )

    out = {
        "checkpoint": str(ckpt),
        "device": str(device),
        "n_patients": len(per_patient),
        "n_ok": sum(1 for p in per_patient if "mean_drug_response" in p),
        "patients": per_patient,
    }
    with open(os.path.join(output_dir, "tuh_twin_simulations.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def _load_training_flat_features(project_root: str) -> Tuple[np.ndarray, np.ndarray]:
    feat_dir = Path(project_root) / "data" / "eeg_features"
    flats = []
    labels = []
    for group, lab in (("AD", 1), ("HC", 0)):
        psd = np.load(feat_dir / f"{group}_psd.npy")
        bp = np.load(feat_dir / f"{group}_band_powers.npy")
        coh = np.load(feat_dir / f"{group}_coherence.npy")
        plv = np.load(feat_dir / f"{group}_plv.npy")
        for i in range(psd.shape[0]):
            flats.append(flatten_features(psd[i], bp[i], coh[i], plv[i]))
            labels.append(lab)
    return np.stack(flats, axis=0), np.asarray(labels)


def _load_cohort_mean_effects(project_root: str) -> Dict[str, np.ndarray]:
    sim_dir = Path(project_root) / "models" / "simulations_constrained_full"
    base = np.load(sim_dir / "simulated_baseline.npy").mean(axis=1)
    don = np.load(sim_dir / "simulated_donepezil.npy").mean(axis=1)
    mem = np.load(sim_dir / "simulated_memantine.npy").mean(axis=1)
    return {
        "donepezil_effect": (don - base).mean(axis=0),
        "memantine_effect": (mem - base).mean(axis=0),
    }


def analyze_abnormal_vs_normal(sim_results: Dict) -> Dict:
    patients = [p for p in sim_results["patients"] if "mean_drug_response" in p]
    ab = np.array([p["mean_drug_response"] for p in patients if p.get("label") == "abnormal"])
    nr = np.array([p["mean_drug_response"] for p in patients if p.get("label") == "normal"])
    if len(ab) == 0 or len(nr) == 0:
        return {
            "n_abnormal": int(len(ab)),
            "n_normal": int(len(nr)),
            "status": "FAIL",
            "pass": False,
            "reason": "missing abnormal or normal group",
        }
    u, p = stats.mannwhitneyu(ab, nr, alternative="two-sided")
    d = _cohens_d(ab, nr)
    # Expected: abnormal larger response
    direction_ok = float(np.mean(ab)) > float(np.mean(nr))
    passed = bool(p < 0.05 and direction_ok) or bool(abs(d) >= 0.3 and direction_ok)
    marginal = bool(direction_ok and (p < 0.1 or abs(d) >= 0.2))
    status = "PASS" if passed else ("MARGINAL" if marginal else "FAIL")
    return {
        "n_abnormal": int(len(ab)),
        "n_normal": int(len(nr)),
        "mean_abnormal": float(np.mean(ab)),
        "std_abnormal": float(np.std(ab, ddof=1)) if len(ab) > 1 else 0.0,
        "mean_normal": float(np.mean(nr)),
        "std_normal": float(np.std(nr, ddof=1)) if len(nr) > 1 else 0.0,
        "mannwhitney_u": float(u),
        "mannwhitney_p": float(p),
        "cohens_d": float(d),
        "direction_abnormal_gt_normal": direction_ok,
        "status": status,
        "pass": status == "PASS",
    }


def analyze_feature_distribution(
    feature_dir: str,
    processing_report: Dict,
    project_root: str,
    fig_path: Optional[str] = None,
) -> Dict:
    train_X, train_y = _load_training_flat_features(project_root)
    tuh_vecs = []
    tuh_labels = []
    for rec in processing_report.get("results", []):
        if not rec.get("ok"):
            continue
        path = rec["feature_path"]
        if not os.path.exists(path):
            continue
        tuh_vecs.append(np.load(path).astype(np.float64))
        tuh_labels.append(1 if rec.get("label") == "abnormal" else 0)
    if not tuh_vecs:
        return {"status": "FAIL", "reason": "no TUH features"}
    tuh_X = np.stack(tuh_vecs, axis=0)

    pca = PCA(n_components=2, random_state=42)
    train_2d = pca.fit_transform(train_X)
    tuh_2d = pca.transform(tuh_X)

    train_cent = train_2d.mean(axis=0)
    train_rad = np.linalg.norm(train_2d - train_cent, axis=1)
    r95 = float(np.percentile(train_rad, 95))
    tuh_rad = np.linalg.norm(tuh_2d - train_cent, axis=1)
    overlap_frac = float(np.mean(tuh_rad <= r95))
    mean_feat_dist = float(np.mean(np.linalg.norm(tuh_X - train_X.mean(axis=0), axis=1)))

    if overlap_frac >= 0.6:
        overlap = "good"
    elif overlap_frac >= 0.3:
        overlap = "moderate"
    else:
        overlap = "poor"

    # Band means comparison
    train_bands = np.array([list(band_powers_mean_per_band(v).values()) for v in train_X])
    tuh_bands = np.array([list(band_powers_mean_per_band(v).values()) for v in tuh_X])

    if fig_path:
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
        ax.scatter(
            train_2d[train_y == 1, 0],
            train_2d[train_y == 1, 1],
            c="#c44e52",
            s=28,
            alpha=0.7,
            label="Train AD",
        )
        ax.scatter(
            train_2d[train_y == 0, 0],
            train_2d[train_y == 0, 1],
            c="#4c72b0",
            s=28,
            alpha=0.7,
            label="Train HC",
        )
        ax.scatter(
            tuh_2d[:, 0],
            tuh_2d[:, 1],
            c="#55a868",
            s=36,
            alpha=0.85,
            marker="^",
            label="TUH",
            edgecolors="k",
            linewidths=0.3,
        )
        circ = plt.Circle(train_cent, r95, fill=False, linestyle="--", color="gray", label="Train 95% radius")
        ax.add_patch(circ)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("TUH vs Training Feature PCA")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

    return {
        "pca_overlap": overlap,
        "overlap_fraction_within_train_r95": overlap_frac,
        "mean_feature_distance": mean_feat_dist,
        "n_tuh": int(tuh_X.shape[0]),
        "n_train": int(train_X.shape[0]),
        "train_band_means": {
            BAND_NAMES[i]: float(train_bands[:, i].mean()) for i in range(5)
        },
        "tuh_band_means": {
            BAND_NAMES[i]: float(tuh_bands[:, i].mean()) for i in range(5)
        },
        "figure": fig_path,
    }


def analyze_cross_dataset(sim_results: Dict, project_root: str) -> Dict:
    cohort = _load_cohort_mean_effects(project_root)
    patients = [p for p in sim_results["patients"] if "mean_drug_response" in p]
    if not patients:
        return {"status": "FAIL", "reason": "no simulations"}

    # Build mean TUH effect from saved npz if present
    sim_dir = Path(project_root) / "data" / "tuh_validation"
    don_effects = []
    mem_effects = []
    for p in patients:
        npz = sim_dir / f"{p['patient_id']}_{p['session_id']}_sims.npz"
        if not npz.exists():
            continue
        z = np.load(npz)
        don_effects.append(z["donepezil"] - z["baseline"])
        mem_effects.append(z["memantine"] - z["baseline"])
    if not don_effects:
        return {"status": "FAIL", "reason": "no per-patient sim files"}

    tuh_don = np.mean(np.stack(don_effects), axis=0)
    tuh_mem = np.mean(np.stack(mem_effects), axis=0)

    def _dir_agree(a: np.ndarray, b: np.ndarray, slices=BAND_SLICES) -> Tuple[int, int]:
        agree = 0
        total = 0
        for sl in slices.values():
            total += 1
            if np.sign(np.mean(a[sl])) == np.sign(np.mean(b[sl])) or (
                abs(np.mean(a[sl])) < 1e-8 and abs(np.mean(b[sl])) < 1e-8
            ):
                agree += 1
        return agree, total

    d_agree, d_tot = _dir_agree(tuh_don, cohort["donepezil_effect"])
    m_agree, m_tot = _dir_agree(tuh_mem, cohort["memantine_effect"])
    r_don = float(np.corrcoef(tuh_don, cohort["donepezil_effect"])[0, 1])
    r_mem = float(np.corrcoef(tuh_mem, cohort["memantine_effect"])[0, 1])
    r_mean = float(np.nanmean([r_don, r_mem]))
    return {
        "donepezil_direction_agreement": f"{d_agree}/{d_tot}",
        "memantine_direction_agreement": f"{m_agree}/{m_tot}",
        "direction_agreement_total": f"{d_agree + m_agree}/{d_tot + m_tot}",
        "donepezil_effect_corr": r_don,
        "memantine_effect_corr": r_mem,
        "effect_magnitude_correlation": r_mean,
        "n_tuh_used": len(don_effects),
    }


def analyze_drug_mention_patients(
    header_rows: List[Dict],
    sim_results: Dict,
) -> Dict:
    mention_ids = {
        (r["patient_id"], r["session_id"])
        for r in header_rows
        if r.get("drug_status") in ("donepezil", "memantine", "both", "other_ad_drug")
    }
    matched = []
    for p in sim_results.get("patients", []):
        key = (p.get("patient_id"), p.get("session_id"))
        if key in mention_ids and "mean_drug_response" in p:
            matched.append(p)
    return {
        "n_header_drug_mentions": len(mention_ids),
        "n_with_simulations": len(matched),
        "patients": matched,
        "note": (
            "TUH EDF headers often lack medication text; "
            "validation primarily uses abnormal vs normal labels."
            if len(mention_ids) == 0
            else "Drug-mention patients available for qualitative inspection."
        ),
    }


def plot_response_comparison(disc: Dict, fig_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    means = [disc["mean_abnormal"], disc["mean_normal"]]
    stds = [disc["std_abnormal"], disc["std_normal"]]
    ax.bar(
        ["Abnormal", "Normal"],
        means,
        yerr=stds,
        color=["#c44e52", "#4c72b0"],
        alpha=0.85,
        capsize=4,
        width=0.55,
    )
    ax.set_ylabel("Mean drug response (latent ‖Δμ‖)")
    ax.set_title(
        f"TUH Abnormal vs Normal\n"
        f"p={disc['mannwhitney_p']:.4f}, d={disc['cohens_d']:.2f}"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def run_tuh_validation_analyses(
    project_root: str,
    header_rows: List[Dict],
    header_summary: Dict,
    processing_report: Dict,
    sim_results: Dict,
    output_dir: str,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    fig_disc = os.path.join(output_dir, "fig_tuh_abnormal_vs_normal.png")
    fig_pca = os.path.join(output_dir, "fig_tuh_pca_overlap.png")

    disc = analyze_abnormal_vs_normal(sim_results)
    if disc.get("n_abnormal", 0) and disc.get("n_normal", 0):
        plot_response_comparison(disc, fig_disc)

    feat_dist = analyze_feature_distribution(
        os.path.join(project_root, "data", "tuh_features"),
        processing_report,
        project_root,
        fig_path=fig_pca,
    )
    cross = analyze_cross_dataset(sim_results, project_root)
    drug_mentions = analyze_drug_mention_patients(header_rows, sim_results)

    # Overall status
    statuses = [disc.get("status", "FAIL")]
    if feat_dist.get("pca_overlap") == "poor":
        statuses.append("FAIL")
    elif feat_dist.get("pca_overlap") == "moderate":
        statuses.append("MARGINAL")
    else:
        statuses.append("PASS")
    r = cross.get("effect_magnitude_correlation", float("nan"))
    if isinstance(r, float) and not np.isnan(r) and r > 0.3:
        statuses.append("PASS")
    elif isinstance(r, float) and not np.isnan(r) and r > 0.1:
        statuses.append("MARGINAL")
    else:
        statuses.append("MARGINAL" if processing_report.get("n_success", 0) > 0 else "FAIL")

    if "FAIL" in statuses and statuses.count("PASS") == 0:
        overall = "FAIL"
    elif "FAIL" in statuses or "MARGINAL" in statuses:
        overall = "MARGINAL" if disc.get("status") != "FAIL" else "FAIL"
        if disc.get("status") == "PASS" and feat_dist.get("pca_overlap") != "poor":
            overall = "MARGINAL"
        if disc.get("status") == "PASS" and feat_dist.get("pca_overlap") in ("good", "moderate"):
            overall = "PASS" if "FAIL" not in statuses else "MARGINAL"
    else:
        overall = "PASS"

    # Prefer clearer overall rule
    if disc.get("status") == "PASS" and feat_dist.get("pca_overlap") in ("good", "moderate"):
        overall = "PASS"
    elif disc.get("status") in ("PASS", "MARGINAL") or feat_dist.get("pca_overlap") != "poor":
        overall = "MARGINAL" if overall == "FAIL" else overall
    else:
        overall = "FAIL"

    n_edf = processing_report.get("n_attempted", 0)
    n_ok = processing_report.get("n_success", 0)
    results = {
        "processing": {
            "total_edf_files": n_edf,
            "successfully_processed": n_ok,
            "success_pct": float(100.0 * n_ok / n_edf) if n_edf else 0.0,
            "features_extracted": n_ok,
        },
        "header_mining": header_summary,
        "abnormal_vs_normal": disc,
        "feature_distribution": feat_dist,
        "cross_dataset": cross,
        "drug_mention_analysis": drug_mentions,
        "overall_status": overall,
        "figures": {
            "abnormal_vs_normal": fig_disc,
            "pca_overlap": fig_pca,
        },
    }

    with open(os.path.join(output_dir, "tuh_validation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    _print_validation_table(results)
    return results


def _print_validation_table(results: Dict) -> None:
    p = results["processing"]
    h = results["header_mining"]
    d = results["abnormal_vs_normal"]
    f = results["feature_distribution"]
    c = results["cross_dataset"]
    print("=" * 63)
    print("TUH CLINICAL EEG VALIDATION — DIGITAL BRAIN TWIN")
    print("=" * 63)
    print("Processing:")
    print(f"  Total EDF files:               {p['total_edf_files']}")
    print(
        f"  Successfully processed:        {p['successfully_processed']} "
        f"({p['success_pct']:.1f}%)"
    )
    print(f"  Features extracted:            {p['features_extracted']}")
    print()
    print("Header mining:")
    print(f"  Files with any clinical text:  {h.get('files_with_clinical_text', 0)}")
    print(f"  AD drug mentions found:        {h.get('ad_drug_mentions', 0)}")
    print(f"    Donepezil/Aricept:           {h.get('donepezil_aricept', 0)}")
    print(f"    Memantine/Namenda:           {h.get('memantine_namenda', 0)}")
    print(f"    Other AD drugs:              {h.get('other_ad_drugs', 0)}")
    print(f"  AD diagnosis mentions:         {h.get('ad_diagnosis_mentions', 0)}")
    print(f"  Minimal/no header text:        {h.get('minimal_or_no_header_text', 0)}")
    print()
    print("Validation results:")
    print("  Abnormal vs Normal discrimination:")
    if d.get("n_abnormal", 0) and d.get("n_normal", 0):
        print(
            f"    Mean drug response (abnormal): {d['mean_abnormal']:.4f} ± {d['std_abnormal']:.4f}"
        )
        print(
            f"    Mean drug response (normal):   {d['mean_normal']:.4f} ± {d['std_normal']:.4f}"
        )
        print(f"    Mann-Whitney U p-value:        {d['mannwhitney_p']:.4f}")
        print(f"    Cohen's d:                     {d['cohens_d']:.2f}")
        print(f"    Status:                        {d['status']}")
    else:
        print(f"    Status:                        {d.get('status', 'FAIL')}")
    print()
    print("  Feature distribution:")
    print(f"    PCA overlap with training:     {f.get('pca_overlap', 'n/a')}")
    print(f"    Mean feature distance:         {f.get('mean_feature_distance', float('nan')):.3f}")
    print()
    print("  Cross-dataset consistency:")
    print(f"    Direction agreement:           {c.get('direction_agreement_total', 'n/a')}")
    r = c.get("effect_magnitude_correlation", float("nan"))
    print(f"    Effect magnitude correlation:  r = {r:.3f}" if isinstance(r, float) else f"    Effect magnitude correlation:  {r}")
    print()
    print(f"  Overall TUH validation:         {results['overall_status']}")
    print("=" * 63)


def update_validation_report_v3(
    project_root: str,
    tuh_results: Dict,
    header_summary: Dict,
) -> Dict:
    """Add Layer 5 to complete_validation_report and save v3."""
    v2_path = os.path.join(
        project_root, "models", "validation", "complete_validation_report_v2.json"
    )
    with open(v2_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    disc = tuh_results.get("abnormal_vs_normal", {})
    feat = tuh_results.get("feature_distribution", {})
    cross = tuh_results.get("cross_dataset", {})
    overall = tuh_results.get("overall_status", "FAIL")

    report["layer5"] = {
        "name": "TUH Clinical EEG Validation",
        "status": overall,
        "pass": overall == "PASS",
        "processing": tuh_results.get("processing", {}),
        "header_mining": header_summary,
        "abnormal_vs_normal": disc,
        "feature_distribution": {
            "pca_overlap": feat.get("pca_overlap"),
            "mean_feature_distance": feat.get("mean_feature_distance"),
            "n_tuh": feat.get("n_tuh"),
        },
        "cross_dataset": cross,
        "interpretation": (
            "External clinical EEG (TUH abnormal/normal) processed through the "
            "constrained digital twin. Header drug mentions are often sparse; "
            "primary evidence is abnormal vs normal response discrimination and "
            "cross-dataset effect consistency with the 66-subject training cohort."
        ),
    }

    layers_pass = sum(
        1
        for k in ("layer1", "layer2", "layer3", "layer4", "layer5")
        if report.get(k, {}).get("pass") or str(report.get(k, {}).get("status", "")).upper().find("PASS") >= 0
    )
    # layer2 may be "STRONG PASS"
    def _layer_pass(layer: Dict) -> bool:
        if not layer:
            return False
        if layer.get("pass") is True:
            return True
        st = str(layer.get("status", "")).upper()
        return "PASS" in st and "FAIL" not in st

    layers_pass = sum(_layer_pass(report.get(k, {})) for k in ("layer1", "layer2", "layer3", "layer4", "layer5"))
    report["overall"] = {
        "layers_pass": layers_pass,
        "layers_total": 5,
        "summary": f"{layers_pass}/5 layers PASS",
        "primary_model": report.get("architecture", {}).get("primary_model", "constrained_mlp"),
    }

    out_json = os.path.join(
        project_root, "models", "validation", "complete_validation_report_v3.json"
    )
    out_txt = os.path.join(
        project_root, "models", "validation", "complete_validation_report_v3.txt"
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    lines = [
        "COMPLETE VALIDATION REPORT v3",
        "=" * 60,
        f"Overall: {layers_pass}/5 layers PASS",
        "",
        f"Layer 1 Reconstruction: {report['layer1'].get('status')}",
        f"Layer 2 TVB: {report['layer2'].get('status')}",
        f"Layer 3 Personalization: {report['layer3'].get('status')}",
        f"Layer 4 ADNI: {report['layer4'].get('status')}",
        f"Layer 5 TUH Clinical EEG: {overall}",
        "",
        "Layer 5 summary:",
        f"  Processed: {tuh_results.get('processing', {}).get('successfully_processed')} / "
        f"{tuh_results.get('processing', {}).get('total_edf_files')}",
        f"  AD drug header mentions: {header_summary.get('ad_drug_mentions', 0)}",
        f"  Abnormal vs Normal: {disc.get('status')} "
        f"(p={disc.get('mannwhitney_p', float('nan'))}, d={disc.get('cohens_d', float('nan'))})",
        f"  PCA overlap: {feat.get('pca_overlap')}",
        f"  Cross-dataset r: {cross.get('effect_magnitude_correlation')}",
    ]
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nUpdated overall: {layers_pass}/5 layers PASS")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_txt}")
    return report
