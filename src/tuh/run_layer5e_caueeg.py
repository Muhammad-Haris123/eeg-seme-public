"""
Layer 5e: CAUEEG Normal / MCI / Dementia external validation.

Addition only — does not replace TUH L5, OSF L5b, or P-ADIC L5d.
Primary contrast: Dementia vs Normal (AD-like vs HC-like).
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
from src.tuh.process_caueeg_eeg import default_raw_root, process_caueeg_cohort
from src.tuh.process_tuh_eeg import TARGET_N_EPOCHS_FOR_PSD, unflatten_features
from src.tuh.tuh_validation import _cohens_d


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_processed_features(out_dir: Path) -> Tuple[np.ndarray, List[Dict]]:
    report = json.loads((out_dir / "processing_report.json").read_text(encoding="utf-8"))
    flats, metas = [], []
    for rec in report["results"]:
        if not rec.get("ok"):
            continue
        flat = np.load(rec["feature_path"]).astype(np.float32)
        class_name = rec.get("class_name") or rec.get("group") or "Unknown"
        metas.append(
            {
                "serial": rec["serial"],
                "recording_id": rec["serial"],
                "class_name": class_name,
                "label": class_name.lower(),
                "group": class_name,
                "disease_label": int(rec.get("disease_label", 1 if class_name == "Dementia" else 0)),
                "age": rec.get("age"),
                "split": rec.get("split"),
                "has_ad_tag": bool(rec.get("has_ad_tag")),
                "n_epochs_used": int(rec.get("n_epochs_used") or 0),
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
    ext_bp = flats[:, 380:475]
    ext_mean = float(np.mean(np.abs(ext_bp)))
    ratio = ext_mean / max(train_mean, 1e-12)
    epochs = [m["n_epochs_used"] for m in metas]
    return {
        "train_abs_bandpower_mean": train_mean,
        "caueeg_abs_bandpower_mean": ext_mean,
        "bandpower_ratio_caueeg_over_train": ratio,
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

    for i, (flat, m) in enumerate(zip(flats, metas)):
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
                "class_name": m["class_name"],
                "group": m["group"],
                "disease_label": disease,
                "age": m.get("age"),
                "has_ad_tag": m.get("has_ad_tag"),
                "donepezil_latent_shift": don_shift,
                "memantine_latent_shift": mem_shift,
                "mean_drug_response": mean_resp,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  twin [{i+1}/{len(metas)}]")
    return {"patients": patients, "n_ok": len(patients), "checkpoint": str(ckpt)}


def _pair_stats(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> Dict:
    if len(a) < 2 or len(b) < 2:
        return {
            f"n_{name_a}": int(len(a)),
            f"n_{name_b}": int(len(b)),
            "status": "SKIP",
            "pass": False,
            "reason": "insufficient_n",
        }
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = _cohens_d(a, b)
    # Expect impaired group (a) > control-like (b) for drug-response magnitude
    direction_ok = float(np.mean(a)) > float(np.mean(b))
    passed = bool((p < 0.05 and direction_ok) or (abs(d) >= 0.3 and direction_ok))
    marginal = bool(direction_ok and (p < 0.1 or abs(d) >= 0.2))
    status = "PASS" if passed else ("MARGINAL" if marginal else "FAIL")
    return {
        f"n_{name_a}": int(len(a)),
        f"n_{name_b}": int(len(b)),
        f"mean_{name_a}": float(np.mean(a)),
        f"std_{name_a}": float(np.std(a, ddof=1)),
        f"mean_{name_b}": float(np.mean(b)),
        f"std_{name_b}": float(np.std(b, ddof=1)),
        "mannwhitney_u": float(u),
        "mannwhitney_p": float(p),
        "cohens_d": float(d),
        f"direction_{name_a}_gt_{name_b}": direction_ok,
        "status": status,
        "pass": status == "PASS",
    }


def discriminate(sim: Dict) -> Dict:
    by = {"normal": [], "mci": [], "dementia": []}
    for p in sim["patients"]:
        key = p["label"]
        if key in by:
            by[key].append(p["mean_drug_response"])
    for k in by:
        by[k] = np.asarray(by[k], dtype=np.float64)

    dem_vs_norm = _pair_stats(by["dementia"], by["normal"], "dementia", "normal")
    dem_vs_mci = _pair_stats(by["dementia"], by["mci"], "dementia", "mci")
    mci_vs_norm = _pair_stats(by["mci"], by["normal"], "mci", "normal")

    # AD-tagged dementia subset vs normal
    ad_tag = np.asarray(
        [
            p["mean_drug_response"]
            for p in sim["patients"]
            if p["label"] == "dementia" and p.get("has_ad_tag")
        ],
        dtype=np.float64,
    )
    ad_vs_norm = _pair_stats(ad_tag, by["normal"], "dementia_ad_tag", "normal")

    kruskal = None
    if min(len(by["normal"]), len(by["mci"]), len(by["dementia"])) >= 2:
        h, kp = stats.kruskal(by["normal"], by["mci"], by["dementia"])
        kruskal = {"H": float(h), "p": float(kp)}

    primary = dem_vs_norm
    return {
        "primary_dementia_vs_normal": dem_vs_norm,
        "dementia_vs_mci": dem_vs_mci,
        "mci_vs_normal": mci_vs_norm,
        "ad_tagged_dementia_vs_normal": ad_vs_norm,
        "kruskal_three_group": kruskal,
        "status": primary["status"],
        "pass": primary.get("pass", False),
        "caveats": [
            "Dementia labels are clinical spectrum (not pure biomarker AD)",
            "Twin disease_label: Dementia=1, Normal/MCI=0",
            "No drug-response outcomes in CAUEEG — latent shift is model-side only",
            "Hospital AVG-referenced EEG; domain shift vs training cohort expected",
        ],
    }


def pca_overlap(flats: np.ndarray, metas: List[Dict], root: Path, fig_path: Path) -> Dict:
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

    colors = {"normal": "#4c72b0", "mci": "#dd8452", "dementia": "#c44e52"}
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    ax.scatter(t2[train_y == 1, 0], t2[train_y == 1, 1], c="#c44e52", s=22, alpha=0.55, label="Train AD")
    ax.scatter(t2[train_y == 0, 0], t2[train_y == 0, 1], c="#4c72b0", s=22, alpha=0.55, label="Train HC")
    for lab, col in colors.items():
        mask = np.array([m["label"] == lab for m in metas])
        if not mask.any():
            continue
        ax.scatter(
            x2[mask, 0],
            x2[mask, 1],
            c=col,
            s=18,
            marker="^",
            alpha=0.7,
            edgecolors="k",
            linewidths=0.2,
            label=f"CAUEEG {lab}",
        )
    ax.add_patch(plt.Circle(c, r95, fill=False, ls="--", color="gray"))
    ax.set_title("Layer 5e: CAUEEG vs Training PCA")
    ax.legend(frameon=False, fontsize=7)
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
        "figure": str(fig_path),
    }


def annotate_v3(root: Path, layer5e: Dict) -> None:
    v3_path = root / "models" / "validation" / "complete_validation_report_v3.json"
    if not v3_path.exists():
        return
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    prior_l5 = json.dumps(v3.get("layer5"), sort_keys=True)
    prior_l5b = json.dumps(v3.get("layer5b_ad_labeled_external"), sort_keys=True)
    prior_l5d = json.dumps(v3.get("layer5d_padic_external"), sort_keys=True)
    v3["layer5e_caueeg_external"] = layer5e
    v3_path.write_text(json.dumps(v3, indent=2), encoding="utf-8")
    v3_after = json.loads(v3_path.read_text(encoding="utf-8"))
    assert json.dumps(v3_after.get("layer5"), sort_keys=True) == prior_l5
    assert json.dumps(v3_after.get("layer5b_ad_labeled_external"), sort_keys=True) == prior_l5b
    assert json.dumps(v3_after.get("layer5d_padic_external"), sort_keys=True) == prior_l5d


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = _root()
    raw_root = default_raw_root()
    data_dir = root / "data" / "caueeg_external"
    val_dir = data_dir / "validation"
    fig_dir = root / "models" / "validation" / "figures"
    val_dir.mkdir(parents=True, exist_ok=True)

    report_path = data_dir / "processing_report.json"
    need_extract = (
        args.force_extract
        or (not report_path.exists())
        or (
            report_path.exists()
            and json.loads(report_path.read_text(encoding="utf-8")).get("n_ok", 0) < 100
        )
    )
    if need_extract or args.limit:
        print("=== extracting CAUEEG features ===")
        process_caueeg_cohort(raw_root, data_dir, resume=True, limit=args.limit)
    else:
        print(f"[skip extract] existing {report_path}")

    if args.extract_only:
        print("[extract-only] stopping before twin")
        return 0

    flats, metas = load_processed_features(data_dir)
    print(f"Loaded features: {flats.shape[0]} x {flats.shape[1]}D")
    from collections import Counter

    print("classes", Counter(m["class_name"] for m in metas))

    scale = scale_check(flats, metas, root)
    print("=== scale check ===")
    print(json.dumps(scale, indent=2))

    print("=== twin ===")
    sim = run_twin(flats, metas, root, val_dir)
    disc = discriminate(sim)
    print("=== discrimination ===")
    print(json.dumps(disc, indent=2))

    print("=== PCA ===")
    pca = pca_overlap(flats, metas, root, fig_dir / "layer5e_caueeg_pca.png")
    print(json.dumps(pca, indent=2))

    layer5e = {
        "dataset": "CAUEEG (Kim et al., NeuroImage 2023)",
        "doi": "10.1016/j.neuroimage.2023.120054",
        "split_file": "dementia-no-overlap.json",
        "status": disc["status"],
        "discrimination": disc,
        "scale_check": scale,
        "pca": pca,
        "n_features": int(flats.shape[0]),
        "n_by_class": dict(Counter(m["class_name"] for m in metas)),
        "addition_only": True,
    }
    out_json = root / "models" / "validation" / "layer5e_caueeg_results.json"
    out_json.write_text(json.dumps(layer5e, indent=2), encoding="utf-8")
    annotate_v3(root, layer5e)

    md = root / "models" / "validation" / "layer5e_caueeg_external_diagnosis.md"
    md.write_text(
        "\n".join(
            [
                "# Layer 5e — CAUEEG external validation",
                "",
                f"**Status (primary Dementia vs Normal):** `{disc['status']}`",
                "",
                f"- N features: {flats.shape[0]}",
                f"- Classes: {dict(Counter(m['class_name'] for m in metas))}",
                f"- Band-power ratio (CAUEEG/train): {scale['bandpower_ratio_caueeg_over_train']:.3f}",
                f"- PCA overlap: {pca['pca_overlap']} ({pca['overlap_fraction_within_train_r95']:.2f})",
                "",
                "## Primary (Dementia vs Normal)",
                "```json",
                json.dumps(disc["primary_dementia_vs_normal"], indent=2),
                "```",
                "",
                "## Caveats",
                *[f"- {c}" for c in disc["caveats"]],
                "",
                "Cite: Kim et al., NeuroImage 2023. doi:10.1016/j.neuroimage.2023.120054",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[wrote] {out_json}")
    print(f"[wrote] {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
