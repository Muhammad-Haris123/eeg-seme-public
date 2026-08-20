"""Shared direction / magnitude helpers for constrained vs unconstrained batteries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

from src.models.data_loader import load_drug_embeddings
from src.models.inference import load_trained_model, simulate_post_drug_eeg
from src.tuh.process_tuh_eeg import unflatten_features
from src.tuh.tuh_validation import BAND_SLICES, _cohens_d
from src.validation.stats_ci import fisher_z_ci

FEAT_DIM = 2185


def load_constrained_training_effects(root: Path) -> Dict[str, np.ndarray]:
    cohort_dir = root / "models" / "simulations_constrained_full"
    base = np.load(cohort_dir / "simulated_baseline.npy").mean(1)
    don = np.load(cohort_dir / "simulated_donepezil.npy").mean(1)
    mem = np.load(cohort_dir / "simulated_memantine.npy").mean(1)
    return {
        "donepezil_effect": (don - base).mean(0),
        "memantine_effect": (mem - base).mean(0),
        "source": str(cohort_dir),
    }


def direction_agree(a: np.ndarray, b: np.ndarray) -> Tuple[int, int]:
    ok = tot = 0
    for sl in BAND_SLICES.values():
        tot += 1
        ma, mb = float(np.mean(a[sl])), float(np.mean(b[sl]))
        if np.sign(ma) == np.sign(mb) or (abs(ma) < 1e-8 and abs(mb) < 1e-8):
            ok += 1
    return ok, tot


def score_effects_vs_signature(
    don_effects: List[np.ndarray],
    mem_effects: List[np.ndarray],
    signature: Dict[str, np.ndarray],
) -> Dict:
    if not don_effects:
        return {"status": "FAIL", "reason": "no effects"}
    ext_don = np.mean(np.stack(don_effects), 0)
    ext_mem = np.mean(np.stack(mem_effects), 0)
    d_a, d_t = direction_agree(ext_don, signature["donepezil_effect"])
    m_a, m_t = direction_agree(ext_mem, signature["memantine_effect"])
    r_d = float(np.corrcoef(ext_don, signature["donepezil_effect"])[0, 1])
    r_m = float(np.corrcoef(ext_mem, signature["memantine_effect"])[0, 1])
    r_mean = float(np.nanmean([r_d, r_m]))
    n_feat = int(ext_don.shape[0])
    return {
        "direction_agreement_total": f"{d_a + m_a}/{d_t + m_t}",
        "donepezil_direction_agreement": f"{d_a}/{d_t}",
        "memantine_direction_agreement": f"{m_a}/{m_t}",
        "donepezil_effect_corr": r_d,
        "memantine_effect_corr": r_m,
        "effect_magnitude_correlation": r_mean,
        "donepezil_effect_corr_ci": fisher_z_ci(r_d, n_feat),
        "memantine_effect_corr_ci": fisher_z_ci(r_m, n_feat),
        "effect_magnitude_correlation_ci": fisher_z_ci(r_mean, n_feat),
        "n_used": len(don_effects),
        "signature_source": signature.get("source"),
    }


def mw_groups(a: np.ndarray, b: np.ndarray) -> Dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"status": "SKIP", "n_a": len(a), "n_b": len(b)}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = _cohens_d(a, b)
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": float(np.std(a, ddof=1)),
        "std_b": float(np.std(b, ddof=1)),
        "mannwhitney_u": float(u),
        "mannwhitney_p": float(p),
        "cohens_d": float(d),
    }


@torch.no_grad()
def simulate_flat_cohort(
    flats: np.ndarray,
    disease_labels: Iterable[int],
    subject_ids: Iterable[str],
    checkpoint: Path,
    out_dir: Path,
    num_samples: int = 5,
    seed: int = 42,
    drug_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """
    Run twin sims for a cohort of flattened features.
    Saves `{sid}_sims.npz` and returns patient rows with mean_drug_response.
    drug_embeddings: optional override matching the checkpoint's conditioner
    (e.g. padded one-hot for Phase C ablation). Default: ChemBERTa store.
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(checkpoint, device)
    drug_emb = drug_embeddings if drug_embeddings is not None else load_drug_embeddings()
    out_dir.mkdir(parents=True, exist_ok=True)
    patients = []
    don_effects, mem_effects = [], []

    for i, (flat, disease, sid) in enumerate(zip(flats, disease_labels, subject_ids)):
        eeg = unflatten_features(np.asarray(flat, dtype=np.float32))
        disease = int(disease)
        sims = simulate_post_drug_eeg(
            model, eeg, drug_emb, disease_label=disease, device=device, num_samples=num_samples
        )
        base = sims["baseline"].mean(0)
        don = sims["donepezil"].mean(0)
        mem = sims["memantine"].mean(0)

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

        np.savez_compressed(
            out_dir / f"{sid}_sims.npz",
            baseline=base.astype(np.float32),
            donepezil=don.astype(np.float32),
            memantine=mem.astype(np.float32),
            mu_baseline=mu_b.astype(np.float32),
            mu_donepezil=mu_d.astype(np.float32),
            mu_memantine=mu_m.astype(np.float32),
        )
        don_effects.append(don - base)
        mem_effects.append(mem - base)
        patients.append(
            {
                "subject_id": str(sid),
                "disease_label": disease,
                "donepezil_latent_shift": don_shift,
                "memantine_latent_shift": mem_shift,
                "mean_drug_response": mean_resp,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  twin [{i+1}/{len(flats)}]")

    return {
        "patients": patients,
        "n_ok": len(patients),
        "checkpoint": str(checkpoint),
        "don_effects": don_effects,
        "mem_effects": mem_effects,
    }


@torch.no_grad()
def extract_baseline_mu(
    flats: np.ndarray,
    checkpoint: Path,
    disease_label: float = 0.0,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(checkpoint, device)
    model.eval()
    zero_drug = torch.zeros(1, 384, dtype=torch.float32, device=device)
    disease_t = torch.tensor([disease_label], dtype=torch.float32, device=device)
    drug_lat = model.drug_encoder(zero_drug)
    mus = []
    for i, flat in enumerate(flats):
        eeg = unflatten_features(np.asarray(flat, dtype=np.float32))
        psd = torch.FloatTensor(eeg["psd"]).unsqueeze(0).to(device)
        bp = torch.FloatTensor(eeg["band_powers"]).unsqueeze(0).to(device)
        coh = torch.FloatTensor(eeg["coherence"]).unsqueeze(0).to(device)
        plv = torch.FloatTensor(eeg["plv"]).unsqueeze(0).to(device)
        eeg_lat = model.eeg_encoder(psd, bp, coh, plv)
        fused = model.fusion(eeg_lat, drug_lat, disease_t)
        mu, _ = model.cvae.encode(fused)
        mus.append(mu.cpu().numpy().reshape(-1))
        if (i + 1) % 100 == 0:
            print(f"  encode [{i+1}/{len(flats)}]")
    return np.stack(mus, axis=0).astype(np.float32)
