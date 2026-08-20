"""
Latent-space explanations for the CVAE path (mu only; deterministic during analysis).

Includes drug-induced latent shifts, single-dimension traversal, and lightweight MI summaries.

Complexity varies: O(batch * latent_dim * decode) for traversal.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

from src.explainability.model_wrapper import DigitalTwinWrapper, unflatten_eeg
logger = logging.getLogger(__name__)


def analyze_latent_dimensions(
    wrapper: DigitalTwinWrapper,
    eeg_array: np.ndarray,
    drug_tensor: torch.Tensor,
    labels: np.ndarray,
    *,
    device: Optional[torch.device] = None,
    top_k_dims: int = 10,
) -> Dict[str, Any]:
    """
    Rank CVAE latent dimensions by AD-HC separation (Welch t-test).

    Computes correlation (Pearson) of mu[:, d] vs block-averaged input features.

    Complexity: O(N * latent_dim).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)
    x = torch.from_numpy(eeg_array).float().to(device)
    dtensor = drug_tensor.float().to(device)
    lbl = torch.from_numpy(labels.astype(np.float32)).float().to(device)

    mu = wrapper.forward_latent(x, dtensor, lbl)
    mu_np = mu.detach().cpu().numpy()
    ad = lbl.cpu().numpy() >= 0.5
    hc = ~ad

    scores = []
    latent_dim = mu_np.shape[1]
    for d in range(latent_dim):
        if ad.sum() > 1 and hc.sum() > 1:
            t, p = scipy_stats.ttest_ind(mu_np[ad, d], mu_np[hc, d], equal_var=False)
        else:
            p = 1.0
            t = 0.0
        scores.append((d, abs(float(t)), float(p)))

    scores.sort(key=lambda z: -z[1])
    top = scores[:top_k_dims]

    # variance across subjects
    var_explained = mu_np.var(axis=0)

    report = {
        "top_discriminative_dims": [
            {
                "dim": int(d),
                "variance_across_subjects": float(var_explained[d]),
                "abs_t_stat_rank": float(abst),
                "p_value_approx": float(p),
            }
            for d, abst, p in top
        ],
        "latent_dim": latent_dim,
    }
    return report


def compute_drug_induced_latent_shift(
    wrapper: DigitalTwinWrapper,
    eeg_tensor: torch.Tensor,
    baseline_drug: torch.Tensor,
    donepezil_drug: torch.Tensor,
    memantine_drug: torch.Tensor,
    *,
    disease_label: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Compare CVAE latent means under three drug embeddings (same EEG).

    Complexity: O(3 * encode).
    """
    device = eeg_tensor.device
    b = eeg_tensor.shape[0]
    if disease_label is None:
        disease_label = torch.zeros(b, device=device, dtype=eeg_tensor.dtype)

    with torch.no_grad():
        mu_b = wrapper.forward_latent(eeg_tensor, baseline_drug, disease_label)
        mu_d = wrapper.forward_latent(eeg_tensor, donepezil_drug, disease_label)
        mu_m = wrapper.forward_latent(eeg_tensor, memantine_drug, disease_label)

    vb = mu_d - mu_b
    vm = mu_m - mu_b
    md = float(torch.norm(vb, dim=1).mean())
    mm = float(torch.norm(vm, dim=1).mean())
    cos = float(
        F.cosine_similarity(vb.mean(dim=0, keepdim=True), vm.mean(dim=0, keepdim=True)).item()
    )

    return {
        "baseline_to_donepezil_shift": vb.detach().cpu().numpy(),
        "baseline_to_memantine_shift": vm.detach().cpu().numpy(),
        "shift_magnitude_donepezil": md,
        "shift_magnitude_memantine": mm,
        "shift_direction_similarity": cos,
    }


def compute_latent_traversal(
    wrapper: DigitalTwinWrapper,
    eeg_tensor: torch.Tensor,
    drug_tensor: torch.Tensor,
    dim_idx: int,
    *,
    disease_label: Optional[torch.Tensor] = None,
    n_steps: int = 10,
    range_sigma: float = 2.0,
) -> Dict[str, Any]:
    """
    Vary CVAE latent coordinate ``dim_idx`` along ±range_sigma * sqrt(exp(logvar)).

    Decodes traversed points and extracts mean band powers from reconstructed flat vector.

    Complexity: O(n_steps * decode).
    """
    from src.explainability.feature_bounds import PSD_SIZE, BAND_SIZE, N_CHANNELS

    device = eeg_tensor.device
    batch = eeg_tensor.shape[0]
    if disease_label is None:
        disease_label = torch.zeros(batch, device=device, dtype=eeg_tensor.dtype)

    psd, bp, coh, plv = unflatten_eeg(eeg_tensor)
    drug_latent = wrapper.model.drug_encoder(drug_tensor)
    dl = disease_label.unsqueeze(1) if disease_label.dim() == 1 else disease_label
    eeg_latent = wrapper.model.eeg_encoder(psd, bp, coh, plv)
    fused = wrapper.model.fusion(eeg_latent, drug_latent, dl)
    mu, logvar = wrapper.model.cvae.encode(fused)
    sigma = torch.exp(0.5 * logvar)

    condition = torch.cat([drug_latent, dl], dim=1)
    alphas = torch.linspace(-range_sigma, range_sigma, steps=n_steps, device=device)

    bands_out = []
    for a in alphas:
        z = mu.clone()
        z[:, dim_idx] = mu[:, dim_idx] + a * sigma[:, dim_idx]
        recon = wrapper.model.cvae.decode(z, condition)
        bp_flat = recon[:, PSD_SIZE : PSD_SIZE + BAND_SIZE]
        bp_arr = bp_flat.view(batch, N_CHANNELS, 5).mean(dim=1).mean(dim=0).detach().cpu().numpy()
        bands_out.append(bp_arr[:4])

    arr = np.stack(bands_out, axis=0)
    return {
        "traversal_band_powers_first4": arr,
        "alphas_sigma_units": alphas.detach().cpu().numpy(),
        "dim_idx": int(dim_idx),
    }


def compute_information_bottleneck_analysis(
    wrapper: DigitalTwinWrapper,
    eeg_array: np.ndarray,
    drug_array: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    kNN mutual information estimates (sklearn) between summaries of inputs, mu, reconstruction MSE, label.

    Complexity: O(N^2) worst case for sklearn MI estimators — fine for N~66.

    Raises:
        ImportError: sklearn required.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError as e:
        raise ImportError("pip install scikit-learn") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)
    x = torch.from_numpy(eeg_array).float().to(device)
    d = torch.from_numpy(drug_array).float().to(device)
    lbl = torch.from_numpy(labels.astype(np.float64)).float().to(device)

    with torch.no_grad():
        mu = wrapper.forward_latent(x, d, lbl).cpu().numpy()
        recon = wrapper.forward(x, d, lbl, enable_grad=False)
        mse = ((recon - x) ** 2).mean(dim=1).cpu().numpy()

    # summarize inputs by block energy
    z_sum = np.abs(mu).mean(axis=1)
    inp_energy = np.mean(np.abs(eeg_array), axis=1)

    mi_z_label = mutual_info_regression(mu, lbl.cpu().numpy(), random_state=random_state).mean()
    mi_inp_z = mutual_info_regression(eeg_array[:, : min(128, eeg_array.shape[1])], z_sum, random_state=random_state)[0]

    out = {
        "mi_latent_dims_vs_label_mean": float(mi_z_label),
        "mi_input_slice_vs_latent_summary": float(mi_inp_z),
        "mse_mean": float(mse.mean()),
        "compression_note": "MI estimates are approximate (continuous/discrete mixing); interpret qualitatively.",
    }
    return out
