"""
Occlusion-based sensitivity for 1-D EEG feature vectors (MLP / CVAE).

Treats occlusion as the principled CAM analog when spatial convolutions are absent.

Complexity:
    Block occlusion: O(n_blocks * forward).
    Sliding window: O(n_windows * forward) — use ``stride`` large when prototyping.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from src.explainability.model_wrapper import DigitalTwinWrapper, EEG_FLAT_SIZE, N_CHANNELS, PSD_DIM
from src.explainability.feature_bounds import default_feature_blocks, PSD_SIZE, BAND_SIZE

logger = logging.getLogger(__name__)

CHANNEL_NAMES: Tuple[str, ...] = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
)


def _mse_vec(wrapper: DigitalTwinWrapper, eeg: torch.Tensor, drug: torch.Tensor, dl: torch.Tensor) -> torch.Tensor:
    """Per-row MSE (batch,) — differentiable in eeg."""
    recon = wrapper.forward(eeg, drug, dl, enable_grad=False)
    return ((recon - eeg.detach()) ** 2).mean(dim=1)


def _kl_batch(wrapper: DigitalTwinWrapper, eeg: torch.Tensor, drug: torch.Tensor, dl: torch.Tensor) -> torch.Tensor:
    """Batch mean KL(q(z|x) || N(0,I)) under deterministic encode statistics."""
    from src.explainability.model_wrapper import unflatten_eeg

    psd, bp, coh, plv = unflatten_eeg(eeg)
    with torch.no_grad():
        eeg_latent = wrapper.model.eeg_encoder(psd, bp, coh, plv)
        drug_latent = wrapper.model.drug_encoder(drug)
        dlv = dl.unsqueeze(1) if dl.dim() == 1 else dl
        fused = wrapper.model.fusion(eeg_latent, drug_latent, dlv)
        mu, logvar = wrapper.model.cvae.encode(fused)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl


def _latent_mu(wrapper: DigitalTwinWrapper, eeg: torch.Tensor, drug: torch.Tensor, dl: torch.Tensor) -> torch.Tensor:
    return wrapper.forward_latent(eeg, drug, dl, enable_grad=False)


def run_block_occlusion(
    wrapper: DigitalTwinWrapper,
    eeg_tensor: torch.Tensor,
    drug_tensor: torch.Tensor,
    feature_blocks: Dict[str, Tuple[int, int]],
    *,
    disease_label: Optional[torch.Tensor] = None,
    replace_value: float = 0.0,
) -> Dict[str, Any]:
    """
    Zero-out each contiguous block; measure Δ MSE, Δ KL, latent L2 shift vs full input.

    Args:
        feature_blocks: ``{'psd': (0, 380), ...}`` inclusive-exclusive indices.

    Returns:
        ``per_block`` metrics and ``normalized_sensitivity`` over reconstruction Δ.

    Complexity: O(n_blocks * forward).
    """
    device = eeg_tensor.device
    batch = eeg_tensor.shape[0]
    if disease_label is None:
        disease_label = torch.zeros(batch, device=device, dtype=eeg_tensor.dtype)

    with torch.no_grad():
        base_mse = _mse_vec(wrapper, eeg_tensor, drug_tensor, disease_label).mean().item()
        mu0 = _latent_mu(wrapper, eeg_tensor, drug_tensor, disease_label)
        base_kl = _kl_batch(wrapper, eeg_tensor, drug_tensor, disease_label).mean().item()

    per_block = {}
    recon_deltas = []

    for name, (a, b) in feature_blocks.items():
        pert = eeg_tensor.clone()
        pert[:, a:b] = replace_value
        with torch.no_grad():
            mse_occ = _mse_vec(wrapper, pert, drug_tensor, disease_label).mean().item()
            kl_occ = _kl_batch(wrapper, pert, drug_tensor, disease_label).mean().item()
            mu_occ = _latent_mu(wrapper, pert, drug_tensor, disease_label)
        recon_deltas.append(mse_occ - base_mse)
        latent_shift = (mu_occ - mu0).pow(2).sum(dim=1).sqrt().mean().item()
        per_block[name] = {
            "reconstruction_sensitivity": float(mse_occ - base_mse),
            "latent_sensitivity": float(latent_shift),
            "kl_sensitivity": float(kl_occ - base_kl),
        }

    s = np.array(recon_deltas, dtype=np.float64)
    ssum = np.sum(np.maximum(s, 0)) + 1e-12
    norm = np.maximum(s, 0) / ssum
    for i, name in enumerate(feature_blocks.keys()):
        per_block[name]["normalized_sensitivity"] = float(norm[i])

    return {"per_block": per_block, "baseline_mse": base_mse, "baseline_kl": base_kl}


def run_sliding_window_occlusion(
    wrapper: DigitalTwinWrapper,
    eeg_tensor: torch.Tensor,
    drug_tensor: torch.Tensor,
    *,
    disease_label: Optional[torch.Tensor] = None,
    window_size: int = 50,
    stride: int = 25,
    sigma_scale: float = 4.0,
) -> np.ndarray:
    """
    Accumulate Δ MSE when a window is replaced with zeros at each position; smooth with Gaussian.

    Returns:
        ``sensitivity_map`` length 2185 (positive values).

    Complexity: O(num_windows * batch * forward); large for full sweep.
    """
    device = eeg_tensor.device
    batch = eeg_tensor.shape[0]
    if disease_label is None:
        disease_label = torch.zeros(batch, device=device, dtype=eeg_tensor.dtype)

    with torch.no_grad():
        base_mse = _mse_vec(wrapper, eeg_tensor, drug_tensor, disease_label).mean().item()

    counts = np.zeros(EEG_FLAT_SIZE, dtype=np.float64)
    sens = np.zeros(EEG_FLAT_SIZE, dtype=np.float64)

    for start in range(0, EEG_FLAT_SIZE - window_size + 1, stride):
        end = start + window_size
        pert = eeg_tensor.clone()
        pert[:, start:end] = 0.0
        with torch.no_grad():
            mse_occ = _mse_vec(wrapper, pert, drug_tensor, disease_label).mean().item()
        delta = max(0.0, mse_occ - base_mse)
        sens[start:end] += delta
        counts[start:end] += 1.0

    counts = np.maximum(counts, 1.0)
    sens = sens / counts
    sigma = max(1.0, window_size / sigma_scale)
    sens = gaussian_filter1d(sens, sigma=sigma, mode="nearest")
    return sens.astype(np.float64)


def run_occlusion_per_condition(
    wrapper: DigitalTwinWrapper,
    eeg_tensor: torch.Tensor,
    baseline_drug: torch.Tensor,
    donepezil_drug: torch.Tensor,
    memantine_drug: torch.Tensor,
    feature_blocks: Dict[str, Tuple[int, int]],
    *,
    disease_label: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Block occlusion under each drug embedding; ``sensitivity_shift`` vs baseline drug.

    Complexity: 3 × ``run_block_occlusion``.
    """
    out = {}
    for name, dv in [
        ("baseline", baseline_drug),
        ("donepezil", donepezil_drug),
        ("memantine", memantine_drug),
    ]:
        out[name] = run_block_occlusion(
            wrapper, eeg_tensor, dv, feature_blocks, disease_label=disease_label
        )
    base = out["baseline"]["per_block"]
    shifts = {}
    for blk in feature_blocks:
        shifts[blk] = {
            "donepezil_minus_baseline": out["donepezil"]["per_block"][blk]["reconstruction_sensitivity"]
            - base[blk]["reconstruction_sensitivity"],
            "memantine_minus_baseline": out["memantine"]["per_block"][blk]["reconstruction_sensitivity"]
            - base[blk]["reconstruction_sensitivity"],
        }
    return {"conditions": out, "sensitivity_shift": shifts}


def run_channel_contribution_analysis(
    wrapper: DigitalTwinWrapper,
    eeg_tensor: torch.Tensor,
    drug_tensor: torch.Tensor,
    *,
    disease_label: Optional[torch.Tensor] = None,
    n_channels: int = N_CHANNELS,
) -> Dict[str, float]:
    """
    Occlude PSD + band-power segments per spatial channel (connectivity omitted here).

    Assigns approximate importance per 10–20 channel label.

    Complexity: O(n_channels * forward).

    Raises:
        ValueError: if ``n_channels`` inconsistent with layout.
    """
    if n_channels != N_CHANNELS:
        raise ValueError("Channel layout expects 19 channels for this pipeline.")

    device = eeg_tensor.device
    batch = eeg_tensor.shape[0]
    if disease_label is None:
        disease_label = torch.zeros(batch, device=device, dtype=eeg_tensor.dtype)

    with torch.no_grad():
        base_mse = _mse_vec(wrapper, eeg_tensor, drug_tensor, disease_label).mean().item()

    importance: Dict[str, float] = {}
    for c in range(n_channels):
        pert = eeg_tensor.clone()
        psd_a, psd_b = c * PSD_DIM, (c + 1) * PSD_DIM
        band_a = PSD_SIZE + c * 5
        band_b = band_a + 5
        pert[:, psd_a:psd_b] = 0.0
        pert[:, band_a:band_b] = 0.0
        with torch.no_grad():
            mse_occ = _mse_vec(wrapper, pert, drug_tensor, disease_label).mean().item()
        importance[CHANNEL_NAMES[c]] = float(max(0.0, mse_occ - base_mse))

    return importance
