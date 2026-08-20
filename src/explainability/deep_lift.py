"""
DeepLIFT / DeepLIFT-SHAP (Captum) — complementary gradient-free attributions.

Works with scalar ``forward_reconstruction`` (mean MSE). BatchNorm may reduce explainability;
results are still academically standard for ablation vs IG.

Complexity: O(forward) per attribution call — typically faster than IG for similar quality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.explainability.model_wrapper import DigitalTwinWrapper, DRUG_DIM, EEG_FLAT_SIZE
from src.explainability.integrated_gradients import _expand_labels_to_batch

logger = logging.getLogger(__name__)


class _DeepLiftMSERegressionModule(nn.Module):
    """
    Captum DeepLift ≥0.9 concatenates ``model_out[:, 0]`` and ``model_out[:, 1]``
    internally (binary-classification path). Regression with a single output column
    therefore crashes. We return ``(batch, 2)``: column 0 = per-sample MSE, column 1 =
    zeros (ignored). Use ``DeepLift.attribute(..., target=0)``.
    """

    def __init__(self, wrapper: DigitalTwinWrapper, disease_label: torch.Tensor):
        super().__init__()
        self.wrapper = wrapper
        self.register_buffer("disease_label", disease_label.clone())

    def forward(self, eeg: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        dl = _expand_labels_to_batch(self.disease_label, eeg.shape[0])
        recon = self.wrapper.forward(eeg, drug, dl, enable_grad=True)
        tgt = eeg.detach()
        mse_per_sample = ((recon - tgt) ** 2).mean(dim=1)
        dummy = torch.zeros_like(mse_per_sample)
        return torch.stack([mse_per_sample, dummy], dim=1)


def run_deeplift(
    wrapper: DigitalTwinWrapper,
    eeg_tensor: torch.Tensor,
    drug_tensor: torch.Tensor,
    *,
    disease_label: Optional[torch.Tensor] = None,
    baseline_eeg: Optional[torch.Tensor] = None,
    baseline_drug: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    DeepLIFT attributions averaged over batch -> (2185,) mean |attr_eeg|.

    Args:
        baseline_*: If None, uses zeros matching batch shape.

    Raises:
        ImportError: Captum missing.
        RuntimeError: DeepLIFT may fail on some BatchNorm edge cases — caller can catch.

    Complexity: O(forward * features) internally via Captum.
    """
    try:
        from captum.attr import DeepLift
    except ImportError as e:
        raise ImportError("pip install captum") from e

    device = eeg_tensor.device
    b = eeg_tensor.shape[0]
    if disease_label is None:
        disease_label = torch.zeros(b, device=device, dtype=eeg_tensor.dtype)
    if baseline_eeg is None:
        baseline_eeg = torch.zeros_like(eeg_tensor)
    if baseline_drug is None:
        baseline_drug = torch.zeros_like(drug_tensor)

    model = _DeepLiftMSERegressionModule(wrapper, disease_label).to(device)
    model.eval()
    dl = DeepLift(model)
    attr = dl.attribute(
        (eeg_tensor, drug_tensor),
        baselines=(baseline_eeg, baseline_drug),
        target=0,
    )
    ae = attr[0].detach().cpu().numpy()
    return np.mean(np.abs(ae), axis=0)


def run_deeplift_shap(
    wrapper: DigitalTwinWrapper,
    eeg_tensor: torch.Tensor,
    drug_tensor: torch.Tensor,
    background_samples: np.ndarray,
    *,
    disease_label: Optional[torch.Tensor] = None,
    drug_background: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    DeepLiftShap with a **distribution** of EEG baselines (rows of ``background_samples``).

    Args:
        background_samples: (n_bg, 2185) numpy array.

    Returns:
        Mean absolute EEG attribution vector (2185,).

    Complexity: O(n_bg * cost_deeplift) in worst implementations; Captum aggregates internally.
    """
    try:
        from captum.attr import DeepLiftShap
    except ImportError as e:
        raise ImportError("pip install captum") from e

    device = eeg_tensor.device
    b = eeg_tensor.shape[0]
    if disease_label is None:
        disease_label = torch.zeros(b, device=device, dtype=eeg_tensor.dtype)

    bg = torch.from_numpy(np.asarray(background_samples)).float().to(device)
    if drug_background is None:
        bd = torch.zeros(bg.shape[0], DRUG_DIM, device=device)
    else:
        bd = torch.from_numpy(np.asarray(drug_background)).float().to(device)

    model = _DeepLiftMSERegressionModule(wrapper, disease_label).to(device)
    model.eval()
    dls = DeepLiftShap(model)
    attr = dls.attribute(
        (eeg_tensor, drug_tensor),
        baselines=(bg, bd),
        target=0,
    )
    ae = attr[0].detach().cpu().numpy()
    return np.mean(np.abs(ae), axis=0)


def compare_attribution_methods(
    ig_attrs: np.ndarray,
    deeplift_attrs: np.ndarray,
    occlusion_map: np.ndarray,
    *,
    full_dim_shap_attrs: Optional[np.ndarray] = None,
    ig_normalized_block: Optional[Dict[str, float]] = None,
    block_shap_mean_abs_dict: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Tier 1 (primary): Spearman correlations among IG, DeepLIFT, occlusion — all length 2185.

    Full-dimensional KernelSHAP is excluded from Tier 1 (high-D + limited samples is
    ill-conditioned). Optionally pass ``full_dim_shap_attrs`` to populate
    ``legacy_four_method_matrix`` for diagnostics only.

    Tier 2 (optional): Spearman correlation between length-3 vectors of IG block weights
    vs block-level SHAP mean |values| when ``ig_normalized_block`` and
    ``block_shap_mean_abs_dict`` are provided.
    """
    from scipy.stats import spearmanr

    names_t1 = ["Integrated Gradients", "DeepLIFT", "Occlusion"]
    attrs_t1 = [np.asarray(ig_attrs), np.asarray(deeplift_attrs), np.asarray(occlusion_map)]
    n_t1 = len(attrs_t1)
    tier1_mat = np.eye(n_t1, dtype=np.float64)
    for i in range(n_t1):
        for j in range(i + 1, n_t1):
            r, _ = spearmanr(attrs_t1[i], attrs_t1[j])
            if np.isnan(r):
                r = 0.0
            tier1_mat[i, j] = tier1_mat[j, i] = float(r)

    off_t1 = tier1_mat[np.triu_indices(n_t1, k=1)]
    mean_t1 = float(np.mean(off_t1))
    if mean_t1 >= 0.7:
        label_t1 = "HIGH"
    elif mean_t1 >= 0.5:
        label_t1 = "MODERATE"
    else:
        label_t1 = "LOW"

    pairs_t1 = []
    for i in range(n_t1):
        for j in range(i + 1, n_t1):
            pairs_t1.append((names_t1[i], names_t1[j], float(tier1_mat[i, j])))

    tier2_r = None
    if ig_normalized_block is not None and block_shap_mean_abs_dict is not None:
        order = ("psd", "band_powers", "connectivity")
        try:
            v1 = np.array([float(ig_normalized_block[k]) for k in order])
            v2 = np.array([float(block_shap_mean_abs_dict[k]) for k in order])
            tier2_r, _ = spearmanr(v1, v2)
            if np.isnan(tier2_r):
                tier2_r = None
            else:
                tier2_r = float(tier2_r)
        except KeyError:
            tier2_r = None

    legacy_four: Optional[np.ndarray] = None
    if full_dim_shap_attrs is not None:
        methods4 = [
            np.asarray(ig_attrs),
            np.asarray(full_dim_shap_attrs),
            np.asarray(deeplift_attrs),
            np.asarray(occlusion_map),
        ]
        names4 = ["Integrated Gradients", "KernelSHAP", "DeepLIFT", "Occlusion"]
        legacy_four = np.ones((4, 4), dtype=np.float64)
        for i in range(4):
            for j in range(i + 1, 4):
                r, _ = spearmanr(methods4[i], methods4[j])
                if np.isnan(r):
                    r = 0.0
                legacy_four[i, j] = legacy_four[j, i] = float(r)

    summary: Dict[str, Any] = {
        "tier1_correlation_matrix": tier1_mat,
        "tier1_method_names": names_t1,
        "tier1_mean_agreement": mean_t1,
        "tier1_agreement_label": label_t1,
        "tier1_pairs": pairs_t1,
        "tier2_block_shap_vs_ig_spearman": tier2_r,
        "note_on_full_shap": (
            "Full-dimensional KernelSHAP (2185 features) excluded from Tier 1 agreement; "
            "high-D KernelSHAP is sample-limited. Block-level SHAP compared separately when available."
        ),
        "pairs": pairs_t1,
        "mean_off_diagonal": mean_t1,
        "interpretation": (
            "high_agreement"
            if mean_t1 > 0.7
            else ("moderate" if mean_t1 > 0.4 else "low_agreement")
        ),
        "legacy_four_method_matrix": legacy_four,
    }
    return tier1_mat, summary
