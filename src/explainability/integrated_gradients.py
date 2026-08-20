"""
Integrated Gradients (Sundararajan et al., 2017) for the Digital Twin XAI stack.

Uses Captum on ``DigitalTwinWrapper.forward_reconstruction`` (scalar MSE) with
deterministic CVAE (z=μ). Includes multi-baseline ensembling, per-drug and
per-subject runs, and block-level summaries.

Typical complexity: O(n_steps * forward) per baseline; triple-baseline ~3x;
per-drug 3x; per-subject Nx.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.explainability.model_wrapper import DigitalTwinWrapper, DRUG_DIM, EEG_FLAT_SIZE
from src.explainability.feature_bounds import default_feature_blocks, load_feature_blocks_from_metadata

logger = logging.getLogger(__name__)

IG_CONVERGENCE_WARN = 0.05


def compute_block_significance_test(
    ig_attrs: np.ndarray,
    blocks: Dict[str, Tuple[int, int]],
    *,
    n_permutations: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Permutation null: shuffle absolute attribution magnitudes across the 2185 features;
    for each shuffle, recompute mean |attr| within each fixed index block.

    p-value per block = fraction of null draws with block mean >= observed block mean.
    """
    x = np.abs(np.asarray(ig_attrs, dtype=np.float64).reshape(-1))
    if x.size != EEG_FLAT_SIZE:
        raise ValueError(f"ig_attrs must have length {EEG_FLAT_SIZE}")

    observed: Dict[str, float] = {}
    for name, (s, e) in blocks.items():
        observed[name] = float(np.mean(x[s:e]))

    rng = np.random.default_rng(random_state)
    null_ge: Dict[str, List[bool]] = {name: [] for name in blocks}

    for _ in range(int(n_permutations)):
        shuffled = rng.permutation(x).copy()
        for name, (s, e) in blocks.items():
            null_mean = float(np.mean(shuffled[s:e]))
            null_ge[name].append(null_mean >= observed[name])

    p_values = {name: float(np.mean(null_ge[name])) for name in blocks}

    return {
        "observed_block_mean_abs": observed,
        "p_values": p_values,
        "n_permutations": int(n_permutations),
        "blocks_used": {k: list(v) for k, v in blocks.items()},
    }


def find_converged_ig_steps(
    wrapper: DigitalTwinWrapper,
    eeg_tensor: torch.Tensor,
    drug_tensor: torch.Tensor,
    disease_label: torch.Tensor,
    baseline_eeg: torch.Tensor,
    baseline_drug: torch.Tensor,
    *,
    step_candidates: Optional[List[int]] = None,
    delta_threshold: float = 0.05,
    max_steps: int = 500,
) -> Tuple[int, float, List[Dict[str, Any]], float]:
    """
    Increase IG steps until mean relative change in mean-|attr| EEG vector falls below
    ``delta_threshold``. Uses Captum's completeness-based delta as a secondary metric.

    Returns:
        (n_steps_used, final_step_delta, convergence_history, last_captum_delta_mean)
    """
    IntegratedGradients = _require_captum()
    if step_candidates is None:
        step_candidates = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    step_candidates = sorted({s for s in step_candidates if s <= max_steps and s >= 8})
    if not step_candidates:
        step_candidates = [min(max(8, max_steps), 500)]

    convergence_history: List[Dict[str, Any]] = []
    prev_vec: Optional[np.ndarray] = None

    wrapper.eval()
    n_steps_used = int(step_candidates[-1])
    final_step_delta = 999.0
    last_captum = float("nan")

    for n_steps in step_candidates:
        eeg_in = eeg_tensor.clone().detach().requires_grad_(True)
        drug_in = drug_tensor.clone().detach().requires_grad_(True)

        def _forward(e: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
            dl = _expand_labels_to_batch(disease_label, e.shape[0])
            out = wrapper.forward_reconstruction(e, d, dl, enable_grad=True)
            if out.dim() == 0:
                out = out.unsqueeze(0)
            return out

        ig = IntegratedGradients(_forward)
        raw = ig.attribute(
            (eeg_in, drug_in),
            baselines=(baseline_eeg, baseline_drug),
            n_steps=max(8, int(n_steps)),
            return_convergence_delta=True,
        )
        attrs, delta_t = raw
        ae = attrs[0].detach().cpu().numpy()
        vec = np.abs(ae).mean(axis=0)
        # Convergence delta can be a CUDA tensor; must not call np.asarray on it directly.
        captum_m = float(delta_t.detach().cpu().float().mean().item())

        step_d = float("nan")
        if prev_vec is not None:
            step_d = float(np.abs(vec - prev_vec).mean() / (np.abs(prev_vec).mean() + 1e-12))

        convergence_history.append(
            {
                "n_steps": int(n_steps),
                "step_delta": step_d,
                "captum_delta": captum_m,
            }
        )
        logger.info(
            "IG n_steps=%s step_delta=%.4f captum_delta=%.4f",
            n_steps,
            step_d if prev_vec is not None else float("nan"),
            captum_m,
        )

        if prev_vec is not None and not np.isnan(step_d) and step_d < delta_threshold:
            final_step_delta = step_d
            n_steps_used = int(n_steps)
            last_captum = captum_m
            return n_steps_used, final_step_delta, convergence_history, last_captum

        prev_vec = vec.copy()
        n_steps_used = int(n_steps)
        last_captum = captum_m
        final_step_delta = step_d if not np.isnan(step_d) else 999.0

    logger.warning(
        "IG did not reach step_delta < %s within step list (max %s). Using last run.",
        delta_threshold,
        n_steps_used,
    )
    return n_steps_used, final_step_delta, convergence_history, last_captum


def _expand_labels_to_batch(disease_label: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Captum's IntegratedGradients can temporarily use an expanded batch axis when
    attributing multiple inputs; the closure must mirror that batch size.

    Repeat / tile ``disease_label`` until its dim-0 length equals ``batch_size``.
    """
    dl = disease_label.reshape(-1)
    n = int(dl.numel())
    dtype = dl.dtype
    device = dl.device
    if n == batch_size:
        return dl
    if batch_size % n == 0:
        return dl.repeat(batch_size // n).to(device=device, dtype=dtype)
    reps = (batch_size + n - 1) // n
    return dl.repeat(reps)[:batch_size].to(device=device, dtype=dtype)


def _require_captum():
    try:
        from captum.attr import IntegratedGradients
    except ImportError as e:
        raise ImportError("Install captum: pip install captum") from e
    return IntegratedGradients


def _compute_ig_attributions(
    wrapper: DigitalTwinWrapper,
    eeg: torch.Tensor,
    drug: torch.Tensor,
    disease_label: torch.Tensor,
    baseline_eeg: torch.Tensor,
    baseline_drug: torch.Tensor,
    n_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single IG run; returns (attr_eeg, attr_drug) same shape as inputs."""
    IntegratedGradients = _require_captum()
    wrapper.eval()
    eeg_in = eeg.clone().detach().requires_grad_(True)
    drug_in = drug.clone().detach().requires_grad_(True)

    def _forward(e, d):
        dl = _expand_labels_to_batch(disease_label, e.shape[0])
        # Captum's compute_gradients does outputs[0] then torch.unbind(outputs); a 0-dim
        # scalar raises IndexError on outputs[0]. Force shape (1,).
        out = wrapper.forward_reconstruction(e, d, dl, enable_grad=True)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return out

    ig = IntegratedGradients(_forward)
    attrs = ig.attribute(
        (eeg_in, drug_in),
        baselines=(baseline_eeg, baseline_drug),
        n_steps=max(8, int(n_steps)),
    )
    return attrs[0], attrs[1]


def _ensemble_baselines(
    eeg_batch: torch.Tensor,
    *,
    zero_drug_baseline: bool = True,
    gaussian_sigma: float = 0.02,
    mean_eeg: Optional[torch.Tensor] = None,
    device: torch.device = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Build (baseline_eeg, baseline_drug) tuples for IG."""
    device = device or eeg_batch.device
    batch, d_eeg = eeg_batch.shape
    be0 = torch.zeros(batch, d_eeg, device=device, dtype=eeg_batch.dtype)
    bd0 = torch.zeros(batch, DRUG_DIM, device=device, dtype=eeg_batch.dtype)
    outs: List[Tuple[torch.Tensor, torch.Tensor]] = [(be0, bd0)]
    # Gaussian noise baseline (scaled to input magnitude)
    std = eeg_batch.detach().std(dim=1, keepdim=True).clamp_min(1e-6)
    noise = torch.randn(batch, d_eeg, device=device, dtype=eeg_batch.dtype) * std * gaussian_sigma
    outs.append((noise, bd0.clone()))
    if mean_eeg is not None:
        bm = mean_eeg.to(device).unsqueeze(0).expand(batch, -1)
        outs.append((bm, bd0.clone()))
    return outs


def run_integrated_gradients(
    wrapper: DigitalTwinWrapper,
    eeg_tensor: torch.Tensor,
    drug_tensor: torch.Tensor,
    *,
    disease_label: Optional[torch.Tensor] = None,
    target: str = "reconstruction",
    n_steps: int = 500,
    baseline_type: str = "ensemble",
    mean_background_eeg: Optional[torch.Tensor] = None,
    drug_baseline_tensor: Optional[torch.Tensor] = None,
    use_adaptive_convergence: bool = True,
    adaptive_max_steps: int = 500,
    ig_convergence_threshold: float = IG_CONVERGENCE_WARN,
) -> Dict[str, Any]:
    """
    Integrated Gradients with optional triple-baseline ensemble (zero, Gaussian, dataset mean).

    When ``use_adaptive_convergence`` is True, step count is chosen by ``find_converged_ig_steps``
    on the **zero** baseline (cheap search); final attributions still use ``baseline_type``.

    Args:
        n_steps: Maximum IG steps when adaptive; fixed steps when adaptive is disabled.
        use_adaptive_convergence: If False, run exactly ``n_steps`` without search.

    Returns:
        Dict with attributions (2185,), ``convergence_delta`` (final step-to-step stability),
        ``convergence_history`` for plotting, ``ig_converged`` flag.
    """
    if target != "reconstruction":
        raise ValueError("Only target='reconstruction' is implemented (scalar MSE).")
    device = eeg_tensor.device
    batch = eeg_tensor.shape[0]
    if disease_label is None:
        disease_label = torch.zeros(batch, device=device, dtype=eeg_tensor.dtype)

    labels_used: List[str]
    baseline_list: List[Tuple[torch.Tensor, torch.Tensor]]

    if baseline_type == "zero":
        baseline_list = [
            (torch.zeros(batch, EEG_FLAT_SIZE, device=device), torch.zeros(batch, DRUG_DIM, device=device))
        ]
        labels_used = ["zero"]
    elif baseline_type == "gaussian_noise":
        std = eeg_tensor.detach().std(dim=1, keepdim=True).clamp_min(1e-6)
        noise = torch.randn(batch, EEG_FLAT_SIZE, device=device, dtype=eeg_tensor.dtype) * std * 0.02
        baseline_list = [(noise, torch.zeros(batch, DRUG_DIM, device=device))]
        labels_used = ["gaussian_noise"]
    elif baseline_type == "mean_background":
        if mean_background_eeg is None:
            raise ValueError("mean_background_eeg required for baseline_type mean_background")
        bm = mean_background_eeg.to(device).unsqueeze(0).expand(batch, -1)
        bd = torch.zeros(batch, DRUG_DIM, device=device)
        baseline_list = [(bm, bd)]
        labels_used = ["mean_background"]
    elif baseline_type == "ensemble":
        baseline_list = list(_ensemble_baselines(eeg_tensor, mean_eeg=mean_background_eeg))
        labels_used = ["zero", "gaussian_scaled"]
        if mean_background_eeg is not None:
            bm = mean_background_eeg.to(device).unsqueeze(0).expand(batch, -1)
            bd = torch.zeros(batch, DRUG_DIM, device=device)
            baseline_list.append((bm, bd))
            labels_used.append("mean_background")
    else:
        raise ValueError(f"Unknown baseline_type {baseline_type}")

    be0 = torch.zeros(batch, EEG_FLAT_SIZE, device=device)
    bd0 = torch.zeros(batch, DRUG_DIM, device=device)
    convergence_history: List[Dict[str, Any]] = []

    if use_adaptive_convergence:
        n_used, conv_delta, convergence_history, captum_last = find_converged_ig_steps(
            wrapper,
            eeg_tensor,
            drug_tensor,
            disease_label,
            be0,
            bd0,
            max_steps=int(adaptive_max_steps),
            delta_threshold=float(ig_convergence_threshold),
        )
        convergence_delta = float(conv_delta)
        ig_converged = convergence_delta < ig_convergence_threshold
        captum_delta_mean = captum_last
    else:
        n_used = int(max(8, n_steps))
        convergence_delta = float("nan")
        ig_converged = False
        captum_delta_mean = float("nan")

    agg_eeg: List[torch.Tensor] = []
    agg_drug: List[torch.Tensor] = []
    for be, bd in baseline_list:
        ae, ad = _compute_ig_attributions(
            wrapper, eeg_tensor, drug_tensor, disease_label, be, bd, n_steps=n_used,
        )
        agg_eeg.append(ae.detach())
        agg_drug.append(ad.detach())

    ae_stack = torch.stack(agg_eeg, dim=0).mean(dim=0)
    ad_mean = torch.stack(agg_drug, dim=0).mean(dim=0)

    attributions = ae_stack.abs().mean(dim=0).detach().cpu().numpy()

    if use_adaptive_convergence and not np.isnan(convergence_delta) and (
        convergence_delta > IG_CONVERGENCE_WARN or not ig_converged
    ):
        warnings.warn(
            f"WARNING: IG convergence delta {convergence_delta:.4f} exceeds threshold {IG_CONVERGENCE_WARN}. "
            "Treat attributions as directional.",
            stacklevel=2,
        )

    out = {
        "attributions": attributions,
        "attributions_eeg_mean_abs": attributions,
        "attributions_eeg_raw": ae_stack.detach().cpu().numpy(),
        "attributions_drug_raw_mean": ad_mean.detach().cpu().numpy(),
        "convergence_delta": convergence_delta,
        "captum_delta_mean": captum_delta_mean,
        "convergence_history": convergence_history,
        "baseline_labels": labels_used,
        "baseline_type": baseline_type,
        "n_steps": int(n_used),
        "adaptive_max_steps": int(adaptive_max_steps),
        "ig_converged": bool(ig_converged),
        "feature_dim": EEG_FLAT_SIZE,
    }
    _ = drug_baseline_tensor  # silence unused — reserved for extensions
    return out


def run_ig_per_drug_condition(
    wrapper: DigitalTwinWrapper,
    eeg_tensor: torch.Tensor,
    baseline_drug: torch.Tensor,
    donepezil_drug: torch.Tensor,
    memantine_drug: torch.Tensor,
    *,
    disease_label: Optional[torch.Tensor] = None,
    n_steps: int = 500,
    mean_background_eeg: Optional[torch.Tensor] = None,
    use_adaptive_convergence: bool = False,
    adaptive_max_steps: int = 500,
) -> Dict[str, np.ndarray]:
    """
    IG mean-|attr| vectors for baseline / donepezil / memantine embeddings (same EEG).

    Returns arrays (2185,) each and deltas vs baseline.

    Complexity: ~3 × cost of ``run_integrated_gradients``.
    """
    conditions = {}
    keys = ["baseline", "donepezil", "memantine"]
    drugs = [baseline_drug, donepezil_drug, memantine_drug]
    raw_attrs = {}
    for name, dv in zip(keys, drugs):
        res = run_integrated_gradients(
            wrapper,
            eeg_tensor,
            dv,
            disease_label=disease_label,
            n_steps=n_steps,
            baseline_type="ensemble",
            mean_background_eeg=mean_background_eeg,
            use_adaptive_convergence=use_adaptive_convergence,
            adaptive_max_steps=adaptive_max_steps,
        )
        raw_attrs[name] = res["attributions_eeg_mean_abs"]

    base = raw_attrs["baseline"]
    return {
        "baseline": base,
        "donepezil": raw_attrs["donepezil"],
        "memantine": raw_attrs["memantine"],
        "donepezil_delta": raw_attrs["donepezil"] - base,
        "memantine_delta": raw_attrs["memantine"] - base,
    }


def run_ig_per_subject(
    wrapper: DigitalTwinWrapper,
    eeg_array: np.ndarray,
    drug_tensor: torch.Tensor,
    subject_labels: np.ndarray,
    *,
    n_steps: int = 200,
    disease_labels: Optional[np.ndarray] = None,
    mean_background_eeg: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Per-row IG with batch size 1 for attribution stability.

    Args:
        eeg_array: (N, 2185).
        drug_tensor: (N, 384) per-subject drug embeddings.
        subject_labels: (N,) arbitrary subject ids for keys.
        disease_labels: (N,) 0=HC 1=AD — if None, zeros.

    Returns:
        per_subject dict, AD/HC mean attributions, difference vector.

    Complexity: O(N * n_steps * forward * K baselines).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)
    n = eeg_array.shape[0]
    mean_bg = torch.from_numpy(mean_background_eeg).float().to(device) if mean_background_eeg is not None else None

    per_subject: Dict[int, np.ndarray] = {}
    dl_all = []
    attrs_ad = []
    attrs_hc = []

    for i in range(n):
        eeg = torch.from_numpy(eeg_array[i : i + 1]).float().to(device)
        drug = drug_tensor[i : i + 1].detach()
        dl = (
            torch.from_numpy(disease_labels[i : i + 1]).float().to(device)
            if disease_labels is not None
            else torch.zeros(1, device=device)
        )
        dl_all.append(dl.item())
        res = run_integrated_gradients(
            wrapper,
            eeg,
            drug,
            disease_label=dl,
            n_steps=n_steps,
            baseline_type="ensemble",
            mean_background_eeg=mean_bg,
            use_adaptive_convergence=False,
            adaptive_max_steps=int(n_steps),
        )
        idx = int(subject_labels[i])
        vec = res["attributions_eeg_mean_abs"]
        per_subject[idx] = vec
        if dl.item() >= 0.5:
            attrs_ad.append(vec)
        else:
            attrs_hc.append(vec)

    ad_mean = np.mean(np.stack(attrs_ad, axis=0), axis=0) if attrs_ad else np.zeros(EEG_FLAT_SIZE)
    hc_mean = np.mean(np.stack(attrs_hc, axis=0), axis=0) if attrs_hc else np.zeros(EEG_FLAT_SIZE)

    return {
        "per_subject": per_subject,
        "ad_mean_attribution": ad_mean,
        "hc_mean_attribution": hc_mean,
        "ad_minus_hc": ad_mean - hc_mean,
        "disease_labels_used": dl_all,
    }


def compute_ig_feature_block_summary(
    attributions: np.ndarray,
    feature_block_sizes: Optional[Dict[str, Union[int, Tuple[int, int]]]] = None,
    *,
    project_root: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Aggregate per-block statistics and top-10 indices within each block.

    ``normalized_importance`` uses **mean absolute attribution per feature** within each
    block, then normalizes across blocks so they sum to 1. Summing ``|attr|`` over all
    features in a block unfairly favors large blocks (e.g. connectivity vs band powers).

    Args:
        attributions: Shape (2185,) mean absolute attribution vector.
        feature_block_sizes: Either lengths {'psd':380,...} or explicit intervals from ``feature_bounds``.

    Returns:
        Nested dict with per-block stats and top indices.

    Complexity: O(2185).
    """
    from pathlib import Path as _Path

    blocks = load_feature_blocks_from_metadata(_Path(project_root) if project_root else None)

    summary: Dict[str, Any] = {"blocks": {}, "normalized_importance": {}}

    block_mean_abs: Dict[str, float] = {}
    for name, (start, end) in blocks.items():
        seg = attributions[start:end]
        mean_abs = float(np.mean(np.abs(seg)))
        block_mean_abs[name] = mean_abs
        block_sum = {
            "mean_abs_attr": mean_abs,
            "std_attr": float(np.std(seg)),
            "max_attr": float(np.max(np.abs(seg))),
            "slice": [int(start), int(end)],
        }
        top_local = np.argsort(np.abs(seg))[-10:][::-1]
        block_sum["top10_indices_global"] = [int(start + j) for j in top_local]
        summary["blocks"][name] = block_sum

    total_mean = sum(block_mean_abs.values()) + 1e-12
    for name in blocks:
        summary["normalized_importance"][name] = block_mean_abs[name] / total_mean

    return summary
