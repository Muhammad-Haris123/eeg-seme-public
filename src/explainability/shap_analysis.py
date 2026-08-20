"""
KernelSHAP (Lundberg & Lee, 2017) for EEG feature importance with fixed drug conditioning.

The model-agnostic explainer uses a scalar **per-sample mean reconstruction MSE** as the
prediction; only the 2185-D EEG vector is treated as input features (drug is fixed per run).

Complexity: O(n_test * nsamples * n_features) in the worst case for KernelSHAP—use
``--fast`` and cache (pkl) for interactive work.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.explainability.model_wrapper import DigitalTwinWrapper, DRUG_DIM, EEG_FLAT_SIZE
from src.explainability.feature_bounds import default_feature_blocks, load_feature_blocks_from_metadata

logger = logging.getLogger(__name__)

CHUNK = 10


def _mean_mse_per_row(
    wrapper: DigitalTwinWrapper,
    eeg_np: np.ndarray,
    drug_row: np.ndarray,
    disease_val: float,
    device: torch.device,
) -> np.ndarray:
    """Numpy (N, 2185) -> (N,) mean MSE per row. Deterministic forward, no grad."""
    n = eeg_np.shape[0]
    out = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, n, CHUNK):
            e = eeg_np[s : s + CHUNK]
            b = e.shape[0]
            x = torch.from_numpy(e).float().to(device)
            d = torch.from_numpy(np.tile(drug_row, (b, 1))).float().to(device)
            dl = torch.full((b,), disease_val, device=device, dtype=x.dtype)
            recon = wrapper.forward(x, d, dl, enable_grad=False)
            mse = ((recon - x) ** 2).mean(dim=1).cpu().numpy()
            out[s : s + b] = mse
    return out


def build_shap_explainer(
    wrapper: DigitalTwinWrapper,
    background_data: np.ndarray,
    drug_tensor_row: np.ndarray,
    *,
    disease_val: float = 0.0,
    device: Optional[torch.device] = None,
    n_background: int = 50,
) -> Tuple[Any, Callable[[np.ndarray], np.ndarray]]:
    """
    Configure ``shap.KernelExplainer`` on EEG features only.

    Args:
        wrapper: DigitalTwinWrapper on ``device``.
        background_data: (N, 2185) training EEG features.
        drug_tensor_row: (384,) ChemBERTa vector repeated for every row.
        disease_val: Scalar disease label for all evaluations.
        n_background: Size of summarized background passed to SHAP.

    Returns:
        (explainer, predict_fn); ``predict_fn`` accepts (batch, 2185).

    Complexity:
        Each predict call is O(batch * forward).

    Raises:
        ImportError: if ``shap`` is missing.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("pip install shap") from e

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)
    bg = background_data[:n_background].astype(np.float64)
    drug_row = np.asarray(drug_tensor_row).reshape(-1).astype(np.float64)

    def predict_fn(X: np.ndarray) -> np.ndarray:
        return _mean_mse_per_row(wrapper, np.asarray(X, dtype=np.float64), drug_row, disease_val, device)

    explainer = shap.KernelExplainer(predict_fn, bg)
    return explainer, predict_fn


def compute_shap_values(
    explainer: Any,
    test_data: np.ndarray,
    *,
    nsamples: int = 150,
    drug_label: str = "baseline",
    cache_path: Optional[Path] = None,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Run KernelSHAP and optionally cache ``pkl``.

    Args:
        explainer: ``shap.KernelExplainer``.
        test_data: (n_test, 2185).
        nsamples: ``link`` kernel samples (SHAP argument).
        drug_label: Used in filename / metadata.
        cache_path: If set and exists, load from disk.

    Returns:
        Dict with ``shap_values``, ``expected_value``, timing, etc.

    Complexity:
        Very high for large ``nsamples`` and many test rows—prefer small ``n_test`` for FYP runs.
    """
    if cache_path and cache_path.is_file():
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if not silent:
            logger.info("Loaded SHAP cache %s", cache_path)
        return data

    t0 = time.perf_counter()
    import shap

    test_data = np.asarray(test_data, dtype=np.float64)
    sv = explainer.shap_values(test_data, nsamples=int(nsamples), silent=silent)
    if isinstance(sv, list):
        sv = sv[0]
    ev = float(explainer.expected_value)
    elapsed = time.perf_counter() - t0

    result = {
        "shap_values": np.asarray(sv),
        "expected_value": ev,
        "drug_condition": drug_label,
        "n_test_subjects": int(test_data.shape[0]),
        "computation_time_seconds": float(elapsed),
        "nsamples": int(nsamples),
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    return result


def compute_shap_all_conditions(
    wrapper: DigitalTwinWrapper,
    train_data: np.ndarray,
    test_data: np.ndarray,
    baseline_drug: np.ndarray,
    donepezil_drug: np.ndarray,
    memantine_drug: np.ndarray,
    *,
    nsamples: int = 150,
    cache_dir: Optional[Path] = None,
    disease_val: float = 0.0,
    n_background: int = 50,
) -> Dict[str, Any]:
    """
    SHAP under three drug embeddings; compute deltas vs baseline.

    Args:
        train_data: Background EEG (N_train, 2185).
        test_data: Explained instances (N_test, 2185).
        *_drug: (384,) vectors.

    Returns:
        Nested dict with per-condition SHAP arrays and ``delta_*`` keys.

    Complexity:
        3 × single-condition SHAP cost.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)
    out: Dict[str, Any] = {}
    drugs = {
        "baseline": baseline_drug,
        "donepezil": donepezil_drug,
        "memantine": memantine_drug,
    }
    for name, dv in drugs.items():
        cp = (cache_dir / f"shap_{name}.pkl") if cache_dir else None
        explainer, _ = build_shap_explainer(
            wrapper,
            train_data,
            dv,
            disease_val=disease_val,
            device=device,
            n_background=n_background,
        )
        out[name] = compute_shap_values(explainer, test_data, nsamples=nsamples, drug_label=name, cache_path=cp)

    sb = out["baseline"]["shap_values"]
    out["delta_donepezil_minus_baseline"] = out["donepezil"]["shap_values"] - sb
    out["delta_memantine_minus_baseline"] = out["memantine"]["shap_values"] - sb
    return out


def get_shap_feature_ranking(
    shap_values: np.ndarray,
    *,
    feature_names: Optional[List[str]] = None,
    top_k: int = 30,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Aggregate |SHAP| across test rows -> global ranking table.

    Args:
        shap_values: (n_test, 2185).
        feature_names: Optional length-2185 names (else ``feat_i``).

    Returns:
        DataFrame sorted by ``mean_abs_shap`` descending.

    Complexity: O(n_test * 2185).
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    blocks = load_feature_blocks_from_metadata(project_root)

    def block_of(idx: int) -> str:
        for name, (a, b) in blocks.items():
            if a <= idx < b:
                return name
        return "unknown"

    names = feature_names or [f"feat_{i}" for i in range(EEG_FLAT_SIZE)]
    rows = []
    order = np.argsort(-mean_abs)
    for rank, j in enumerate(order[:top_k], start=1):
        i = int(j)
        rows.append(
            {
                "feature_idx": i,
                "feature_name": names[i] if i < len(names) else f"feat_{i}",
                "feature_block": block_of(i),
                "mean_abs_shap": float(mean_abs[i]),
                "std_shap": float(np.std(shap_values[:, i])),
                "rank": rank,
            }
        )
    return pd.DataFrame(rows)


def compute_shap_interaction_summary(
    shap_values_dict: Dict[str, np.ndarray],
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compare block-level mean |SHAP| ranks across ``baseline``, ``donepezil``, ``memantine``.

    Returns:
        ``block_rank_changes``: list of dicts with keys ``block``, ``rank_baseline``, etc.

    Complexity: O(3 * 2185).
    """
    blocks = load_feature_blocks_from_metadata(project_root)
    conditions = ["baseline", "donepezil", "memantine"]
    block_scores: Dict[str, Dict[str, float]] = {c: {} for c in conditions}

    for cond in conditions:
        if cond not in shap_values_dict:
            continue
        sv = shap_values_dict[cond]
        mean_abs = np.mean(np.abs(sv), axis=0)
        for name, (a, b) in blocks.items():
            block_scores[cond][name] = float(np.mean(mean_abs[a:b]))

    # ranks per condition (1 = highest)
    ranks: Dict[str, Dict[str, int]] = {}
    for cond in conditions:
        if cond not in block_scores:
            continue
        items = sorted(block_scores[cond].items(), key=lambda x: -x[1])
        ranks[cond] = {blk: r + 1 for r, (blk, _) in enumerate(items)}

    changes = []
    for blk in blocks:
        row = {"block": blk}
        for cond in conditions:
            row[f"rank_{cond}"] = ranks.get(cond, {}).get(blk, np.nan)
        changes.append(row)

    return {
        "block_mean_abs_shap": block_scores,
        "block_ranks": ranks,
        "block_rank_changes": changes,
    }


def aggregate_eeg_to_block_means(
    eeg_array: np.ndarray,
    blocks: Dict[str, Tuple[int, int]],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Map (N, 2185) EEG to (N, n_blocks) where each column is the within-block mean."""
    order = tuple(blocks.keys())
    cols = []
    for name in order:
        start, end = blocks[name]
        cols.append(eeg_array[:, start:end].mean(axis=1))
    return np.stack(cols, axis=1), order


def block_means_to_full_eeg(
    reference: np.ndarray,
    block_vals: np.ndarray,
    blocks: Dict[str, Tuple[int, int]],
) -> np.ndarray:
    """
    Expand (B, n_blocks) coalition block means to full (B, 2185) by scaling each block
    in ``reference`` so its mean matches the target column (preserves intra-block shape).
    """
    ref = np.asarray(reference, dtype=np.float64).reshape(1, -1).copy()
    block_vals = np.atleast_2d(np.asarray(block_vals, dtype=np.float64))
    if block_vals.shape[1] != len(blocks):
        raise ValueError(f"Expected {len(blocks)} block columns, got {block_vals.shape[1]}")
    B = block_vals.shape[0]
    out = np.repeat(ref, B, axis=0)
    order = list(blocks.keys())
    for col, name in enumerate(order):
        start, end = blocks[name]
        seg = out[:, start:end]
        cur_mean = seg.mean(axis=1, keepdims=True) + 1e-12
        scale = block_vals[:, col : col + 1] / cur_mean
        out[:, start:end] = seg * scale
    return out


def _make_full_eeg_from_coalition_mask(
    coalition_row: np.ndarray,
    reference_eeg: np.ndarray,
    blocks: Dict[str, Tuple[int, int]],
    block_order: Tuple[str, ...],
) -> np.ndarray:
    """coalition_row[i] in {0,1}: 1 = block present, 0 = absent (zeroed)."""
    full = np.asarray(reference_eeg, dtype=np.float64).reshape(1, -1).copy()
    cr = np.asarray(coalition_row).reshape(-1)
    for i, name in enumerate(block_order):
        if i >= len(cr) or float(cr[i]) < 0.5:
            s, e = blocks[name]
            full[:, s:e] = 0.0
    return full


def compute_block_level_shap(
    wrapper: DigitalTwinWrapper,
    train_eeg: np.ndarray,
    test_eeg: np.ndarray,
    drug_row: np.ndarray,
    *,
    disease_val: float = 0.0,
    nsamples: int = 500,
    n_background: int = 50,
    project_root: Optional[Path] = None,
    silent: bool = False,
    max_test_subjects: int = 5,
) -> Dict[str, Any]:
    """
    Block-level KernelSHAP with **binary coalition masking** (present = input values,
    absent = zeros). Avoids unstable multiplicative scaling when block means are tiny.

    Prediction: per-sample mean reconstruction MSE via ``_mean_mse_per_row``.
    """
    import shap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = wrapper.to(device)
    blocks = load_feature_blocks_from_metadata(project_root)
    block_order = tuple(blocks.keys())
    n_blocks = len(block_order)

    drug_row = np.asarray(drug_row).reshape(-1).astype(np.float64)

    n_bg = min(n_background, len(train_eeg))
    background = np.ones((n_bg, n_blocks), dtype=np.float32)

    n_test = min(int(max_test_subjects), int(test_eeg.shape[0]))
    all_sv: List[np.ndarray] = []
    expected_values: List[float] = []
    times: List[float] = []
    base_mse_per_subject: List[float] = []

    for i in range(n_test):
        ref_eeg = test_eeg[i : i + 1].astype(np.float64)

        def _predict_fn(coalition_matrix: np.ndarray) -> np.ndarray:
            coal = np.atleast_2d(np.asarray(coalition_matrix, dtype=np.float64))
            out_mse = np.empty(coal.shape[0], dtype=np.float64)
            for row_idx in range(coal.shape[0]):
                full = _make_full_eeg_from_coalition_mask(coal[row_idx], ref_eeg, blocks, block_order)
                out_mse[row_idx : row_idx + 1] = _mean_mse_per_row(
                    wrapper, full, drug_row, disease_val, device
                )
            return out_mse

        ones = np.ones((1, n_blocks), dtype=np.float32)
        base_mse_per_subject.append(float(_predict_fn(ones)[0]))

        t0 = time.perf_counter()
        explainer = shap.KernelExplainer(_predict_fn, background)
        try:
            sv = explainer.shap_values(
                ones,
                nsamples=int(nsamples),
                silent=silent,
                l1_reg=f"num_features({n_blocks})",
            )
        except Exception:
            sv = explainer.shap_values(ones, nsamples=int(nsamples), silent=silent)
        times.append(time.perf_counter() - t0)
        if isinstance(sv, list):
            sv = sv[0]
        all_sv.append(np.asarray(sv).reshape(1, -1))
        expected_values.append(float(explainer.expected_value))

    shap_stack = np.vstack(all_sv)

    mean_abs_per_block = {
        block_order[j]: float(np.mean(np.abs(shap_stack[:, j])))
        for j in range(shap_stack.shape[1])
    }

    block_importance = {}
    tot = sum(mean_abs_per_block.values()) + 1e-12
    for j, name in enumerate(block_order):
        vals = shap_stack[:, j]
        block_importance[name] = {
            "mean_abs_shap": float(np.abs(vals).mean()),
            "mean_shap": float(vals.mean()),
            "std_shap": float(vals.std()),
            "all_values": vals.tolist(),
            "normalized": float(mean_abs_per_block[name] / tot),
        }

    logger.info("\n=== BLOCK SHAP RAW (coalition mask) ===")
    for name in block_order:
        imp = block_importance[name]
        logger.info(
            "  %s: mean_abs=%.6f values=%s",
            name,
            imp["mean_abs_shap"],
            [f"{v:.6f}" for v in imp["all_values"]],
        )

    near_zero = sum(1 for v in mean_abs_per_block.values() if abs(float(v)) < 1e-15)
    if near_zero > 0:
        warnings.warn(
            "Block-level SHAP: some blocks near zero. shap_values=\n"
            + np.array2string(shap_stack, precision=6)
        )

    return {
        "shap_values": shap_stack,
        "expected_value_per_instance": np.asarray(expected_values, dtype=np.float64),
        "expected_value_mean": float(np.mean(expected_values)) if expected_values else float("nan"),
        "block_names": list(block_order),
        "method": "block_coalition_kernel_shap",
        "nsamples": int(nsamples),
        "computation_time_seconds_total": float(sum(times)),
        "mean_abs_per_block": mean_abs_per_block,
        "block_importance": block_importance,
        "base_mse_per_subject": base_mse_per_subject,
    }


def compute_block_level_shap_all_conditions(
    wrapper: DigitalTwinWrapper,
    train_data: np.ndarray,
    test_data: np.ndarray,
    baseline_drug: np.ndarray,
    donepezil_drug: np.ndarray,
    memantine_drug: np.ndarray,
    *,
    disease_val: float = 0.0,
    nsamples: int = 300,
    n_background: int = 50,
    project_root: Optional[Path] = None,
    silent: bool = False,
    max_test_subjects: int = 5,
) -> Dict[str, Any]:
    """
    Block-level KernelSHAP for baseline / donepezil / memantine (same ``test_data`` rows).

    Returns:
        Bundle with per-condition outputs plus deltas vs baseline (mean SHAP arrays).
    """
    drugs = {
        "baseline": baseline_drug,
        "donepezil": donepezil_drug,
        "memantine": memantine_drug,
    }
    out: Dict[str, Any] = {}
    for name, dv in drugs.items():
        out[name] = compute_block_level_shap(
            wrapper,
            train_data,
            test_data,
            dv,
            disease_val=disease_val,
            nsamples=nsamples,
            n_background=n_background,
            project_root=project_root,
            silent=silent,
            max_test_subjects=max_test_subjects,
        )
    sb = out["baseline"]["shap_values"]
    out["delta_donepezil_minus_baseline"] = out["donepezil"]["shap_values"] - sb
    out["delta_memantine_minus_baseline"] = out["memantine"]["shap_values"] - sb
    return out


def export_block_shap_summary_table(bundle: Dict[str, Any]) -> pd.DataFrame:
    """Rows = blocks; columns = mean |SHAP| per condition (+ deltas)."""
    rows = []
    names = bundle["baseline"]["block_names"]
    for j, blk in enumerate(names):
        row: Dict[str, Any] = {"block": blk}
        for cond in ("baseline", "donepezil", "memantine"):
            sv = bundle[cond]["shap_values"]
            row[f"mean_abs_shap_{cond}"] = float(np.mean(np.abs(sv[:, j])))
        rows.append(row)
    return pd.DataFrame(rows)
