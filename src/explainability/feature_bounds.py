"""
Feature block indices for 2185-D EEG vectors (aligned with Phase 1 + prepare_target_features).

Loads optional shapes from ``data/eeg_features/*_features_metadata.json`` when present;
falls back to MODEL_CONFIG-derived sizes (must match train_phase2).

Complexity: O(1) after first load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from src.models.config import DATA_ROOT, MODEL_CONFIG

N_CHANNELS = MODEL_CONFIG["eeg_encoder"]["n_channels"]
N_BANDS = MODEL_CONFIG["eeg_encoder"]["n_bands"]
PSD_DIM = MODEL_CONFIG["eeg_encoder"]["psd_dim"]

PSD_SIZE = N_CHANNELS * PSD_DIM
BAND_SIZE = N_CHANNELS * N_BANDS
TRIU_SIZE = N_CHANNELS * (N_CHANNELS - 1) // 2
CONNECTIVITY_SIZE = N_BANDS * TRIU_SIZE * 2
EEG_FLAT_SIZE = PSD_SIZE + BAND_SIZE + CONNECTIVITY_SIZE


def default_feature_blocks() -> Dict[str, Tuple[int, int]]:
    """Inclusive-exclusive half-open intervals [start, end) for PSD, band_powers, connectivity."""
    return {
        "psd": (0, PSD_SIZE),
        "band_powers": (PSD_SIZE, PSD_SIZE + BAND_SIZE),
        "connectivity": (PSD_SIZE + BAND_SIZE, EEG_FLAT_SIZE),
    }


def load_feature_blocks_from_metadata(
    project_root: Path | None = None,
) -> Dict[str, Tuple[int, int]]:
    """
    Validate dimensions against JSON metadata if available; else use defaults.

    Returns:
        Same structure as ``default_feature_blocks``.
    """
    root = project_root or Path(__file__).resolve().parent.parent.parent
    meta_path = root / "data" / "eeg_features" / "HC_features_metadata.json"
    blocks = default_feature_blocks()
    if not meta_path.is_file():
        return blocks
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        psd_shape = meta["shapes"]["psd"]
        bp_shape = meta["shapes"]["band_powers"]
        psd_n = int(psd_shape[1] * psd_shape[2])
        bp_n = int(bp_shape[1] * bp_shape[2])
        conn_n = EEG_FLAT_SIZE - psd_n - bp_n
        if psd_n + bp_n + conn_n != EEG_FLAT_SIZE:
            return blocks
        return {
            "psd": (0, psd_n),
            "band_powers": (psd_n, psd_n + bp_n),
            "connectivity": (psd_n + bp_n, EEG_FLAT_SIZE),
        }
    except (KeyError, json.JSONDecodeError, ValueError):
        return blocks


def feature_block_sizes_dict(blocks: Dict[str, Tuple[int, int]]) -> Dict[str, int]:
    """Map block name -> length."""
    return {name: end - start for name, (start, end) in blocks.items()}
