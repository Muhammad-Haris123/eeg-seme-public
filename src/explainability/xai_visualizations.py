"""
Structured visualizations for EEG flat-vector attributions (CAM-style maps).

Maps 2185-D attribution vectors into channel × frequency / channel × band grids
and connectivity summaries. Not classical Grad-CAM (no conv layers); thesis text
should call these "feature-space saliency maps" or "importance heatmaps".
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.explainability.model_wrapper import (
    N_CHANNELS,
    N_BANDS,
    PSD_DIM,
    PSD_SIZE,
    BAND_SIZE,
    TRIU_SIZE,
)


def attribution_to_psd_grid(mean_abs_attr_2185: np.ndarray) -> np.ndarray:
    """First PSD_SIZE dims → (n_channels, psd_dim)."""
    v = mean_abs_attr_2185[:PSD_SIZE]
    return v.reshape(N_CHANNELS, PSD_DIM)


def attribution_to_band_grid(mean_abs_attr_2185: np.ndarray) -> np.ndarray:
    """Band segment → (n_channels, n_bands)."""
    sl = slice(PSD_SIZE, PSD_SIZE + BAND_SIZE)
    v = mean_abs_attr_2185[sl]
    return v.reshape(N_CHANNELS, N_BANDS)


def attribution_to_connectivity_band_matrices(mean_abs_attr_2185: np.ndarray) -> list[np.ndarray]:
    """
    Connectivity segment: per band, coh triu + plv triu (each length TRIU_SIZE).
    Aggregate coh+plv into one symmetric importance per edge.
    """
    conn = mean_abs_attr_2185[PSD_SIZE + BAND_SIZE :]
    triu_idx = np.triu_indices(N_CHANNELS, k=1)
    mats: list[np.ndarray] = []
    offset = 0
    for _ in range(N_BANDS):
        coh = conn[offset : offset + TRIU_SIZE]
        plv = conn[offset + TRIU_SIZE : offset + 2 * TRIU_SIZE]
        offset += 2 * TRIU_SIZE
        edge_imp = coh + plv
        full = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.float64)
        full[triu_idx[0], triu_idx[1]] = edge_imp
        full = full + full.T
        mats.append(full)
    return mats


def save_psd_band_heatmaps(
    mean_abs_attr: np.ndarray,
    out_dir: Path,
    prefix: str = "xai",
) -> dict[str, str]:
    """Save PNGs for PSD and band power grids. Returns paths relative to out_dir."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    psd = attribution_to_psd_grid(mean_abs_attr)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(psd, aspect="auto", cmap="magma")
    ax.set_xlabel("PSD bin index")
    ax.set_ylabel("EEG channel index")
    ax.set_title("Mean |attribution|: PSD block (channel × frequency bins)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    p = out_dir / f"{prefix}_psd_importance.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["psd"] = str(p.name)

    bands = attribution_to_band_grid(mean_abs_attr)
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(bands, aspect="auto", cmap="viridis")
    ax.set_xticks(range(N_BANDS))
    ax.set_xticklabels(band_names, rotation=30, ha="right")
    ax.set_ylabel("EEG channel index")
    ax.set_title("Mean |attribution|: band powers (channel × band)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    p2 = out_dir / f"{prefix}_band_importance.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths["bands"] = str(p2.name)

    return paths


def save_connectivity_band_maps(
    mean_abs_attr: np.ndarray,
    out_dir: Path,
    prefix: str = "xai",
) -> dict[str, str]:
    """One heatmap per frequency band for connectivity-derived importance."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    mats = attribution_to_connectivity_band_matrices(mean_abs_attr)
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]
    paths: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (mat, name) in enumerate(zip(mats, band_names)):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(mat, cmap="coolwarm")
        ax.set_title(f"Connectivity importance ({name}) — coh+PLV |attr|")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fp = out_dir / f"{prefix}_connectivity_{name}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        paths[name] = str(fp.name)
    return paths


def save_drug_embedding_importance(
    mean_abs_attr_drug: np.ndarray,
    out_dir: Path,
    prefix: str = "xai",
    chunk_size: int = 32,
) -> str | None:
    """Bar chart: mean |attr| per chunk of ChemBERTa dimensions (384 → 12 bars)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    n = mean_abs_attr_drug.shape[0]
    chunks = []
    labels = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append(float(np.mean(mean_abs_attr_drug[start:end])))
        labels.append(f"{start}-{end - 1}")
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 3))
    ax.bar(range(len(chunks)), chunks, color="#c44e52")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean |attribution|")
    ax.set_title("ChemBERTa embedding: attribution by dimension block")
    plt.tight_layout()
    fp = out_dir / f"{prefix}_drug_embedding_chunks.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return str(fp.name)
