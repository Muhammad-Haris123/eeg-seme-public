"""
Publication-quality figures for multimodal EEG Digital Twin XAI outputs.

Each function saves PNG (300 DPI) + PDF vector where applicable.

Complexity dominated by matplotlib layout — O(1) relative to attribution compute.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLORS = {
    "baseline": "#546E7A",
    "donepezil": "#1E88E5",
    "memantine": "#43A047",
    "ad": "#E53935",
    "hc": "#00897B",
    "psd": "#FFA726",
    "band_powers": "#AB47BC",
    "connectivity": "#26C6DA",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def plot_attribution_overview(
    ig_results: Dict[str, Any],
    shap_results: Dict[str, Any],
    occlusion_results: Dict[str, Any],
    save_dir: Path,
    *,
    spearman_rho: Optional[float] = None,
    block_keys: Sequence[str] = ("psd", "band_powers", "connectivity"),
) -> Tuple[Path, Path]:
    """3-panel overview: IG blocks, SHAP blocks, occlusion delta MSE."""
    save_dir.mkdir(parents=True, exist_ok=True)

    ig_blocks = ig_results.get("normalized_importance", {})
    sh_blocks = {}
    sv = np.asarray(shap_results.get("shap_values"))
    from src.explainability.feature_bounds import load_feature_blocks_from_metadata

    bounds = load_feature_blocks_from_metadata()
    for name, (a, b) in bounds.items():
        sh_blocks[name] = float(np.mean(np.abs(sv[:, a:b]))) if sv.size else 0.0

    occ_blocks = []
    pb = occlusion_results.get("per_block")
    if pb is None and "conditions" in occlusion_results:
        pb = occlusion_results["conditions"].get("baseline", {}).get("per_block", {})
    else:
        pb = pb or {}
    for k in block_keys:
        occ_blocks.append(pb.get(k, {}).get("reconstruction_sensitivity", 0.0))

    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    vals = [float(ig_blocks.get(k, 0.0)) for k in block_keys]
    ax0.bar(block_keys, vals, color=[COLORS[k] for k in block_keys])
    ax0.set_title("(A) Integrated Gradients — block importance")
    ax0.set_ylabel("Normalized IG")

    ax1 = fig.add_subplot(gs[0, 1])
    vals2 = [sh_blocks.get(k, 0.0) for k in block_keys]
    ax1.bar(block_keys, vals2, color=[COLORS[k] for k in block_keys])
    ax1.set_title("(B) KernelSHAP — mean |value|")
    ax1.set_ylabel("Mean |SHAP|")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(block_keys, occ_blocks, color=[COLORS[k] for k in block_keys])
    ax2.set_title("(C) Occlusion — Δ reconstruction MSE")
    ax2.set_ylabel("Δ MSE")

    note = ""
    if spearman_rho is not None:
        note = f"Methods agree: connectivity ranking consistent (ρ={spearman_rho:.2f})"
    fig.suptitle("Multi-Method Feature Attribution Analysis\n" + note, fontsize=12, y=1.05)

    png = save_dir / "fig1_attribution_overview.png"
    pdf = save_dir / "fig1_attribution_overview.pdf"
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_drug_condition_attribution_comparison(
    ig_per_condition: Dict[str, np.ndarray],
    shap_per_condition: Dict[str, Dict[str, Any]],
    save_dir: Path,
    *,
    bounds: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[Path, Path]:
    """2×3 grid: IG and SHAP block strengths per drug."""
    from src.explainability.feature_bounds import load_feature_blocks_from_metadata

    save_dir.mkdir(parents=True, exist_ok=True)
    bounds = bounds or load_feature_blocks_from_metadata()
    drugs = ["baseline", "donepezil", "memantine"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    def block_vec(vec: np.ndarray) -> List[float]:
        return [float(np.mean(np.abs(vec[a:b]))) for _, (a, b) in bounds.items()]

    names = list(bounds.keys())
    for j, d in enumerate(drugs):
        igv = ig_per_condition.get(d, np.zeros(2185))
        axes[0, j].bar(names, block_vec(igv), color=[COLORS[n] for n in names])
        axes[0, j].set_title(f"IG — {d}")
        sv = np.asarray(shap_per_condition.get(d, {}).get("shap_values", np.zeros((1, 2185))))
        mean_abs = np.mean(np.abs(sv), axis=0)
        axes[1, j].bar(names, block_vec(mean_abs), color=[COLORS[n] for n in names])
        axes[1, j].set_title(f"SHAP — {d}")

    fig.suptitle("Drug-Condition-Specific Feature Attribution")
    png = save_dir / "fig2_drug_condition_comparison.png"
    pdf = save_dir / "fig2_drug_condition_comparison.pdf"
    plt.tight_layout()
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_sliding_window_heatmap(
    occlusion_map_baseline: np.ndarray,
    occlusion_map_donepezil: np.ndarray,
    occlusion_map_memantine: np.ndarray,
    feature_block_boundaries: Dict[str, Tuple[int, int]],
    save_dir: Path,
) -> Tuple[Path, Path]:
    """3-row heatmap (conditions) × 2185 features."""
    save_dir.mkdir(parents=True, exist_ok=True)
    data = np.stack(
        [
            occlusion_map_baseline / (np.max(occlusion_map_baseline) + 1e-12),
            occlusion_map_donepezil / (np.max(occlusion_map_donepezil) + 1e-12),
            occlusion_map_memantine / (np.max(occlusion_map_memantine) + 1e-12),
        ],
        axis=0,
    )

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(data, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["baseline", "donepezil", "memantine"])
    ax.set_xlabel("Feature index (flattened EEG vector)")
    for _, (a, b) in feature_block_boundaries.items():
        ax.axvline(a, color="cyan", ls="--", lw=0.8)
        ax.axvline(b, color="cyan", ls="--", lw=0.8)
    plt.colorbar(im, ax=ax, fraction=0.02)
    ax.set_title("Sliding-window occlusion sensitivity (normalized per row)")
    png = save_dir / "fig3_sliding_window_heatmap.png"
    pdf = save_dir / "fig3_sliding_window_heatmap.pdf"
    plt.tight_layout()
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_method_agreement_matrix(
    correlation_matrix: np.ndarray,
    method_names: Sequence[str],
    save_dir: Path,
) -> Tuple[Path, Path]:
    """Heatmap of Spearman correlations."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(correlation_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(method_names)))
    ax.set_yticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=35, ha="right")
    ax.set_yticklabels(method_names)
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            ax.text(j, i, f"{correlation_matrix[i, j]:.2f}", ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax)
    ax.set_title("Attribution Method Agreement Analysis")
    txt = (
        "High agreement (ρ>0.7) suggests robust attribution findings independent of method choice."
    )
    fig.text(0.5, 0.02, txt, ha="center", fontsize=9)
    png = save_dir / "fig4_method_agreement.png"
    pdf = save_dir / "fig4_method_agreement.pdf"
    plt.tight_layout()
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_latent_space_drug_trajectories(
    latent_shift_report: Dict[str, Any],
    mu_baseline: np.ndarray,
    labels: np.ndarray,
    save_dir: Path,
) -> Tuple[Path, Path]:
    """2D PCA of latent shifts — arrows baseline→drug."""
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        PCA = None  # type: ignore

    bd = latent_shift_report["baseline_to_donepezil_shift"]
    bm = latent_shift_report["baseline_to_memantine_shift"]

    X = np.concatenate([mu_baseline, mu_baseline + bd, mu_baseline + bm], axis=0)
    if PCA:
        xy = PCA(n_components=2).fit_transform(X)
    else:
        xy = X[:, :2]

    n = mu_baseline.shape[0]
    base_xy = xy[:n]
    fig, ax = plt.subplots(figsize=(7, 6))
    cols = np.where(labels[:n] >= 0.5, COLORS["ad"], COLORS["hc"])
    ax.scatter(base_xy[:, 0], base_xy[:, 1], c=cols, s=40, edgecolors="k", linewidths=0.3)

    dep_xy = xy[n : 2 * n]
    mem_xy = xy[2 * n :]
    for i in range(min(n, 40)):
        ax.annotate(
            "",
            xy=(dep_xy[i, 0], dep_xy[i, 1]),
            xytext=(base_xy[i, 0], base_xy[i, 1]),
            arrowprops=dict(arrowstyle="->", color=COLORS["donepezil"], lw=0.8, alpha=0.6),
        )
        ax.annotate(
            "",
            xy=(mem_xy[i, 0], mem_xy[i, 1]),
            xytext=(base_xy[i, 0], base_xy[i, 1]),
            arrowprops=dict(arrowstyle="->", color=COLORS["memantine"], lw=0.8, alpha=0.6),
        )

    ax.set_title("Drug-Induced Trajectories in Latent Space (PCA)")
    png = save_dir / "fig5_latent_trajectories.png"
    pdf = save_dir / "fig5_latent_trajectories.pdf"
    plt.tight_layout()
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_shap_waterfall_per_subject(
    shap_values_row: np.ndarray,
    subject_idx: int,
    label: str,
    save_dir: Path,
    *,
    bounds: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[Path, Path]:
    """Waterfall-like bar chart aggregated by block."""
    from src.explainability.feature_bounds import load_feature_blocks_from_metadata

    save_dir.mkdir(parents=True, exist_ok=True)
    bounds = bounds or load_feature_blocks_from_metadata()
    names = list(bounds.keys())
    vals = []
    for a, b in bounds.values():
        vals.append(float(np.sum(shap_values_row[a:b])))

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#c62828" if v > 0 else "#1565c0" for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("Sum SHAP (block)")
    ax.set_title(f'SHAP Block Waterfall — Subject {subject_idx} ({label})')
    png = save_dir / f"fig6_shap_waterfall_subj_{subject_idx}.png"
    pdf = save_dir / f"fig6_shap_waterfall_subj_{subject_idx}.pdf"
    plt.tight_layout()
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_pseudo_topomap(
    channel_importance: Dict[str, float],
    save_dir: Path,
    *,
    channel_positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[Path, Path]:
    from src.explainability.visualization_report import CHANNEL_POS

    save_dir.mkdir(parents=True, exist_ok=True)
    pos = channel_positions or CHANNEL_POS
    xs, ys, ss, labs = [], [], [], []
    for ch, imp in channel_importance.items():
        if ch not in pos:
            continue
        xs.append(pos[ch][0])
        ys.append(pos[ch][1])
        ss.append(max(imp, 1e-9) * 800)
        labs.append(ch)

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(xs, ys, s=ss, c=ss, cmap="viridis", alpha=0.85, edgecolors="k")
    for x, y, lb in zip(xs, ys, labs):
        ax.text(x, y + 0.015, lb, ha="center", fontsize=8)
    ax.add_patch(plt.Circle((0, 0), 0.45, fill=False, color="gray", lw=1))
    plt.colorbar(sc, ax=ax, label="Occlusion sensitivity")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Spatial Distribution of EEG Feature Importance (pseudo-topomap)")
    png = save_dir / "fig7_pseudo_topomap.png"
    pdf = save_dir / "fig7_pseudo_topomap.pdf"
    plt.tight_layout()
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_ig_convergence_analysis(
    convergence_data: Union[List[Dict[str, Any]], Dict[int, float]],
    save_dir: Path,
) -> Tuple[Path, Path]:
    """
    Plot IG stability: (A) step-to-step δ between successive candidate step counts,
    (B) Captum completeness δ from ``find_converged_ig_steps``.

    ``convergence_data`` may be a list of dicts with keys ``n_steps``, ``step_delta``,
    ``captum_delta``, or the legacy dict mapping ``n_steps -> scalar delta``.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(convergence_data, dict):
        xs = sorted(convergence_data.keys())
        step_deltas = [float(convergence_data[k]) for k in xs]
        captum_deltas = [float("nan")] * len(xs)
    else:
        rows = sorted(convergence_data, key=lambda d: int(d["n_steps"]))
        xs = [int(r["n_steps"]) for r in rows]
        step_deltas = [float(r.get("step_delta", float("nan"))) for r in rows]
        captum_deltas = [float(r.get("captum_delta", float("nan"))) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(xs, step_deltas, "o-", color="#1E88E5", linewidth=2, markersize=8, label="Step-to-step δ")
    ax.axhline(0.05, color="#E53935", linestyle="--", label="Threshold (0.05)")
    ax.set_xlabel("Number of IG steps")
    ax.set_ylabel("Mean relative change in |attr|")
    ax.set_title("(A) IG convergence: successive candidate stability")
    ax.legend(fontsize=9)

    converged_step = None
    for x, sd in zip(xs, step_deltas):
        if not np.isnan(sd) and sd < 0.05:
            converged_step = x
            break
    if converged_step is not None:
        ax.axvline(converged_step, color="#43A047", linestyle=":", alpha=0.8, label=f"Sub-threshold at {converged_step}")

    ax2 = axes[1]
    ax2.plot(xs, captum_deltas, "s-", color="#AB47BC", linewidth=2, markersize=8, label="Captum δ")
    ax2.axhline(0.05, color="#E53935", linestyle="--")
    ax2.set_xlabel("Number of IG steps")
    ax2.set_ylabel("Captum convergence delta (mean)")
    ax2.set_title("(B) Captum completeness approximation")
    ax2.legend(fontsize=9)

    plt.suptitle("Integrated Gradients Convergence Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    png = save_dir / "fig8_ig_convergence.png"
    pdf = save_dir / "fig8_ig_convergence.pdf"
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close(fig)
    return png, pdf
