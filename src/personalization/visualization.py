from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def _to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr.mean(axis=1)
    return arr


def plot_patient_landscape(
    latent_mu: np.ndarray,
    ad_mask: np.ndarray,
    cluster_labels: np.ndarray,
    sim_baseline: np.ndarray,
    sim_donepezil: np.ndarray,
    sim_memantine: np.ndarray,
    output_dir: str | Path,
    clustered_ad_global_ids: np.ndarray | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_baseline = _to_2d(sim_baseline)
    sim_donepezil = _to_2d(sim_donepezil)
    sim_memantine = _to_2d(sim_memantine)

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    emb = tsne.fit_transform(latent_mu)
    ad_idx = np.where(ad_mask)[0]
    hc_idx = np.where(~ad_mask)[0]
    ad_emb = emb[ad_idx]
    hc_emb = emb[hc_idx]

    if clustered_ad_global_ids is None:
        clustered_ad_global_ids = ad_idx[: cluster_labels.shape[0]]
    clustered_ad_global_ids = np.asarray(clustered_ad_global_ids, dtype=int)
    clustered_ad_emb = emb[clustered_ad_global_ids]

    cluster_ids = sorted([int(c) for c in np.unique(cluster_labels)])
    palette = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]

    done_alpha = (sim_donepezil - sim_baseline)[:, 418:437].mean(axis=1)
    done_theta = (sim_donepezil - sim_baseline)[:, 399:418].mean(axis=1)
    done_conn = (sim_donepezil - sim_baseline)[:, 475:1330].mean(axis=1)
    mem_alpha = (sim_memantine - sim_baseline)[:, 418:437].mean(axis=1)
    mem_theta = (sim_memantine - sim_baseline)[:, 399:418].mean(axis=1)
    mem_conn = (sim_memantine - sim_baseline)[:, 475:1330].mean(axis=1)

    heat_cols = np.stack([done_alpha, done_theta, done_conn, mem_alpha, mem_theta, mem_conn], axis=1)
    order = np.argsort(cluster_labels)
    sorted_labels = cluster_labels[order]
    sorted_heat = heat_cols[order]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    ax = axes[0]
    for i, c in enumerate(cluster_ids):
        m = cluster_labels == c
        pts = clustered_ad_emb[m]
        ax.scatter(pts[:, 0], pts[:, 1], s=45, alpha=0.85, color=palette[i % len(palette)], label=f"Cluster {c+1} (n={m.sum()})")
        centroid = pts.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], marker="x", s=180, linewidths=2.5, color=palette[i % len(palette)])
    unclustered_ad_ids = np.setdiff1d(ad_idx, clustered_ad_global_ids)
    if unclustered_ad_ids.size:
        pts = emb[unclustered_ad_ids]
        ax.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.35, color="#bbbbbb", label=f"AD unlabeled (n={pts.shape[0]})")
    if hc_emb.size:
        ax.scatter(hc_emb[:, 0], hc_emb[:, 1], marker="^", s=35, color="#888888", alpha=0.75, label=f"HC (n={hc_emb.shape[0]})")
    ax.set_title("A) t-SNE latent landscape")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    im = ax.imshow(sorted_heat, aspect="auto", cmap="coolwarm")
    ax.set_title("B) Drug response heatmap")
    ax.set_xticks(range(6))
    ax.set_xticklabels(["Alpha_D", "Theta_D", "Conn_D", "Alpha_M", "Theta_M", "Conn_M"], rotation=35, ha="right", fontsize=8)
    ax.set_yticks([])
    boundaries = np.where(np.diff(sorted_labels) != 0)[0]
    for b in boundaries:
        ax.axhline(b + 0.5, color="black", linewidth=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    data = [done_alpha[cluster_labels == c] for c in cluster_ids]
    ax.boxplot(data, labels=[f"C{c+1}" for c in cluster_ids], patch_artist=True)
    for patch, c in zip(ax.artists, cluster_ids):
        patch.set_facecolor(palette[c % len(palette)])
    ax.set_title("C) Donepezil alpha response")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Response magnitude")

    plt.tight_layout()
    plt.savefig(output_dir / "fig_patient_landscape.png", dpi=300)
    plt.close(fig)


def plot_cluster_drug_profiles(
    drug_response_results: dict,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_cluster = drug_response_results["by_cluster"]
    clusters = sorted(by_cluster.keys(), key=lambda x: int(x))
    bands = ["alpha", "theta", "delta", "beta", "connectivity"]
    angles = np.linspace(0, 2 * np.pi, len(bands), endpoint=False).tolist()
    angles += angles[:1]
    palette = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"polar": True}, dpi=300)
    for ax, drug in zip(axes, ["donepezil", "memantine"]):
        for i, c in enumerate(clusters):
            vals = [by_cluster[c][drug][b]["mean"] for b in bands]
            vals += vals[:1]
            ax.plot(angles, vals, color=palette[i % len(palette)], linewidth=2, label=f"Cluster {int(c)+1}")
            ax.fill(angles, vals, color=palette[i % len(palette)], alpha=0.15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["Alpha", "Theta", "Delta", "Beta", "Conn"])
        ax.set_title(f"{drug.capitalize()} profile")
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_cluster_drug_profiles.png", dpi=300)
    plt.close(fig)
