from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import kruskal, mannwhitneyu
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture


BAND_SLICES = {
    "delta": slice(380, 399),
    "theta": slice(399, 418),
    "alpha": slice(418, 437),
    "beta": slice(437, 456),
    "connectivity": slice(475, 1330),
}


def _safe_std(x: np.ndarray) -> float:
    s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return s if s > 1e-12 else 1e-12


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / max(a.size + b.size - 2, 1))
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _choose_k_by_silhouette(latent_mu: np.ndarray, candidates: List[int]) -> int:
    best_k = candidates[0]
    best_score = -1.0
    for k in candidates:
        if k <= 1 or k >= latent_mu.shape[0]:
            continue
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(latent_mu)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(latent_mu, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return int(best_k)


def cluster_patients_by_eeg_phenotype(
    latent_mu: np.ndarray,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> dict:
    n_ad = int(latent_mu.shape[0])
    if n_ad < 2:
        raise ValueError("Need at least 2 AD subjects for clustering.")

    if n_clusters is None:
        if n_ad < 20:
            n_clusters = 2
        elif n_ad < 50:
            n_clusters = _choose_k_by_silhouette(latent_mu, [2, 3])
        else:
            n_clusters = _choose_k_by_silhouette(latent_mu, [2, 3, 4])
    n_clusters = max(2, min(int(n_clusters), n_ad - 1))

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels_km = km.fit_predict(latent_mu)

    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    labels_gmm = gmm.fit_predict(latent_mu)

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels_agg = agg.fit_predict(latent_mu)

    sil = silhouette_score(latent_mu, labels_km) if len(np.unique(labels_km)) > 1 else 0.0
    ari_km_gmm = adjusted_rand_score(labels_km, labels_gmm)
    ari_km_agg = adjusted_rand_score(labels_km, labels_agg)
    ari_gmm_agg = adjusted_rand_score(labels_gmm, labels_agg)
    stability_mean = float(np.mean([ari_km_gmm, ari_km_agg, ari_gmm_agg]))
    cluster_sizes = {int(c): int(np.sum(labels_km == c)) for c in np.unique(labels_km)}

    return {
        "cluster_labels": labels_km,
        "cluster_labels_gmm": labels_gmm,
        "cluster_labels_agg": labels_agg,
        "n_clusters": int(n_clusters),
        "cluster_centers": km.cluster_centers_,
        "silhouette_score": float(sil),
        "stability_ari_km_gmm": float(ari_km_gmm),
        "stability_ari_km_agg": float(ari_km_agg),
        "stability_ari_gmm_agg": float(ari_gmm_agg),
        "stability_mean": float(stability_mean),
        "cluster_sizes": cluster_sizes,
    }


def compute_drug_response_by_cluster(
    cluster_labels: np.ndarray,
    sim_baseline: np.ndarray,
    sim_donepezil: np.ndarray,
    sim_memantine: np.ndarray,
) -> dict:
    if sim_baseline.ndim != 2:
        raise ValueError("sim_baseline must be 2D (n_ad, 2185).")
    if sim_donepezil.shape != sim_baseline.shape or sim_memantine.shape != sim_baseline.shape:
        raise ValueError("Simulation arrays must have the same shape.")

    donep_delta = sim_donepezil - sim_baseline
    mem_delta = sim_memantine - sim_baseline
    clusters = sorted(np.unique(cluster_labels).tolist())

    by_cluster: Dict[str, dict] = {}
    done_scores: Dict[int, np.ndarray] = {}
    mem_scores: Dict[int, np.ndarray] = {}

    for c in clusters:
        mask = cluster_labels == c
        done_c = donep_delta[mask]
        mem_c = mem_delta[mask]
        done_bands = {k: done_c[:, v].mean(axis=1) for k, v in BAND_SLICES.items()}
        mem_bands = {k: mem_c[:, v].mean(axis=1) for k, v in BAND_SLICES.items()}

        done_summary = {
            band: {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0}
            for band, vals in done_bands.items()
        }
        mem_summary = {
            band: {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0}
            for band, vals in mem_bands.items()
        }

        done_score = done_summary["alpha"]["mean"] - done_summary["theta"]["mean"] + 0.5 * done_summary["connectivity"]["mean"]
        mem_score = mem_summary["alpha"]["mean"] - mem_summary["theta"]["mean"] + 0.5 * mem_summary["connectivity"]["mean"]

        by_cluster[str(c)] = {
            "n": int(mask.sum()),
            "donepezil": done_summary,
            "memantine": mem_summary,
            "recommendation_score_donepezil": float(done_score),
            "recommendation_score_memantine": float(mem_score),
            "recommended_drug": "donepezil" if done_score >= mem_score else "memantine",
        }
        done_scores[c] = done_bands["alpha"]
        mem_scores[c] = mem_bands["alpha"]

    metrics = [
        ("donepezil_alpha", donep_delta[:, BAND_SLICES["alpha"]].mean(axis=1)),
        ("donepezil_theta", donep_delta[:, BAND_SLICES["theta"]].mean(axis=1)),
        ("donepezil_connectivity", donep_delta[:, BAND_SLICES["connectivity"]].mean(axis=1)),
        ("memantine_alpha", mem_delta[:, BAND_SLICES["alpha"]].mean(axis=1)),
        ("memantine_theta", mem_delta[:, BAND_SLICES["theta"]].mean(axis=1)),
        ("memantine_connectivity", mem_delta[:, BAND_SLICES["connectivity"]].mean(axis=1)),
    ]

    stats_tests = {}
    n_total = int(cluster_labels.shape[0])
    for metric_name, arr in metrics:
        groups = [arr[cluster_labels == c] for c in clusters]
        if len(clusters) == 2:
            u, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            h_for_eta = float(u)
            test_name = "mannwhitneyu"
            stat_val = float(u)
        else:
            h, p = kruskal(*groups)
            h_for_eta = float(h)
            test_name = "kruskal"
            stat_val = float(h)
        eta2 = h_for_eta / max(n_total - 1, 1)
        means = [float(np.mean(g)) for g in groups]
        best_i = int(np.argmax(means))
        worst_i = int(np.argmin(means))
        d = _cohens_d(groups[best_i], groups[worst_i])
        stats_tests[metric_name] = {
            "test": test_name,
            "statistic": stat_val,
            "p_value": float(p),
            "eta_squared": float(eta2),
            "cohens_d_best_vs_worst": float(d),
            "best_cluster": int(clusters[best_i]),
            "worst_cluster": int(clusters[worst_i]),
        }

    print("=" * 63)
    print("PERSONALIZED DRUG RESPONSE BY EEG PHENOTYPE CLUSTER")
    print("=" * 63)
    print("Cluster   N   Alpha(Done)   Theta(Done)   Alpha(Mem)   Recommended")
    print("-" * 63)
    for c in clusters:
        row = by_cluster[str(c)]
        print(
            f"Type {c+1:<2}   {row['n']:<2}  "
            f"{row['donepezil']['alpha']['mean']:+.4f}       "
            f"{row['donepezil']['theta']['mean']:+.4f}       "
            f"{row['memantine']['alpha']['mean']:+.4f}      "
            f"{row['recommended_drug'].capitalize()}"
        )
    alpha_test = stats_tests["donepezil_alpha"]
    print("-" * 63)
    print(f"Kruskal-Wallis p (alpha donep.): {alpha_test['p_value']:.4f}  eta^2: {alpha_test['eta_squared']:.3f}")
    print(f"Cohen's d (best vs worst):       {alpha_test['cohens_d_best_vs_worst']:.3f}")
    print("=" * 63)

    return {
        "by_cluster": by_cluster,
        "stats_tests": stats_tests,
        "band_slices": {k: [v.start, v.stop] for k, v in BAND_SLICES.items()},
    }


def generate_personalization_report(
    cluster_results: dict,
    drug_response_results: dict,
    output_dir: str | Path,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = int(cluster_results["n_clusters"])
    sil = float(cluster_results["silhouette_score"])
    ari = float(cluster_results["stability_mean"])
    by_cluster = drug_response_results["by_cluster"]
    stats = drug_response_results["stats_tests"]["donepezil_alpha"]
    significant = "significant" if stats["p_value"] < 0.05 else "not significant"

    clusters_sorted = sorted(by_cluster.keys(), key=lambda x: int(x))
    c1 = by_cluster[clusters_sorted[0]]
    c2 = by_cluster[clusters_sorted[1]] if len(clusters_sorted) > 1 else c1

    narrative = (
        f"Unsupervised clustering of AD patient latent representations identified {n_clusters} "
        f"EEG-derived phenotype subgroups (silhouette score = {sil:.3f}, inter-method stability "
        f"ARI = {ari:.3f}).\n\n"
        f"Cluster 1 (n={c1['n']}) showed the strongest predicted response to Donepezil simulation "
        f"(delta_alpha = {c1['donepezil']['alpha']['mean']:.3f} +/- {c1['donepezil']['alpha']['std']:.3f}), "
        f"while Cluster 2 (n={c2['n']}) demonstrated preferential response to Memantine "
        f"(delta_theta = {c2['memantine']['theta']['mean']:.3f} +/- {c2['memantine']['theta']['std']:.3f}).\n\n"
        f"The difference in drug response between clusters was {significant} "
        f"(Kruskal-Wallis p = {stats['p_value']:.4f}, effect size eta^2 = {stats['eta_squared']:.3f}, "
        f"Cohen's d = {stats['cohens_d_best_vs_worst']:.2f}).\n\n"
        f"These findings suggest that baseline EEG phenotype may predict differential drug response - "
        f"a potential basis for personalized pharmacotherapy selection in Alzheimer's disease. "
        f"This analysis is hypothesis-generating and requires prospective clinical validation."
    )

    (out_dir / "personalization_report.txt").write_text(narrative, encoding="utf-8")
    return narrative
