"""
CVAE vs TVB directional validation.

Compares simulated drug-effect directions from the trained CVAE against
TVB (or synthetic-fallback) ground-truth feature changes.

Author: Research Team
Date: 2026
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

from src.models.config import DATA_ROOT, PROJECT_ROOT, SIMULATIONS_DIR
from src.tvb_validation.tvb_simulator import (
    ALPHA_SLICE,
    BETA_SLICE,
    CONNECTIVITY_SLICE,
    DELTA_SLICE,
    EEG_FLAT_SIZE,
    THETA_SLICE,
    GROUND_TRUTH,
    band_block_mean,
    connectivity_block_mean,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Colorblind-friendly palette (Okabe–Ito)
COLORS = {
    "real": "#0072B2",
    "tvb_healthy": "#D55E00",
    "tvb_ad": "#009E73",
    "tvb_drug": "#E69F00",
    "match": "#009E73",
    "mismatch": "#D55E00",
    "cvae": "#0072B2",
    "tvb": "#CC79A7",
}

BAND_EXTRACTORS = {
    "Alpha": lambda a: band_block_mean(a, "alpha"),
    "Theta": lambda a: band_block_mean(a, "theta"),
    "Delta": lambda a: band_block_mean(a, "delta"),
    "Beta": lambda a: band_block_mean(a, "beta"),
    "Conn.": connectivity_block_mean,
}

# Kept for figure A axis labeling / slice-based bars (same reshape extractors)
BAND_SLICES = {
    "Alpha": ALPHA_SLICE,
    "Theta": THETA_SLICE,
    "Delta": DELTA_SLICE,
    "Beta": BETA_SLICE,
    "Conn.": CONNECTIVITY_SLICE,
}

# Literature-expected directions for the printed table (TVB column labels)
EXPECTED_DIRECTIONS = {
    ("Alpha", "Donepezil"): (+1, "INCREASE (+)"),
    ("Theta", "Donepezil"): (-1, "DECREASE (-)"),
    ("Delta", "Donepezil"): (-1, "DECREASE (-)"),
    ("Beta", "Donepezil"): (+1, "INCREASE (+)"),
    ("Conn.", "Donepezil"): (+1, "INCREASE (+)"),
    ("Alpha", "Memantine"): (+1, "INCREASE (+)"),
    ("Theta", "Memantine"): (-1, "DECREASE (-)"),
    ("Delta", "Memantine"): (-1, "DECREASE (-)"),
    ("Beta", "Memantine"): (+1, "INCREASE (+)"),
    ("Conn.", "Memantine"): (+1, "INCREASE (+)"),
}


def _resolve_dir(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _load_cvae_sims(sim_dir: Path) -> Dict[str, np.ndarray]:
    """Load CVAE simulations; average over sample dim if 3D (n, samples, 2185)."""
    out = {}
    for name in ("baseline", "donepezil", "memantine"):
        path = sim_dir / f"simulated_{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing CVAE simulation: {path}")
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr.mean(axis=1)
        elif arr.ndim != 2:
            raise ValueError(f"Unexpected shape for {path}: {arr.shape}")
        if arr.shape[-1] != EEG_FLAT_SIZE:
            raise ValueError(f"Expected last dim {EEG_FLAT_SIZE}, got {arr.shape}")
        out[name] = arr.astype(np.float64)
    return out


def _detect_prefix(tvb_dir: Path) -> str:
    """Prefer tvb_ features; fall back to synthetic_fallback_."""
    if (tvb_dir / "tvb_alzheimer_moderate_features.npy").exists():
        return "tvb_"
    if (tvb_dir / "synthetic_fallback_alzheimer_moderate_features.npy").exists():
        return "synthetic_fallback_"
    # Partial: any matching
    for pref in ("tvb_", "synthetic_fallback_"):
        if list(tvb_dir.glob(f"{pref}*_features.npy")):
            return pref
    raise FileNotFoundError(
        f"No TVB/fallback feature files found in {tvb_dir}. Run stage 1 first."
    )


def _load_tvb_features(tvb_dir: Path, prefix: str) -> Dict[str, np.ndarray]:
    mapping = {
        "baseline": f"{prefix}alzheimer_moderate_features.npy",
        "donepezil": f"{prefix}alzheimer_donepezil_features.npy",
        "memantine": f"{prefix}alzheimer_memantine_features.npy",
        "healthy": f"{prefix}healthy_control_features.npy",
    }
    out = {}
    for key, fname in mapping.items():
        path = tvb_dir / fname
        if not path.exists():
            if key == "healthy":
                continue
            raise FileNotFoundError(f"Missing TVB features: {path}")
        out[key] = np.load(path).astype(np.float64)
    return out


def _mean_band(arr: np.ndarray, sl: slice) -> float:
    return float(np.nanmean(arr[:, sl]))


def _sign_label(change: float) -> Tuple[int, str]:
    if change > 0:
        return +1, "INCREASE (+)"
    if change < 0:
        return -1, "DECREASE (-)"
    return 0, "NONE (0)"


def compare_cvae_to_tvb(
    cvae_simulations_dir: str | Path = "models/simulations",
    tvb_features_dir: str | Path = "data/tvb_validation",
    output_dir: str | Path = "data/tvb_validation",
) -> Dict[str, Any]:
    """
    Compare directional changes: CVAE vs TVB (or synthetic fallback).
    """
    sim_dir = _resolve_dir(cvae_simulations_dir)
    tvb_dir = _resolve_dir(tvb_features_dir)
    out_dir = _resolve_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cvae = _load_cvae_sims(sim_dir)
    prefix = _detect_prefix(tvb_dir)
    tvb = _load_tvb_features(tvb_dir, prefix)

    checks: List[Dict[str, Any]] = []
    cvae_effects: List[float] = []
    tvb_effects: List[float] = []

    rows_print: List[Tuple[str, str, str, str]] = []

    for drug_name, drug_key in (("Donepezil", "donepezil"), ("Memantine", "memantine")):
        for band_name, extractor in BAND_EXTRACTORS.items():
            cvae_change = extractor(cvae[drug_key]) - extractor(cvae["baseline"])
            tvb_change = extractor(tvb[drug_key]) - extractor(tvb["baseline"])

            cvae_dir, cvae_lab = _sign_label(cvae_change)
            tvb_dir, tvb_lab = _sign_label(tvb_change)

            # Match CVAE direction to TVB observed direction
            match = cvae_dir == tvb_dir
            if cvae_dir == 0 or tvb_dir == 0:
                match = cvae_dir == tvb_dir

            expected_sign, expected_lab = EXPECTED_DIRECTIONS[(band_name, drug_name)]

            checks.append(
                {
                    "feature": f"{band_name} ({drug_name})",
                    "band": band_name,
                    "drug": drug_name,
                    "cvae_change": cvae_change,
                    "tvb_change": tvb_change,
                    "cvae_direction": cvae_lab,
                    "tvb_direction": tvb_lab,
                    "expected_literature": expected_lab,
                    "match": bool(match),
                    "tvb_matches_literature": bool(tvb_dir == expected_sign),
                }
            )
            cvae_effects.append(cvae_change)
            tvb_effects.append(tvb_change)
            # TVB column = observed TVB sign; Match = CVAE vs TVB
            rows_print.append(
                (
                    f"{band_name} ({drug_name})",
                    tvb_lab,
                    cvae_lab,
                    "YES" if match else "NO",
                )
            )

    n_match = sum(1 for c in checks if c["match"])
    n_total = len(checks)
    accuracy = 100.0 * n_match / n_total if n_total else 0.0

    if len(cvae_effects) >= 2 and np.std(cvae_effects) > 0 and np.std(tvb_effects) > 0:
        r, p = stats.pearsonr(tvb_effects, cvae_effects)
    else:
        r, p = float("nan"), float("nan")

    # Interpretation
    if n_match == 10:
        interpretation = "Perfect — publish in Nature"
    elif n_match >= 7:
        interpretation = "Strong — publishable in any good journal"
    elif n_match >= 5:
        interpretation = "Moderate — publishable with caveats"
    else:
        interpretation = "Weak — model needs retraining or constraints"

    # Print exact table
    lines = [
        "=" * 67,
        "CVAE vs TVB DIRECTIONAL VALIDATION",
        "=" * 67,
        f"{'Feature':<20} {'TVB Direction':<16} {'CVAE Direction':<16} {'Match':<6}",
        "-" * 67,
    ]
    for feat, tvb_d, cvae_d, m in rows_print:
        # Recompute CVAE label from checks for accuracy
        lines.append(f"{feat:<20} {tvb_d:<16} {cvae_d:<16} {m:<6}")
    lines += [
        "-" * 67,
        f"Overall directional accuracy: {n_match}/{n_total} ({accuracy:.1f}%)",
        f"Effect size correlation (Pearson r): {r:.3f} (p = {p:.4f})",
        "=" * 67,
        "",
        "Interpretation guide:",
        "  10/10: Perfect — publish in Nature",
        "  7-9/10: Strong — publishable in any good journal",
        "  5-6/10: Moderate — publishable with caveats",
        "  <5/10: Weak — model needs retraining or constraints",
        f"",
        f"This run: {n_match}/{n_total} -> {interpretation}",
        f"Feature source prefix: {prefix}",
    ]
    table = "\n".join(lines)
    print(table)

    results = {
        "prefix": prefix,
        "n_match": n_match,
        "n_total": n_total,
        "accuracy_pct": accuracy,
        "pearson_r": None if np.isnan(r) else float(r),
        "pearson_p": None if np.isnan(p) else float(p),
        "interpretation": interpretation,
        "checks": checks,
        "cvae_effects": cvae_effects,
        "tvb_effects": tvb_effects,
        "ground_truth_reference": GROUND_TRUTH,
        "table_text": table,
    }

    out_path = out_dir / "tvb_validation_results.json"
    serializable = {
        k: v
        for k, v in results.items()
        if k not in ("cvae_effects", "tvb_effects")
    }
    serializable["cvae_effects"] = [float(x) for x in cvae_effects]
    serializable["tvb_effects"] = [float(x) for x in tvb_effects]
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return results


def _flatten_real_eeg_features(features_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load AD/HC component arrays and flatten to (n, 2185)."""
    from src.tvb_validation.tvb_simulator import _flatten_features

    rows = []
    labels = []
    for group, lab in (("AD", 1), ("HC", 0)):
        psd_p = features_dir / f"{group}_psd.npy"
        bp_p = features_dir / f"{group}_band_powers.npy"
        coh_p = features_dir / f"{group}_coherence.npy"
        plv_p = features_dir / f"{group}_plv.npy"
        if not all(p.exists() for p in (psd_p, bp_p, coh_p, plv_p)):
            continue
        psd = np.load(psd_p)
        bp = np.load(bp_p)
        coh = np.load(coh_p)
        plv = np.load(plv_p)
        n = psd.shape[0]
        for i in range(n):
            rows.append(_flatten_features(psd[i], bp[i], coh[i], plv[i]))
            labels.append(lab)
    if not rows:
        raise FileNotFoundError(f"No real EEG feature components in {features_dir}")
    return np.stack(rows, axis=0), np.asarray(labels)


def _rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None) -> float:
    """Unbiased MMD^2 with RBF kernel (on PCA-reduced or subsampled features)."""
    # Subsample for tractability
    rng = np.random.default_rng(0)
    n = min(len(x), len(y), 200)
    xi = x[rng.choice(len(x), n, replace=False)]
    yi = y[rng.choice(len(y), n, replace=False)]
    # Use first 100 dims after z-score for speed
    both = np.vstack([xi, yi])
    mu, sd = both.mean(0), both.std(0) + 1e-8
    xi = (xi - mu) / sd
    yi = (yi - mu) / sd
    d = min(100, xi.shape[1])
    xi, yi = xi[:, :d], yi[:, :d]
    if gamma is None:
        # median heuristic on pooled pairwise distances (sample)
        pooled = np.vstack([xi, yi])
        idx = rng.choice(len(pooled), min(80, len(pooled)), replace=False)
        diffs = pooled[idx, None, :] - pooled[None, idx, :]
        dists = np.sqrt((diffs ** 2).sum(-1))
        med = np.median(dists[dists > 0])
        gamma = 1.0 / (2.0 * med ** 2 + 1e-8)

    def k(a, b):
        sq = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * sq)

    kxx = k(xi, xi)
    kyy = k(yi, yi)
    kxy = k(xi, yi)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    mmd2 = kxx.sum() / (n * (n - 1)) + kyy.sum() / (n * (n - 1)) - 2.0 * kxy.mean()
    return float(max(mmd2, 0.0))


def compute_feature_space_overlap(
    tvb_features_dir: str | Path = "data/tvb_validation",
    real_features_dir: str | Path = "data/eeg_features",
    output_dir: str | Path = "data/tvb_validation",
) -> Dict[str, Any]:
    """
    PCA overlap, MMD, and per-feature KS tests between real and TVB features.
    """
    tvb_dir = _resolve_dir(tvb_features_dir)
    real_dir = _resolve_dir(real_features_dir)
    out_dir = _resolve_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = _detect_prefix(tvb_dir)
    real_X, _ = _flatten_real_eeg_features(real_dir)

    def _load(cond: str) -> Optional[np.ndarray]:
        p = tvb_dir / f"{prefix}{cond}_features.npy"
        return np.load(p) if p.exists() else None

    tvb_h = _load("healthy_control")
    tvb_ad = _load("alzheimer_moderate")
    tvb_don = _load("alzheimer_donepezil")
    tvb_mem = _load("alzheimer_memantine")
    tvb_parts = [x for x in (tvb_h, tvb_ad, tvb_don, tvb_mem) if x is not None]
    if not tvb_parts:
        raise FileNotFoundError("No TVB feature arrays for overlap analysis.")
    tvb_all = np.vstack(tvb_parts)

    # PCA fit on real (sklearn if available, else SVD)
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        real_2d = pca.fit_transform(real_X)
        var_ratio = pca.explained_variance_ratio_

        def _transform(x: np.ndarray) -> np.ndarray:
            return pca.transform(x)

    except ImportError:
        mu = real_X.mean(axis=0)
        xc = real_X - mu
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        components = vt[:2]
        total_var = (s ** 2).sum()
        var_ratio = (s[:2] ** 2) / total_var
        real_2d = xc @ components.T

        def _transform(x: np.ndarray) -> np.ndarray:
            return (x - mu) @ components.T

    tvb_h_2d = _transform(tvb_h) if tvb_h is not None else None
    tvb_ad_2d = _transform(tvb_ad) if tvb_ad is not None else None
    drug_stack = np.vstack([x for x in (tvb_don, tvb_mem) if x is not None])
    tvb_drug_2d = _transform(drug_stack)

    # MMD
    mmd = _rbf_mmd(real_X, tvb_all)

    # KS tests per feature (subsample features if needed)
    n_feat = real_X.shape[1]
    # Compare real vs TVB-AD (most relevant)
    tvb_ref = tvb_ad if tvb_ad is not None else tvb_all
    n_not_sig = 0
    # Sample up to 500 features for reporting speed
    rng = np.random.default_rng(0)
    feat_idx = rng.choice(n_feat, size=min(500, n_feat), replace=False)
    for fi in tqdm(feat_idx, desc="KS tests"):
        a = real_X[:, fi]
        b = tvb_ref[:, fi]
        if np.std(a) < 1e-15 and np.std(b) < 1e-15:
            n_not_sig += 1
            continue
        _, pval = stats.ks_2samp(a, b)
        if pval > 0.05:
            n_not_sig += 1
    frac_not_sig = n_not_sig / len(feat_idx)

    limitation = None
    if mmd > 0.5 or frac_not_sig < 0.2:
        limitation = (
            "TVB features occupy an adjacent but distinct region of feature "
            "space — validation is directional only, not distributional."
        )
        print(f"[LIMITATION] {limitation}")

    # PCA figure (also saved as tvb_pca_overlap.png)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(real_2d[:, 0], real_2d[:, 1], c=COLORS["real"], s=28, alpha=0.65, label="Real EEG", edgecolors="none")
    if tvb_h_2d is not None:
        ax.scatter(tvb_h_2d[:, 0], tvb_h_2d[:, 1], c=COLORS["tvb_healthy"], s=40, alpha=0.8, label="TVB healthy", marker="^")
    if tvb_ad_2d is not None:
        ax.scatter(tvb_ad_2d[:, 0], tvb_ad_2d[:, 1], c=COLORS["tvb_ad"], s=40, alpha=0.8, label="TVB AD", marker="s")
    ax.scatter(tvb_drug_2d[:, 0], tvb_drug_2d[:, 1], c=COLORS["tvb_drug"], s=40, alpha=0.8, label="TVB drug", marker="D")
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)")
    ax.set_title("Feature-space overlap: Real EEG vs TVB")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    pca_path = out_dir / "tvb_pca_overlap.png"
    fig.savefig(pca_path, dpi=300)
    fig.savefig(out_dir / "fig_tvb_pca_overlap.png", dpi=300)
    plt.close(fig)

    overlap = {
        "prefix": prefix,
        "mmd": mmd,
        "ks_fraction_not_significant": frac_not_sig,
        "ks_n_features_tested": int(len(feat_idx)),
        "pca_variance_explained": np.asarray(var_ratio).tolist(),
        "limitation": limitation,
        "pca_figure": str(pca_path),
    }
    (out_dir / "tvb_feature_overlap.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    print(f"MMD^2={mmd:.4f}; KS not-sig fraction={frac_not_sig:.3f}")
    print(f"Saved: {pca_path}")
    return overlap


def generate_validation_figures(
    tvb_features_dir: str | Path = "data/tvb_validation",
    output_dir: str | Path = "data/tvb_validation",
    comparison_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Generate four publication-quality validation figures at 300 DPI."""
    tvb_dir = _resolve_dir(tvb_features_dir)
    out_dir = _resolve_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _detect_prefix(tvb_dir)

    def load(cond: str) -> np.ndarray:
        return np.load(tvb_dir / f"{prefix}{cond}_features.npy")

    healthy = load("healthy_control")
    ad = load("alzheimer_moderate")
    don = load("alzheimer_donepezil")
    mem = load("alzheimer_memantine")

    paths: Dict[str, Path] = {}

    # ----- Figure A: band power comparison -----
    bands = ["Delta", "Theta", "Alpha", "Beta"]
    extractors = [
        lambda a: band_block_mean(a, "delta"),
        lambda a: band_block_mean(a, "theta"),
        lambda a: band_block_mean(a, "alpha"),
        lambda a: band_block_mean(a, "beta"),
    ]
    groups = [
        ("Healthy", healthy, "#0072B2"),
        ("AD", ad, "#D55E00"),
        ("AD+Donepezil", don, "#009E73"),
        ("AD+Memantine", mem, "#E69F00"),
    ]
    means = np.zeros((len(bands), len(groups)))
    stds = np.zeros_like(means)
    for bi, band_key in enumerate(["delta", "theta", "alpha", "beta"]):
        for gi, (_, arr, _) in enumerate(groups):
            bp = arr[:, 380:475].reshape(arr.shape[0], 19, 5)
            vals = bp[:, :, bi].mean(axis=1)
            means[bi, gi] = vals.mean()
            stds[bi, gi] = vals.std()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(bands))
    width = 0.2
    for gi, (label, _, color) in enumerate(groups):
        ax.bar(
            x + (gi - 1.5) * width,
            means[:, gi],
            width,
            yerr=stds[:, gi],
            label=label,
            color=color,
            capsize=3,
            edgecolor="none",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("Mean feature value (comparison slice)")
    ax.set_title("TVB band-power comparison across conditions")
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p_a = out_dir / "fig_tvb_band_comparison.png"
    fig.savefig(p_a, dpi=300)
    plt.close(fig)
    paths["fig_a"] = p_a

    # ----- Figure B: direction match matrix -----
    if comparison_results is None:
        results_path = out_dir / "tvb_validation_results.json"
        if results_path.exists():
            comparison_results = json.loads(results_path.read_text(encoding="utf-8"))
        else:
            comparison_results = compare_cvae_to_tvb(
                tvb_features_dir=tvb_dir, output_dir=out_dir
            )

    checks = comparison_results["checks"]
    labels = [c["feature"] for c in checks]
    match_flags = [c["match"] for c in checks]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (lab, ok) in enumerate(zip(labels, match_flags)):
        color = COLORS["match"] if ok else COLORS["mismatch"]
        ax.barh(i, 1.0, color=color, edgecolor="white", height=0.85)
        ax.text(0.5, i, "YES" if ok else "NO", ha="center", va="center", color="white", fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("CVAE vs TVB direction match")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    p_b = out_dir / "fig_tvb_direction_match.png"
    fig.savefig(p_b, dpi=300)
    plt.close(fig)
    paths["fig_b"] = p_b

    # ----- Figure C: PCA (reuse / regenerate) -----
    p_c = out_dir / "fig_tvb_pca_overlap.png"
    if not p_c.exists():
        compute_feature_space_overlap(tvb_features_dir=tvb_dir, output_dir=out_dir)
    paths["fig_c"] = p_c

    # ----- Figure D: effect-size scatter -----
    tvb_eff = np.asarray(comparison_results["tvb_effects"], dtype=float)
    cvae_eff = np.asarray(comparison_results["cvae_effects"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(tvb_eff, cvae_eff, c=COLORS["tvb"], s=70, zorder=3, edgecolors="k", linewidths=0.4)
    if np.std(tvb_eff) > 0:
        coef = np.polyfit(tvb_eff, cvae_eff, 1)
        xs = np.linspace(tvb_eff.min(), tvb_eff.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), color=COLORS["real"], lw=2, label="Linear fit")
    r = comparison_results.get("pearson_r")
    p = comparison_results.get("pearson_p")
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("TVB effect size (drug − AD baseline)")
    ax.set_ylabel("CVAE effect size (drug − baseline)")
    ax.set_title(f"Effect-size agreement (r = {r if r is not None else float('nan'):.3f})")
    if r is not None:
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {r:.3f}\np = {p:.4f}" if p is not None else f"r = {r:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p_d = out_dir / "fig_tvb_effect_scatter.png"
    fig.savefig(p_d, dpi=300)
    plt.close(fig)
    paths["fig_d"] = p_d

    print("Figures saved:")
    for k, pth in paths.items():
        print(f"  {k}: {pth}")
    return paths
