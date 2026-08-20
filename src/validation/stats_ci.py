"""Confidence intervals for manuscript stats (Fisher-z for r, subject bootstrap, etc.)."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> Dict[str, float]:
    """
    95% (default) CI for Pearson r via Fisher z-transform.

    Note: for cohort effect-vector correlations prefer ``subject_bootstrap_effect_metrics``;
    feature length is not an independent-observation count.
    """
    r = float(r)
    n = int(n)
    if n < 4 or not math.isfinite(r) or abs(r) >= 1.0:
        return {
            "r": r,
            "n": n,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "method": "fisher_z",
            "alpha": alpha,
            "note": "n<4 or |r|>=1; CI undefined",
        }
    r_c = max(min(r, 1.0 - 1e-12), -1.0 + 1e-12)
    z = math.atanh(r_c)
    se = 1.0 / math.sqrt(n - 3)
    from scipy.stats import norm

    zcrit = float(norm.ppf(1.0 - alpha / 2.0))
    lo = math.tanh(z - zcrit * se)
    hi = math.tanh(z + zcrit * se)
    return {
        "r": r,
        "n": n,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "method": "fisher_z",
        "alpha": alpha,
    }


def format_r_ci(r: float, n: int, digits: int = 3) -> str:
    ci = fisher_z_ci(r, n)
    if not math.isfinite(ci["ci_low"]):
        return f"{r:.{digits}f} (CI undefined)"
    return (
        f"{r:.{digits}f} (95% CI {ci['ci_low']:.{digits}f} to {ci['ci_high']:.{digits}f})"
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-15 or nb < 1e-15:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def subject_bootstrap_effect_metrics(
    don_effects: np.ndarray,
    mem_effects: np.ndarray,
    sig_don: np.ndarray,
    sig_mem: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict:
    """
    Subject-level bootstrap CIs for mean Donepezil/Memantine effect-vector
    Pearson r and cosine similarity vs a fixed training signature.

    ``don_effects`` / ``mem_effects`` shape: (n_subjects, n_features).
    """
    don = np.asarray(don_effects, dtype=float)
    mem = np.asarray(mem_effects, dtype=float)
    if don.ndim != 2 or mem.ndim != 2 or don.shape != mem.shape:
        raise ValueError("don_effects and mem_effects must share shape (n_subj, n_feat)")
    n_subj = int(don.shape[0])
    rng = np.random.default_rng(int(seed))

    def _point(idx: np.ndarray) -> Dict[str, float]:
        ed = don[idx].mean(0)
        em = mem[idx].mean(0)
        r_d = _pearson(ed, sig_don)
        r_m = _pearson(em, sig_mem)
        c_d = _cosine(ed, sig_don)
        c_m = _cosine(em, sig_mem)
        return {
            "donepezil_effect_corr": r_d,
            "memantine_effect_corr": r_m,
            "effect_magnitude_correlation": float(np.nanmean([r_d, r_m])),
            "donepezil_cosine": c_d,
            "memantine_cosine": c_m,
            "cosine_mean": float(np.nanmean([c_d, c_m])),
        }

    point = _point(np.arange(n_subj))
    keys = list(point.keys())
    boots = {k: np.empty(n_boot, dtype=float) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, n_subj, size=n_subj)
        row = _point(idx)
        for k in keys:
            boots[k][b] = row[k]

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    out: Dict = {
        "n_subjects": n_subj,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
        "method": "subject_bootstrap_percentile",
        "point": point,
        "ci": {},
    }
    for k in keys:
        arr = boots[k]
        out["ci"][k] = {
            "ci_low": float(np.nanpercentile(arr, lo_q)),
            "ci_high": float(np.nanpercentile(arr, hi_q)),
            "boot_mean": float(np.nanmean(arr)),
            "boot_std": float(np.nanstd(arr, ddof=1)),
        }
    r = point["effect_magnitude_correlation"]
    ci = out["ci"]["effect_magnitude_correlation"]
    out["effect_magnitude_correlation_formatted"] = (
        f"{r:.3f} (subject-bootstrap 95% percentile interval "
        f"{ci['ci_low']:.3f} to {ci['ci_high']:.3f}; "
        f"boot mean {ci['boot_mean']:.3f}; n_subj={n_subj}, B={n_boot})"
    )
    out["reporting_note"] = (
        "Full-sample point estimate is reported with the percentile interval of "
        "subject-bootstrap replicates of the same estimator. The bootstrap "
        "distribution can be slightly downward-biased relative to the full-sample "
        "point; the interval reflects subject-sampling variability, not feature-length Fisher z."
    )
    return out
