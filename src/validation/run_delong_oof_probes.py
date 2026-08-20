"""
Phase C: paired DeLong tests on stored Phase-1 OOF probe score vectors.

READ-ONLY of models/validation/oof_probe_scores_phase1.npz.
Does NOT overwrite locked probe JSONs or the Phase-1 OOF store.
Writes a NEW non-locked artifact under models/validation/.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
IN_NPZ = ROOT / "models" / "validation" / "oof_probe_scores_phase1.npz"
IN_JSON = ROOT / "models" / "validation" / "oof_probe_scores_phase1.json"
OUT_JSON = ROOT / "models" / "validation" / "delong_oof_probe_results.json"
OUT_MD = ROOT / "models" / "validation" / "delong_oof_probe_summary.md"
SCRIPT_PATH = Path(__file__).resolve()

# Display names for manuscript (keys match npz columns).
SCORE_KEYS = {
    "latent_probe": "latent_probe_oof_probability",
    "theta_alpha": "theta_alpha_oof_probability",
    "packed_2185": "packed_2185_oof_probability",
    "best_nested_head": "best_nested_head_oof_probability",
}

# Primary planned pairs (encoding gap storyline).
PRIMARY_PAIRS = [
    ("latent_probe", "theta_alpha"),
    ("latent_probe", "packed_2185"),
    ("best_nested_head", "theta_alpha"),
    ("best_nested_head", "packed_2185"),
]


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks for DeLong structural components (Sun & Xu / DeLong 1988)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_roc_variance(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ground_truth: (n,) binary {0,1}
    predictions: (m, n) scores for m models
    Returns aucs (m,), covariance (m, m)
    """
    order = (-ground_truth).argsort()  # positives first
    label_1_count = int(ground_truth.sum())
    predictions_sorted = predictions[:, order]
    m = predictions_sorted.shape[0]
    n_pos = label_1_count
    n_neg = predictions_sorted.shape[1] - n_pos
    assert n_pos > 0 and n_neg > 0

    # Structural components
    V10 = np.zeros((m, n_pos))
    V01 = np.zeros((m, n_neg))
    for r in range(m):
        pos = predictions_sorted[r, :n_pos]
        neg = predictions_sorted[r, n_pos:]
        # For each positive, fraction of negatives below it (midrank form)
        tx = _compute_midrank(pos)
        ty = _compute_midrank(neg)
        tz = _compute_midrank(predictions_sorted[r])
        V10[r] = (tz[:n_pos] - tx) / n_neg
        V01[r] = 1.0 - (tz[n_pos:] - ty) / n_pos

    aucs = V01.mean(axis=1)
    # Equivalent check: aucs ≈ V10.mean(axis=1)
    s10 = np.cov(V10, bias=True) if n_pos > 1 else np.zeros((m, m))
    s01 = np.cov(V01, bias=True) if n_neg > 1 else np.zeros((m, m))
    cov = s10 / n_pos + s01 / n_neg
    return aucs, cov


def delong_paired_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Dict:
    preds = np.vstack([scores_a, scores_b])
    aucs, cov = delong_roc_variance(y_true, preds)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = (auc_a - auc_b) / np.sqrt(var)
        p = float(2.0 * stats.norm.sf(abs(z)))
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_diff_a_minus_b": float(auc_a - auc_b),
        "z": float(z),
        "p_two_sided": p,
        "var_diff": var,
        "cov_aa": float(cov[0, 0]),
        "cov_bb": float(cov[1, 1]),
        "cov_ab": float(cov[0, 1]),
        "sklearn_auc_a": float(roc_auc_score(y_true, scores_a)),
        "sklearn_auc_b": float(roc_auc_score(y_true, scores_b)),
    }


def main() -> int:
    if not IN_NPZ.exists():
        raise FileNotFoundError(IN_NPZ)

    z = np.load(IN_NPZ, allow_pickle=True)
    y = np.asarray(z["true_label"], dtype=int)
    scores = {name: np.asarray(z[col], dtype=float) for name, col in SCORE_KEYS.items()}

    # Sanity: sklearn OOF AUCs
    sklearn_aucs = {name: float(roc_auc_score(y, s)) for name, s in scores.items()}

    comparisons: List[Dict] = []
    for a, b in combinations(SCORE_KEYS.keys(), 2):
        row = delong_paired_test(y, scores[a], scores[b])
        row["model_a"] = a
        row["model_b"] = b
        row["primary_planned"] = frozenset((a, b)) in {
            frozenset(p) for p in PRIMARY_PAIRS
        }
        comparisons.append(row)

    # Sort: primary first, then by |z|
    comparisons.sort(key=lambda r: (not r["primary_planned"], -abs(r["z"])))

    meta = {}
    if IN_JSON.exists():
        src = json.loads(IN_JSON.read_text(encoding="utf-8"))
        meta = {
            "n": src.get("n"),
            "n_pos_dementia": src.get("n_pos_dementia"),
            "n_neg_normal": src.get("n_neg_normal"),
            "cv_protocol": src.get("cv_protocol"),
            "seed": src.get("seed"),
            "best_nested_head_name": src.get("best_nested_head_name"),
            "source_oof_json": str(IN_JSON),
        }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "phase_c_delong_oof_probes",
        "status": "NEW_NON_LOCKED",
        "method": (
            "DeLong et al. (1988) paired ROC AUC comparison via structural "
            "components / midrank covariance (Sun & Xu form)."
        ),
        "multiplicity": (
            "No multiplicity correction; primary pairs listed explicitly; "
            "remaining pairs are secondary."
        ),
        "input_npz": str(IN_NPZ),
        "script": str(SCRIPT_PATH),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "sklearn_oof_aucs": sklearn_aucs,
        "primary_pairs": [{"a": a, "b": b} for a, b in PRIMARY_PAIRS],
        "comparisons": comparisons,
        "source_meta": meta,
        "claim_boundary": (
            "Non-locked secondary statistics on stored OOF vectors. "
            "Does not change locked CV-mean AUCs (0.579 / 0.675 / 0.699) "
            "or locked best-head OOF (0.593)."
        ),
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase C DeLong on stored OOF probe scores",
        "",
        f"- Status: `{payload['status']}`",
        f"- n = {payload['n']} (pos={payload['n_pos']}, neg={payload['n_neg']})",
        f"- Method: {payload['method']}",
        f"- Multiplicity: {payload['multiplicity']}",
        "",
        "## sklearn OOF AUCs (sanity)",
        "",
    ]
    for k, v in sklearn_aucs.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.extend(["", "## Primary paired comparisons", ""])
    for row in comparisons:
        if not row["primary_planned"]:
            continue
        lines.append(
            f"- **{row['model_a']}** vs **{row['model_b']}**: "
            f"AUC {row['auc_a']:.6f} vs {row['auc_b']:.6f}; "
            f"Δ={row['auc_diff_a_minus_b']:+.6f}; "
            f"z={row['z']:.3f}; p={row['p_two_sided']:.6g}"
        )
    lines.extend(["", "## All pairwise comparisons", ""])
    for row in comparisons:
        tag = "primary" if row["primary_planned"] else "secondary"
        lines.append(
            f"- [{tag}] {row['model_a']} vs {row['model_b']}: "
            f"z={row['z']:.3f}, p={row['p_two_sided']:.6g}"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            payload["claim_boundary"],
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[saved] {OUT_JSON}")
    print(f"[saved] {OUT_MD}")
    for row in comparisons:
        if row["primary_planned"]:
            print(
                f"PRIMARY {row['model_a']} vs {row['model_b']}: "
                f"z={row['z']:.3f} p={row['p_two_sided']:.6g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
