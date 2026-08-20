"""
Prior-only direction baseline (Plan-to-8.3 B2).

Constructs a 2185-D signed effect from literature EEG priors
(`get_eeg_effect_prior`) and scores the same 10-block direction metric
against the constrained training mean effect vector
(`models/simulations_constrained_full`).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.drugs.pharmacological_embedding import get_eeg_effect_prior
from src.validation.stats_ci import fisher_z_ci

BAND_SLICES = {
    "delta": slice(380, 399),
    "theta": slice(399, 418),
    "alpha": slice(418, 437),
    "beta": slice(437, 456),
    "connectivity": slice(475, 1330),
}


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_constrained_training_effects(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    cohort_dir = root / "models" / "simulations_constrained_full"
    base = np.load(cohort_dir / "simulated_baseline.npy").mean(1)
    don = np.load(cohort_dir / "simulated_donepezil.npy").mean(1)
    mem = np.load(cohort_dir / "simulated_memantine.npy").mean(1)
    return (don - base).mean(0), (mem - base).mean(0)


def prior_effect_vector(drug: str, feat_dim: int = 2185) -> np.ndarray:
    prior = get_eeg_effect_prior(drug)
    vec = np.zeros(feat_dim, dtype=np.float64)
    for name, sl in BAND_SLICES.items():
        vec[sl] = float(prior[name])
    return vec


def direction_agree(a: np.ndarray, b: np.ndarray) -> Tuple[int, int]:
    ok = tot = 0
    for sl in BAND_SLICES.values():
        tot += 1
        ma, mb = float(np.mean(a[sl])), float(np.mean(b[sl]))
        if np.sign(ma) == np.sign(mb) or (abs(ma) < 1e-8 and abs(mb) < 1e-8):
            ok += 1
    return ok, tot


def score_drug(prior_vec: np.ndarray, cohort_vec: np.ndarray) -> Dict:
    a, t = direction_agree(prior_vec, cohort_vec)
    r = float(np.corrcoef(prior_vec, cohort_vec)[0, 1])
    return {
        "direction_agreement": f"{a}/{t}",
        "agree": a,
        "total": t,
        "effect_corr": r,
        "effect_corr_ci": fisher_z_ci(r, n=int(prior_vec.shape[0])),
        "band_means_prior": {k: float(np.mean(prior_vec[sl])) for k, sl in BAND_SLICES.items()},
        "band_means_cohort": {k: float(np.mean(cohort_vec[sl])) for k, sl in BAND_SLICES.items()},
    }


def main() -> int:
    root = _root()
    cohort_don, cohort_mem = load_constrained_training_effects(root)
    prior_don = prior_effect_vector("donepezil", feat_dim=cohort_don.shape[0])
    prior_mem = prior_effect_vector("memantine", feat_dim=cohort_mem.shape[0])

    don = score_drug(prior_don, cohort_don)
    mem = score_drug(prior_mem, cohort_mem)
    total_a = don["agree"] + mem["agree"]
    total_t = don["total"] + mem["total"]
    r_mean = float(np.nanmean([don["effect_corr"], mem["effect_corr"]]))

    out = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": (
            "Prior-only signed band/connectivity effect vs constrained "
            "training mean effect (same 10-block direction metric as external twins)"
        ),
        "reference_signature": "models/simulations_constrained_full",
        "prior_source": "get_eeg_effect_prior / DRUG_PHARMACOLOGY eeg_effects",
        "donepezil": don,
        "memantine": mem,
        "direction_agreement_total": f"{total_a}/{total_t}",
        "effect_magnitude_correlation_mean": r_mean,
        "effect_magnitude_correlation_mean_ci": fisher_z_ci(
            r_mean, n=int(cohort_don.shape[0])
        ),
        "interpretation_note": (
            "This baseline does not use a CVAE. Perfect or near-perfect agreement "
            "means the constrained training signature largely recovers the "
            "literature prior signs; imperfect agreement means the twin learned "
            "additional structure beyond the prior constants."
        ),
    }

    out_json = root / "models" / "validation" / "prior_only_direction.json"
    out_md = root / "models" / "validation" / "prior_only_direction.md"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# Prior-only direction baseline",
        "",
        f"- Total direction agreement: **{out['direction_agreement_total']}**",
        f"- Mean effect correlation r = {r_mean:.3f}",
        f"- Donepezil: {don['direction_agreement']} (r={don['effect_corr']:.3f})",
        f"- Memantine: {mem['direction_agreement']} (r={mem['effect_corr']:.3f})",
        "",
        out["interpretation_note"],
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: out[k] for k in (
        "direction_agreement_total",
        "effect_magnitude_correlation_mean",
        "donepezil",
        "memantine",
    )}, indent=2))
    print(f"[wrote] {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
