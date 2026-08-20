"""
Five-fold ensemble external direction scoring from existing fold_sims.

For each cohort subject, average Donepezil/Memantine effect vectors across
folds 0..4, then score once against the fixed constrained training signature.

Writes ONLY under models/validation/fold_ensemble/.
Does NOT re-simulate, retrain, or overwrite fold_sims / locked reports.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.validation.direction_metrics import (
    load_constrained_training_effects,
    score_effects_vs_signature,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_subject_effects(sim_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for p in sorted(sim_dir.glob("*_sims.npz")):
        sid = p.name.replace("_sims.npz", "")
        z = np.load(p)
        don = np.asarray(z["donepezil"], dtype=np.float64) - np.asarray(
            z["baseline"], dtype=np.float64
        )
        mem = np.asarray(z["memantine"], dtype=np.float64) - np.asarray(
            z["baseline"], dtype=np.float64
        )
        out[sid] = (don, mem)
    return out


def ensemble_cohort(
    fold_sims_root: Path,
    cohort: str,
    fold_ids: List[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], Dict]:
    per_fold: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
    for k in fold_ids:
        d = fold_sims_root / f"fold_{k}" / cohort
        if not d.is_dir():
            raise FileNotFoundError(d)
        per_fold.append(_load_subject_effects(d))

    common = set(per_fold[0].keys())
    for m in per_fold[1:]:
        common &= set(m.keys())
    sids = sorted(common)
    if not sids:
        raise RuntimeError(f"No common subject IDs for cohort={cohort}")

    don_list: List[np.ndarray] = []
    mem_list: List[np.ndarray] = []
    for sid in sids:
        dons = [per_fold[i][sid][0] for i in range(len(fold_ids))]
        mems = [per_fold[i][sid][1] for i in range(len(fold_ids))]
        don_list.append(np.mean(np.stack(dons, axis=0), axis=0))
        mem_list.append(np.mean(np.stack(mems, axis=0), axis=0))

    meta = {
        "n_subjects_common": len(sids),
        "n_per_fold": [len(m) for m in per_fold],
        "n_folds": len(fold_ids),
        "folds": fold_ids,
        "subject_ids": sids,
    }
    return don_list, mem_list, sids, meta


def main() -> int:
    root = _root()
    out_dir = root / "models" / "validation" / "fold_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_ids = [0, 1, 2, 3, 4]
    cohorts = ["tuh", "osf", "padic"]
    fold_sims_root = root / "models" / "validation" / "fold_sims"
    sig = load_constrained_training_effects(root)

    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "constrained_fivefold_ensemble_external_direction",
        "method": (
            "Per subject, average Donepezil and Memantine effect vectors "
            "(drug - baseline) across folds 0..4, then score cohort-mean "
            "effects vs fixed constrained training signature "
            "(same score_effects_vs_signature as per-fold sensitivity)."
        ),
        "signature_source": sig["source"],
        "fold_sims_root": str(fold_sims_root),
        "folds": fold_ids,
        "outputs_restricted_to": str(out_dir),
        "cohorts": {},
    }

    for name in cohorts:
        print(f"=== ensemble {name} ===")
        don_list, mem_list, sids, meta = ensemble_cohort(
            fold_sims_root, name, fold_ids
        )
        scored = score_effects_vs_signature(don_list, mem_list, sig)
        # Cosine of cohort-mean effects (matches bootstrap companion metric)
        ext_don = np.mean(np.stack(don_list), 0)
        ext_mem = np.mean(np.stack(mem_list), 0)
        cos_d = float(
            np.dot(ext_don, sig["donepezil_effect"])
            / (np.linalg.norm(ext_don) * np.linalg.norm(sig["donepezil_effect"]))
        )
        cos_m = float(
            np.dot(ext_mem, sig["memantine_effect"])
            / (np.linalg.norm(ext_mem) * np.linalg.norm(sig["memantine_effect"]))
        )
        block = {
            **meta,
            **scored,
            "donepezil_effect_cosine": cos_d,
            "memantine_effect_cosine": cos_m,
            "effect_magnitude_cosine_mean": float(np.nanmean([cos_d, cos_m])),
        }
        # Drop large subject_ids list from printed summary path; keep in JSON
        results["cohorts"][name] = block
        print(
            f"  n={meta['n_subjects_common']} "
            f"{scored['direction_agreement_total']} "
            f"r={scored['effect_magnitude_correlation']:.6f} "
            f"cos={block['effect_magnitude_cosine_mean']:.6f}"
        )

    json_path = out_dir / "fold_ensemble_external_direction_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Five-fold ensemble external direction",
        "",
        "Per-subject mean of fold_0..4 effect vectors, then score vs "
        "constrained training signature.",
        "",
        f"Signature: `{sig['source']}`",
        "",
    ]
    for name in cohorts:
        c = results["cohorts"][name]
        lines.append(
            f"- **{name}** (n={c['n_subjects_common']}): "
            f"{c['direction_agreement_total']}; "
            f"r={c['effect_magnitude_correlation']:.6f}; "
            f"cos={c['effect_magnitude_cosine_mean']:.6f}"
        )
    md_path = out_dir / "fold_ensemble_external_direction_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
