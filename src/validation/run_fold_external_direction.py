"""
Five-fold constrained external direction robustness (TUH / OSF / P-ADIC).

Uses models/checkpoints_constrained/fold_{0..4}_best.pt against the fixed
training constrained signature. Main paper checkpoint remains fold_0 alias.
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
    simulate_flat_cohort,
)
from src.validation.run_unconstrained_external_battery import load_osf, load_padic, load_tuh


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _summarize(fold_scores: List[Dict]) -> Dict:
    rs = [float(s["effect_magnitude_correlation"]) for s in fold_scores]
    agrees = [s["direction_agreement_total"] for s in fold_scores]
    return {
        "n_folds": len(fold_scores),
        "effect_magnitude_correlation_mean": float(np.mean(rs)),
        "effect_magnitude_correlation_std": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0,
        "effect_magnitude_correlation_per_fold": rs,
        "direction_agreement_per_fold": agrees,
        "all_folds_10_of_10": all(a == "10/10" for a in agrees),
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--cohorts", type=str, default="tuh,osf,padic")
    args = ap.parse_args()

    root = _root()
    sig = load_constrained_training_effects(root)
    fold_ids = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    cohort_names = [c.strip() for c in args.cohorts.split(",") if c.strip()]

    loaders = {
        "tuh": load_tuh,
        "osf": load_osf,
        "padic": load_padic,
    }
    data = {}
    for name in cohort_names:
        flats, diseases, sids, labels = loaders[name](root)
        data[name] = (flats, diseases, sids, labels)
        print(f"loaded {name}: n={len(sids)}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "constrained_fivefold_external_direction",
        "signature_source": sig["source"],
        "num_samples": int(args.num_samples),
        "checkpoint_pattern": "models/checkpoints_constrained/fold_{k}_best.pt",
        "main_paper_checkpoint": "models/checkpoints_constrained/checkpoint_constrained.pt (fold_0 alias)",
        "cohorts": {},
        "folds": {},
    }

    for k in fold_ids:
        ckpt = root / "models" / "checkpoints_constrained" / f"fold_{k}_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        print(f"\n=== fold {k}: {ckpt.name} ===")
        fold_block = {"checkpoint": str(ckpt), "cohorts": {}}
        for name in cohort_names:
            flats, diseases, sids, _ = data[name]
            out_dir = root / "models" / "validation" / "fold_sims" / f"fold_{k}" / name
            pack = simulate_flat_cohort(
                flats,
                diseases,
                sids,
                ckpt,
                out_dir,
                num_samples=int(args.num_samples),
                seed=42 + k,
            )
            scored = score_effects_vs_signature(
                pack["don_effects"], pack["mem_effects"], sig
            )
            fold_block["cohorts"][name] = scored
            print(
                f"  {name}: {scored['direction_agreement_total']} "
                f"r={scored['effect_magnitude_correlation']:.3f}"
            )
        results["folds"][f"fold_{k}"] = fold_block

    for name in cohort_names:
        fold_scores = [results["folds"][f"fold_{k}"]["cohorts"][name] for k in fold_ids]
        results["cohorts"][name] = _summarize(fold_scores)
        s = results["cohorts"][name]
        print(
            f"SUMMARY {name}: r={s['effect_magnitude_correlation_mean']:.3f}"
            f"±{s['effect_magnitude_correlation_std']:.3f}; "
            f"agreements={s['direction_agreement_per_fold']}"
        )

    path = root / "models" / "validation" / "fold_external_direction_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# Five-fold constrained external direction robustness",
        "",
        f"Signature: `{sig['source']}`",
        f"Folds: {fold_ids}",
        "",
    ]
    for name in cohort_names:
        s = results["cohorts"][name]
        lines.append(
            f"- **{name}**: r = {s['effect_magnitude_correlation_mean']:.3f} "
            f"± {s['effect_magnitude_correlation_std']:.3f}; "
            f"agreements {s['direction_agreement_per_fold']}; "
            f"all 10/10 = {s['all_folds_10_of_10']}"
        )
    md = root / "models" / "validation" / "fold_external_direction_summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
