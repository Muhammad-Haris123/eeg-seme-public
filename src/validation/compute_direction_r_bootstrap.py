"""
Subject-level bootstrap 95% CIs for constrained external effect-vector r.

Uses existing fold-0 sims under data/{tuh_validation,ad_labeled_external,padic_external}.
Does NOT treat feature length 2185 as an independent-observation n.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.validation.direction_metrics import load_constrained_training_effects
from src.validation.stats_ci import subject_bootstrap_effect_metrics


def _load_effects(sim_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    paths = sorted(sim_dir.glob("*_sims.npz"))
    if not paths:
        raise FileNotFoundError(f"No *_sims.npz in {sim_dir}")
    dons, mems, sids = [], [], []
    for p in paths:
        z = np.load(p)
        dons.append(z["donepezil"] - z["baseline"])
        mems.append(z["memantine"] - z["baseline"])
        sids.append(p.name.replace("_sims.npz", ""))
    return np.stack(dons), np.stack(mems), sids


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sig = load_constrained_training_effects(root)
    cohorts = {
        "tuh": root / "data" / "tuh_validation",
        "osf": root / "data" / "ad_labeled_external" / "validation",
        "padic": root / "data" / "padic_external" / "validation",
    }
    out = {
        "timestamp": datetime.now().isoformat(),
        "method": "subject_bootstrap_percentile",
        "n_boot": 2000,
        "seed": 42,
        "signature_source": sig["source"],
        "note": (
            "CIs resample subjects with replacement, then recompute cohort-mean "
            "effect vectors vs the fixed training constrained signature. "
            "Fisher-z CIs with n=2185 features are NOT used for inference."
        ),
        "cohorts": {},
    }
    rows = []
    for name, sim_dir in cohorts.items():
        don, mem, sids = _load_effects(sim_dir)
        boot = subject_bootstrap_effect_metrics(
            don, mem, sig["donepezil_effect"], sig["memantine_effect"], n_boot=2000, seed=42
        )
        # Point estimates from all subjects (matches locked protocol)
        out["cohorts"][name] = {
            "n_used": len(sids),
            "sim_dir": str(sim_dir),
            **boot,
        }
        rows.append(f"- **{name}**: {boot['effect_magnitude_correlation_formatted']}")
        print(name, boot["effect_magnitude_correlation_formatted"])

    path = root / "models" / "validation" / "direction_r_subject_bootstrap_ci.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = root / "models" / "validation" / "direction_r_subject_bootstrap_ci.md"
    md.write_text(
        "# Subject-bootstrap 95% CIs for constrained direction r\n\n"
        "Uncertainty over **subjects**, not feature length.\n\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"[wrote] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
