"""Compute Fisher-z 95% CIs for locked constrained direction r values."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.validation.stats_ci import fisher_z_ci, format_r_ci

FEAT_DIM = 2185  # effect-vector length for corrcoef(r)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    v3 = json.loads((root / "models" / "validation" / "complete_validation_report_v3.json").read_text(encoding="utf-8"))
    rows = []
    mapping = [
        ("tuh", v3["layer5"]["cross_dataset"]),
        ("osf", v3["layer5b_ad_labeled_external"]["cross_dataset"]),
        ("padic", v3["layer5d_padic_external"]["cross_dataset"]),
    ]
    out = {"timestamp": datetime.now().isoformat(), "n_features_for_r": FEAT_DIM, "cohorts": {}}
    for name, cross in mapping:
        r = float(cross["effect_magnitude_correlation"])
        r_d = float(cross.get("donepezil_effect_corr", r))
        r_m = float(cross.get("memantine_effect_corr", r))
        block = {
            "direction_agreement_total": cross.get("direction_agreement_total"),
            "effect_magnitude_correlation": r,
            "effect_magnitude_correlation_ci": fisher_z_ci(r, FEAT_DIM),
            "effect_magnitude_correlation_formatted": format_r_ci(r, FEAT_DIM),
            "donepezil_effect_corr": r_d,
            "donepezil_effect_corr_ci": fisher_z_ci(r_d, FEAT_DIM),
            "memantine_effect_corr": r_m,
            "memantine_effect_corr_ci": fisher_z_ci(r_m, FEAT_DIM),
            "n_used": cross.get("n_tuh_used") or cross.get("n_used"),
        }
        out["cohorts"][name] = block
        rows.append(f"- {name}: {block['effect_magnitude_correlation_formatted']}")

    path = root / "models" / "validation" / "direction_r_fisher_ci.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = root / "models" / "validation" / "direction_r_fisher_ci.md"
    md.write_text(
        "# Fisher-z 95% CIs for constrained direction r\n\n"
        f"n for Fisher z = feature length {FEAT_DIM} (vector-vs-vector correlation).\n\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out["cohorts"], indent=2))
    print(f"[wrote] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
