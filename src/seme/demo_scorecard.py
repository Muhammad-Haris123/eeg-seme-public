"""
SEME public demo: print a locked scorecard from frozen JSON only.

No raw EEG, no .pt checkpoints, no network. Safe for the public mirror.

Usage (from repo root):
  python src/seme/demo_scorecard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "models" / "validation"


def _load(name: str) -> dict:
    path = VAL / name
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> int:
    v3 = _load("complete_validation_report_v3.json")
    unc = _load("unconstrained_external_battery.json")
    prior = _load("prior_only_direction.json")
    latent = _load("layer5e_caueeg_latent_probe_results.json")
    head = _load("layer5e_caueeg_classifier_head_results.json")
    raw = _load("layer5e_caueeg_raw2185_probe_results.json")

    def locked(layer: str):
        cd = _get(v3, layer, "cross_dataset", default={})
        return cd.get("direction_agreement_total"), cd.get("effect_magnitude_correlation")

    pairs = [
        ("TUH", "layer5"),
        ("OSF", "layer5b_ad_labeled_external"),
        ("P-ADIC", "layer5d_padic_external"),
    ]

    print("=" * 64)
    print("SEME scorecard (frozen JSON demo; no EEG / no checkpoints)")
    print("=" * 64)
    print("Primary (S) - constrained locked signs + continuous r")
    for name, key in pairs:
        signs, r = locked(key)
        print(f"  {name:7s}  signs={signs}   r={float(r):.3f}")

    print("Primary contrast - unconstrained locked signs")
    for name, key in [("TUH", "tuh"), ("OSF", "osf"), ("P-ADIC", "padic")]:
        signs = _get(unc, "cohorts", key, "direction", "direction_agreement_total")
        print(f"  {name:7s}  signs={signs}")

    print("Circularity budget - prior-only")
    print(
        f"  signs={prior.get('direction_agreement_total')}  "
        f"mean_r={float(prior.get('effect_magnitude_correlation_mean')):.3f}"
    )

    print("Reporting (Encoding) - CAUEEG Dementia vs Normal")
    lat = float(_get(latent, "latent_probe_dementia_vs_normal", "cv", "roc_auc", "mean"))
    ta = float(
        _get(latent, "feature_probe_theta_alpha_dementia_vs_normal", "cv", "roc_auc", "mean")
    )
    r2185 = float(_get(raw, "raw2185_logistic_dementia_vs_normal", "cv", "roc_auc", "mean"))
    best = float(_get(head, "best_head", "oof_roc_auc"))
    print(f"  latent AUC:     {lat:.3f}")
    print(f"  theta/alpha:    {ta:.3f}")
    print(f"  2185-D ref:     {r2185:.3f}")
    print(f"  best head OOF:  {best:.3f}")
    print("-" * 64)
    print("NOTE: signature concordance != empirical post-dose PD validation.")
    print("Weights: https://doi.org/10.5281/zenodo.22028681")
    print("Code:    https://github.com/Muhammad-Haris123/eeg-seme-public")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"demo_scorecard failed: {exc}", file=sys.stderr)
        raise
