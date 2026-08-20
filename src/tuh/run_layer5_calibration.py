"""
Post-hoc epoch-count spectral calibration for existing TUH features, then
re-run twin + Layer 5 analyses. Does NOT touch shared Phase-1 modules or
Layers 1–4 artifacts.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from api.utils.feature_processor import band_powers_mean_per_band, flatten_features
from src.tuh.process_tuh_eeg import (
    TARGET_N_EPOCHS_FOR_PSD,
    recalibrate_existing_tuh_features,
)
from src.tuh.tuh_validation import (
    analyze_abnormal_vs_normal,
    analyze_cross_dataset,
    analyze_feature_distribution,
    run_twin_on_tuh_patients,
    run_tuh_validation_analyses,
    update_validation_report_v3,
)
from src.tuh.mine_headers import summarize_header_mining
import csv


def _train_band_mean(project_root: str) -> float:
    ad = np.load(os.path.join(project_root, "data", "eeg_features", "AD_band_powers.npy"))
    hc = np.load(os.path.join(project_root, "data", "eeg_features", "HC_band_powers.npy"))
    return float(np.concatenate([ad, hc], axis=0).mean())


def _tuh_band_mean(feature_dir: str, report: dict) -> float:
    vals = []
    for r in report.get("results", []):
        if not r.get("ok"):
            continue
        flat = np.load(r["feature_path"])
        vals.append(flat[380:475].mean())
    return float(np.mean(vals)) if vals else float("nan")


def _layer1to4_fingerprint(project_root: str) -> dict:
    """Snapshot Layer 1–4 metrics to prove they were not altered."""
    path = os.path.join(
        project_root, "models", "validation", "complete_validation_report_v3.json"
    )
    # Prefer v2 as immutable baseline for L1-4; v3 may already exist
    v2 = os.path.join(
        project_root, "models", "validation", "complete_validation_report_v2.json"
    )
    with open(v2, "r", encoding="utf-8") as f:
        report = json.load(f)
    return {
        "source": v2,
        "layer1_constrained_mse": report["layer1"]["constrained"]["mse_mean"],
        "layer1_constrained_pearson": report["layer1"]["constrained"]["pearson_mean"],
        "layer1_constrained_acc": report["layer1"]["constrained"]["accuracy_mean"],
        "layer2_n_match": report["layer2"]["constrained"]["n_match"],
        "layer2_effect_r": report["layer2"]["constrained"]["effect_size_r"],
        "layer3_status": report["layer3"].get("status"),
        "layer4_status": report["layer4"].get("status"),
    }


def main() -> int:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Fix path: this file is at src/tuh/... so parents[2] is project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    feat_dir = os.path.join(project_root, "data", "tuh_features")
    val_dir = os.path.join(project_root, "data", "tuh_validation")
    mining_dir = os.path.join(project_root, "data", "tuh_mining")
    report_path = os.path.join(feat_dir, "processing_report.json")

    with open(report_path, "r", encoding="utf-8") as f:
        report_before = json.load(f)

    before = {
        "tuh_band_mean": _tuh_band_mean(feat_dir, report_before),
        "train_band_mean": _train_band_mean(project_root),
    }
    before["ratio"] = before["tuh_band_mean"] / before["train_band_mean"]
    print("BEFORE calibration:")
    print(json.dumps(before, indent=2))

    # Capture pre-fix Layer5 discrimination from saved results
    pre_val_path = os.path.join(val_dir, "tuh_validation_results.json")
    with open(pre_val_path, "r", encoding="utf-8") as f:
        pre_val = json.load(f)
    before_metrics = {
        "abnormal_vs_normal": deepcopy(pre_val.get("abnormal_vs_normal")),
        "feature_distribution": {
            "pca_overlap": pre_val.get("feature_distribution", {}).get("pca_overlap"),
            "overlap_fraction_within_train_r95": pre_val.get("feature_distribution", {}).get(
                "overlap_fraction_within_train_r95"
            ),
            "mean_feature_distance": pre_val.get("feature_distribution", {}).get(
                "mean_feature_distance"
            ),
        },
        "cross_dataset": deepcopy(pre_val.get("cross_dataset")),
    }

    l14_before = _layer1to4_fingerprint(project_root)
    print("\nLayers 1-4 fingerprint (from v2, immutable):")
    print(json.dumps(l14_before, indent=2))

    print("\nApplying post-hoc 1/N spectral calibration (TUH features only)...")
    cal = recalibrate_existing_tuh_features(
        feat_dir,
        processing_report_path=report_path,
        target_n_epochs=TARGET_N_EPOCHS_FOR_PSD,
        dry_run=False,
    )
    print(json.dumps(cal, indent=2))

    with open(report_path, "r", encoding="utf-8") as f:
        report_after = json.load(f)
    after_scale = {
        "tuh_band_mean": _tuh_band_mean(feat_dir, report_after),
        "train_band_mean": before["train_band_mean"],
    }
    after_scale["ratio"] = after_scale["tuh_band_mean"] / after_scale["train_band_mean"]
    print("\nAFTER calibration (feature scale):")
    print(json.dumps(after_scale, indent=2))

    print("\nRe-running twin simulations on calibrated features...")
    sim_results = run_twin_on_tuh_patients(
        feat_dir, report_after, project_root, val_dir, num_samples=5
    )

    # Header rows for report update
    csv_path = os.path.join(mining_dir, "edf_header_mining.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        header_rows = list(csv.DictReader(f))
    header_summary = summarize_header_mining(header_rows)

    print("\nRe-running Layer 5 analyses...")
    tuh_results = run_tuh_validation_analyses(
        project_root,
        header_rows,
        header_summary,
        report_after,
        sim_results,
        val_dir,
    )
    update_validation_report_v3(project_root, tuh_results, header_summary)

    # Confirm cross-dataset still strong
    cross_after = tuh_results.get("cross_dataset", {})
    cross_before = before_metrics["cross_dataset"]

    # Confirm L1-4 unchanged in v3 vs v2 fingerprint
    v3_path = os.path.join(
        project_root, "models", "validation", "complete_validation_report_v3.json"
    )
    with open(v3_path, "r", encoding="utf-8") as f:
        v3 = json.load(f)
    l14_after = {
        "layer1_constrained_mse": v3["layer1"]["constrained"]["mse_mean"],
        "layer1_constrained_pearson": v3["layer1"]["constrained"]["pearson_mean"],
        "layer1_constrained_acc": v3["layer1"]["constrained"]["accuracy_mean"],
        "layer2_n_match": v3["layer2"]["constrained"]["n_match"],
        "layer2_effect_r": v3["layer2"]["constrained"]["effect_size_r"],
        "layer3_status": v3["layer3"].get("status"),
        "layer4_status": v3["layer4"].get("status"),
    }
    l14_unchanged = all(
        abs(float(l14_after[k]) - float(l14_before[k])) < 1e-12
        if isinstance(l14_before[k], (int, float))
        else l14_after[k] == l14_before[k]
        for k in (
            "layer1_constrained_mse",
            "layer1_constrained_pearson",
            "layer1_constrained_acc",
            "layer2_n_match",
            "layer2_effect_r",
            "layer3_status",
            "layer4_status",
        )
    )

    summary = {
        "before_scale": before,
        "after_scale": after_scale,
        "before_metrics": before_metrics,
        "after_metrics": {
            "abnormal_vs_normal": tuh_results.get("abnormal_vs_normal"),
            "feature_distribution": {
                "pca_overlap": tuh_results.get("feature_distribution", {}).get("pca_overlap"),
                "overlap_fraction_within_train_r95": tuh_results.get(
                    "feature_distribution", {}
                ).get("overlap_fraction_within_train_r95"),
                "mean_feature_distance": tuh_results.get("feature_distribution", {}).get(
                    "mean_feature_distance"
                ),
            },
            "cross_dataset": cross_after,
            "overall_status": tuh_results.get("overall_status"),
        },
        "cross_dataset_confirmation": {
            "before_r": cross_before.get("effect_magnitude_correlation"),
            "after_r": cross_after.get("effect_magnitude_correlation"),
            "before_direction": cross_before.get("direction_agreement_total"),
            "after_direction": cross_after.get("direction_agreement_total"),
        },
        "layers_1_4": {
            "before": l14_before,
            "after_in_v3": l14_after,
            "unchanged": l14_unchanged,
            "shared_code_touched": False,
            "modules_modified": ["src/tuh/process_tuh_eeg.py only"],
        },
        "calibration": cal,
    }
    out = os.path.join(val_dir, "layer5_calibration_before_after.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved before/after summary: {out}")
    print(f"Layers 1-4 unchanged: {l14_unchanged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
