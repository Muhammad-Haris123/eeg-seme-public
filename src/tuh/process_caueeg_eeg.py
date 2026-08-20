"""
Load CAUEEG EDFs into the shared 2185-D EEG feature pipeline.

Uses dementia-no-overlap Normal / MCI / Dementia labels.
Reuses TUH channel mapping (strips -AVG) + extract_tuh_features.
South Korea mains → 60 Hz notch.
Does not modify shared src/eeg/* modules.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.tuh.process_tuh_eeg import (
    CROP_SECONDS,
    MAX_EPOCHS_FOR_FEATURES,
    TARGET_N_EPOCHS_FOR_PSD,
    extract_tuh_features,
    load_tuh_edf,
)

CAUEEG_NOTCH_HZ = 60.0  # South Korea


def default_raw_root() -> Path:
    return Path(r"E:\caueeg_external\extracted\caueeg-dataset")


def load_dementia_records(
    raw_root: Path,
    split_file: str = "dementia-no-overlap.json",
    splits: Optional[Tuple[str, ...]] = None,
) -> List[Dict]:
    """Flatten train/validation/test into one list of labeled recordings."""
    if splits is None:
        splits = ("train_split", "validation_split", "test_split")
    data = json.loads((raw_root / split_file).read_text(encoding="utf-8"))
    out: List[Dict] = []
    for sp in splits:
        for rec in data.get(sp, []):
            class_name = rec.get("class_name") or "Unknown"
            class_label = int(rec.get("class_label", -1))
            # Twin disease flag: Dementia≈AD-like (1), Normal/MCI (0)
            disease_label = 1 if class_name == "Dementia" else 0
            out.append(
                {
                    "serial": str(rec["serial"]),
                    "age": rec.get("age"),
                    "symptom": rec.get("symptom") or [],
                    "class_name": class_name,
                    "class_label": class_label,
                    "disease_label": disease_label,
                    "split": sp.replace("_split", ""),
                    "has_ad_tag": any(
                        str(s).lower() in ("ad", "alzheimers", "alzheimer")
                        for s in (rec.get("symptom") or [])
                    ),
                }
            )
    # de-dupe by serial (keep first)
    seen = set()
    uniq = []
    for r in out:
        if r["serial"] in seen:
            continue
        seen.add(r["serial"])
        uniq.append(r)
    return uniq


def edf_path_for(raw_root: Path, serial: str) -> Path:
    return raw_root / "signal" / "edf" / f"{serial}.edf"


def process_caueeg_cohort(
    raw_root: Path,
    out_dir: Path,
    split_file: str = "dementia-no-overlap.json",
    crop_seconds: float = CROP_SECONDS,
    resume: bool = True,
    limit: Optional[int] = None,
) -> Dict:
    """Extract calibrated 2185-D features for CAUEEG dementia-benchmark subjects."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    records = load_dementia_records(raw_root, split_file=split_file)
    if limit is not None:
        records = records[: int(limit)]

    report_path = out_dir / "processing_report.json"
    existing_by_serial: Dict[str, Dict] = {}
    if resume and report_path.exists():
        prev = json.loads(report_path.read_text(encoding="utf-8"))
        for r in prev.get("results", []):
            if r.get("ok") and r.get("serial"):
                existing_by_serial[r["serial"]] = r

    results: List[Dict] = []
    epoch_by_class: Dict[str, List[int]] = {"Normal": [], "MCI": [], "Dementia": []}

    print(f"[CAUEEG] {len(records)} labeled recordings ({split_file})")
    for i, rec in enumerate(records):
        serial = rec["serial"]
        if serial in existing_by_serial:
            r = existing_by_serial[serial]
            results.append(r)
            n_ep = int(r.get("n_epochs_used") or 0)
            if rec["class_name"] in epoch_by_class and n_ep:
                epoch_by_class[rec["class_name"]].append(n_ep)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(records)}] skip existing {serial}")
            continue

        edf = edf_path_for(raw_root, serial)
        rec_out = dict(rec)
        try:
            if not edf.exists():
                raise FileNotFoundError(edf)
            raw, load_meta = load_tuh_edf(
                str(edf),
                crop_seconds=crop_seconds,
                notch_freq=CAUEEG_NOTCH_HZ,
            )
            flat, structured, feat_meta = extract_tuh_features(
                raw,
                apply_ica=False,
                max_epochs=MAX_EPOCHS_FOR_FEATURES,
                notch_freq=CAUEEG_NOTCH_HZ,
                apply_epoch_power_calibration=True,
                target_n_epochs=TARGET_N_EPOCHS_FOR_PSD,
            )
            n_ep = int(feat_meta.get("epoch_power_calibration", {}).get("n_epochs_used", 0))
            if rec["class_name"] in epoch_by_class:
                epoch_by_class[rec["class_name"]].append(n_ep)

            feat_path = feat_dir / f"{serial}_features.npy"
            struct_path = feat_dir / f"{serial}_structured.npz"
            np.save(feat_path, flat.astype(np.float32))
            np.savez_compressed(
                struct_path,
                psd=structured["psd"],
                band_powers=structured["band_powers"],
                coherence=structured["coherence"],
                plv=structured["plv"],
            )
            rec_out.update(
                {
                    "ok": True,
                    "feature_path": str(feat_path),
                    "structured_path": str(struct_path),
                    "n_epochs_used": n_ep,
                    "load_meta": load_meta,
                    "epoch_power_calibration": feat_meta.get("epoch_power_calibration"),
                    "group": rec["class_name"],
                    "label": rec["class_name"].lower(),
                }
            )
            print(
                f"  [{i+1}/{len(records)}] {serial} {rec['class_name']} "
                f"age={rec.get('age')} epochs={n_ep} "
                f"scale={feat_meta.get('epoch_power_calibration', {}).get('scale')}"
            )
        except Exception as e:
            rec_out.update(
                {
                    "ok": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "group": rec["class_name"],
                    "label": rec["class_name"].lower(),
                }
            )
            print(f"  [{i+1}/{len(records)}] FAIL {serial}: {e}")
        results.append(rec_out)

        # checkpoint every 25
        if (i + 1) % 25 == 0:
            _write_report(
                report_path,
                raw_root,
                out_dir,
                split_file,
                crop_seconds,
                results,
                epoch_by_class,
            )

    report = _write_report(
        report_path, raw_root, out_dir, split_file, crop_seconds, results, epoch_by_class
    )
    print(f"[done] {report['n_ok']} ok / {report['n_fail']} fail -> {report_path}")
    return report


def _write_report(
    report_path: Path,
    raw_root: Path,
    out_dir: Path,
    split_file: str,
    crop_seconds: float,
    results: List[Dict],
    epoch_by_class: Dict[str, List[int]],
) -> Dict:
    def _summ(v: List[int]) -> Dict:
        return {
            "n": len(v),
            "min": int(min(v)) if v else None,
            "median": float(np.median(v)) if v else None,
            "max": int(max(v)) if v else None,
            "mean": float(np.mean(v)) if v else None,
        }

    by_class = {}
    for name in ("Normal", "MCI", "Dementia"):
        ok = [r for r in results if r.get("ok") and r.get("class_name") == name]
        by_class[name] = len(ok)

    report = {
        "dataset": "CAUEEG",
        "citation_doi": "10.1016/j.neuroimage.2023.120054",
        "raw_root": str(raw_root),
        "out_dir": str(out_dir),
        "split_file": split_file,
        "notch_hz": CAUEEG_NOTCH_HZ,
        "crop_seconds": crop_seconds,
        "target_n_epochs_for_psd": TARGET_N_EPOCHS_FOR_PSD,
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": sum(1 for r in results if not r.get("ok")),
        "n_by_class_ok": by_class,
        "epoch_counts": {k: _summ(v) for k, v in epoch_by_class.items()},
        "caveats": [
            "Dementia is clinical spectrum (AD and non-AD subtypes), not pure AD-only",
            "Average-referenced hospital EEG (Fp1-AVG …); mapped via TUH -AVG strip",
            "No treatment-response / drug outcome labels in this release",
        ],
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=default_raw_root())
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "caueeg_external",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "README_LOCATION.txt").write_text(
        f"Raw CAUEEG at: {args.raw_root}\nFeatures under this folder.\n"
        "Academic/non-commercial use only. Cite Kim et al. NeuroImage 2023.\n",
        encoding="utf-8",
    )
    process_caueeg_cohort(
        args.raw_root,
        args.out_dir,
        resume=not args.no_resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
