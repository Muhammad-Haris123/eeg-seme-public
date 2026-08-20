"""
External BrainLAT (syn51549340) EEG importer.

Reads BrainLAT metadata CSVs and discovers EEGLAB .set files under:
  syn51549340/EEG data/1_AD/...
  syn51549340/EEG data/5_HC/...

For each subject with eeg == 1, locates the corresponding .set file,
loads minimal header information (channels, sfreq, n_samples),
and writes a summary JSON report.

This module does NOT save any NumPy arrays – it only inspects the
dataset and builds an integrity report that is later consumed by the
BrainLAT external Phase 1 script.

Outputs:
  data/external_brainlat/raw_import_metadata.json

Author: Research Team
Date: 2026
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import mne
import pandas as pd

from ..utils.config import PROJECT_ROOT, DATA_ROOT, STANDARD_10_20_CHANNELS

# Root to BrainLAT EEG data (local copy of syn51549340)
BRAINLAT_EEG_ROOT = PROJECT_ROOT / "syn51549340" / "EEG data"

# Output base for BrainLAT external data
EXTERNAL_BRAINLAT_ROOT = DATA_ROOT / "external_brainlat"
EXTERNAL_BRAINLAT_RAW_DIR = EXTERNAL_BRAINLAT_ROOT / "raw_import"


@dataclass
class BrainLATSubjectInfo:
    subject_id: str
    group: str  # 'AD' or 'HC'
    dataset_path: str  # CSV path column, e.g. '1_AD/AR'
    eeglab_file: str   # full path to .set file
    sfreq_in: float
    n_samples: int
    duration_sec: float
    channel_names: List[str]
    has_all_required_channels: bool
    missing_required_channels: List[str]


def _csv_path(group: str) -> Path:
    """Return path to BrainLAT records CSV for a group ('AD' or 'HC')."""
    if group == "AD":
        return BRAINLAT_EEG_ROOT / "1_AD" / "records_ad_eeg_data.csv"
    elif group == "HC":
        return BRAINLAT_EEG_ROOT / "5_HC" / "records_hc_eeg_data.csv"
    else:
        raise ValueError(f"Unknown group: {group}")


def _load_records(group: str) -> pd.DataFrame:
    """Load records CSV for a group and filter rows with eeg == 1."""
    csv_path = _csv_path(group)
    if not csv_path.exists():
        raise FileNotFoundError(f"BrainLAT records CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Ensure 'eeg' column exists
    if "eeg" not in df.columns:
        raise ValueError(f"'eeg' column not found in {csv_path}")

    df = df[df["eeg"] == 1].copy()
    return df


def _find_eeglab_file(path_col: str, subject_id: str) -> Optional[Path]:
    """
    Resolve EEG .set file for a subject.

    Expected layout (example for AD):
      BRAINLAT_EEG_ROOT / '1_AD/AR' / 'sub-30001' / 'eeg' / *eeg.set
    """
    # Normalize path separators just in case
    rel_path = Path(path_col.replace("\\", "/"))
    subject_dir = BRAINLAT_EEG_ROOT / rel_path / subject_id / "eeg"

    if not subject_dir.exists():
        return None

    candidates = list(subject_dir.glob("*.set"))
    if not candidates:
        return None

    # Prefer files that look like resting-state rs-hep, else first
    prioritized = sorted(
        candidates,
        key=lambda p: (
            "rs-hep" not in p.name.lower(),
            "rs" not in p.name.lower(),
            p.name,
        ),
    )
    return prioritized[0]


def _inspect_eeglab_file(
    set_path: Path,
    group: str,
    dataset_path: str,
    subject_id: str,
) -> BrainLATSubjectInfo:
    """Load a .set file with MNE (header only) and collect metadata."""
    raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])
    n_samples = int(raw.n_times)
    duration = float(n_samples / sfreq) if sfreq > 0 else 0.0
    ch_names = list(raw.ch_names)

    missing = [ch for ch in STANDARD_10_20_CHANNELS if ch not in ch_names]
    has_all = len(missing) == 0

    return BrainLATSubjectInfo(
        subject_id=subject_id,
        group=group,
        dataset_path=dataset_path,
        eeglab_file=str(set_path),
        sfreq_in=sfreq,
        n_samples=n_samples,
        duration_sec=duration,
        channel_names=ch_names,
        has_all_required_channels=has_all,
        missing_required_channels=missing,
    )


def build_brainlat_import_report() -> Dict:
    """
    Discover BrainLAT EEG subjects and build an import integrity report.

    Returns:
        Dictionary with per-subject metadata and overall counts. The same
        structure is written to
        data/external_brainlat/raw_import_metadata.json.
    """
    EXTERNAL_BRAINLAT_RAW_DIR.mkdir(parents=True, exist_ok=True)

    report: Dict = {
        "root": str(BRAINLAT_EEG_ROOT),
        "required_channels": STANDARD_10_20_CHANNELS,
        "groups": {
            "AD": {
                "n_records_eeg_1": 0,
                "n_imported": 0,
                "n_missing_set": 0,
                "n_missing_channels": 0,
            },
            "HC": {
                "n_records_eeg_1": 0,
                "n_imported": 0,
                "n_missing_set": 0,
                "n_missing_channels": 0,
            },
        },
        "subjects": {
            "AD": [],
            "HC": [],
        },
        "skipped": [],
        "sfreq_distribution": {
            "AD": [],
            "HC": [],
        },
    }

    for group, csv_group in (("AD", "1_AD"), ("HC", "5_HC")):
        try:
            df = _load_records(group)
        except Exception as e:
            report["groups"][group]["error"] = str(e)
            continue

        report["groups"][group]["n_records_eeg_1"] = int(len(df))

        for _, row in df.iterrows():
            dataset_path = str(row["path"])
            subject_id = str(row["id_EEG"])

            set_path = _find_eeglab_file(dataset_path, subject_id)
            if set_path is None:
                report["groups"][group]["n_missing_set"] += 1
                report["skipped"].append(
                    {
                        "subject_id": subject_id,
                        "group": group,
                        "reason": "no_set_file",
                        "dataset_path": dataset_path,
                    }
                )
                continue

            try:
                info = _inspect_eeglab_file(set_path, group, dataset_path, subject_id)
            except Exception as e:
                report["skipped"].append(
                    {
                        "subject_id": subject_id,
                        "group": group,
                        "reason": f"load_error: {e}",
                        "dataset_path": dataset_path,
                        "eeglab_file": str(set_path),
                    }
                )
                continue

            report["sfreq_distribution"][group].append(info.sfreq_in)

            if not info.has_all_required_channels:
                report["groups"][group]["n_missing_channels"] += 1

            report["groups"][group]["n_imported"] += 1
            report["subjects"][group].append(asdict(info))

    # Write JSON report
    out_path = EXTERNAL_BRAINLAT_ROOT / "raw_import_metadata.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    r = build_brainlat_import_report()
    print(
        "BrainLAT import summary:",
        "AD imported =", r["groups"]["AD"]["n_imported"],
        "HC imported =", r["groups"]["HC"]["n_imported"],
    )

