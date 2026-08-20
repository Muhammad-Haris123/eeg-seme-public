"""
Load external OSF EEG data from per-channel .txt files.

Reads Eyes_closed (or Eyes_open) subjects from testing_data/AD and testing_data/Healthy.
Loads required 19 channels only in exact order. Validates equal length across channels.
Output: (n_subjects, 19, n_timepoints) per group.

Author: Research Team
Date: 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Required channels in exact order (10-20 system)
REQUIRED_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
]

# Group mapping: OSF folder name -> internal group name
GROUP_MAP = {
    "AD": "AD",
    "Healthy": "HC",
}


def load_channel_txt(path: Path) -> np.ndarray:
    """Load a single channel .txt file (one value per line)."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(float(line))
                except ValueError:
                    continue
    return np.array(data, dtype=np.float32)


def load_subject_txt(
    subject_dir: Path,
    channels: List[str],
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Load one subject's EEG from per-channel .txt files.
    
    Returns:
        (data, meta) where data is (n_channels, n_timepoints), or None if invalid.
    """
    channel_arrays = []
    lengths = []
    source_paths = {}
    
    for ch in channels:
        ch_path = subject_dir / f"{ch}.txt"
        if not ch_path.exists():
            return None
        arr = load_channel_txt(ch_path)
        if len(arr) == 0:
            return None
        channel_arrays.append(arr)
        lengths.append(len(arr))
        source_paths[ch] = str(ch_path)
    
    # Validate equal length
    if len(set(lengths)) != 1:
        return None
    
    data = np.stack(channel_arrays, axis=0)  # (19, n_timepoints)
    meta = {
        "subject_id": subject_dir.name,
        "channels": channels,
        "n_timepoints": lengths[0],
        "source_paths": source_paths,
    }
    return data, meta


def discover_subjects(
    root: Path,
    group_folder: str,
    condition: str = "Eyes_closed",
) -> List[Path]:
    """Discover Paciente* subject folders under group/condition."""
    group_path = root / group_folder / condition
    if not group_path.exists():
        return []
    subjects = sorted([d for d in group_path.iterdir() if d.is_dir() and d.name.startswith("Paciente")])
    return subjects


def import_osf_txt(
    testing_data_root: Path,
    output_dir: Path,
    condition: str = "Eyes_closed",
    channels: Optional[List[str]] = None,
) -> Dict:
    """
    Import OSF txt channel files into project-compatible arrays.
    
    Args:
        testing_data_root: Path to testing_data/
        output_dir: Path to data/external_osf/raw_arrays/
        condition: 'Eyes_closed' or 'Eyes_open'
        channels: Channel list (default: REQUIRED_CHANNELS)
    
    Returns:
        Integrity report dict with counts, skipped, missing channels, etc.
    """
    channels = channels or REQUIRED_CHANNELS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "condition": condition,
        "channels_required": channels,
        "channels_dropped": [],
        "groups": {},
        "skipped_subjects": [],
        "missing_channels": {},
        "duplicates": [],
        "imported": {},
        "metadata": {},
    }
    
    all_raw = {}
    
    for osf_group, internal_group in GROUP_MAP.items():
        subjects = discover_subjects(testing_data_root, osf_group, condition)
        report["groups"][internal_group] = {"total_folders": len(subjects)}
        
        valid_data = []
        valid_meta = []
        skipped = []
        
        for subj_dir in subjects:
            result = load_subject_txt(subj_dir, channels)
            if result is None:
                skipped.append({
                    "subject": subj_dir.name,
                    "reason": "missing_channel_or_empty",
                    "path": str(subj_dir),
                })
                continue
            
            data, meta = result
            valid_data.append(data)
            valid_meta.append(meta)
        
        report["skipped_subjects"].extend(skipped)
        report["groups"][internal_group]["skipped"] = len(skipped)
        report["groups"][internal_group]["imported"] = len(valid_data)
        
        if len(valid_data) == 0:
            report["imported"][internal_group] = None
            report["metadata"][internal_group] = []
            continue
        
        # Stack: (n_subjects, 19, n_timepoints)
        stacked = np.stack(valid_data, axis=0)
        report["imported"][internal_group] = {
            "shape": list(stacked.shape),
            "n_subjects": stacked.shape[0],
            "n_channels": stacked.shape[1],
            "n_timepoints": stacked.shape[2],
        }
        report["metadata"][internal_group] = valid_meta
        
        # Save
        np.save(output_dir / f"{internal_group}_raw.npy", stacked.astype(np.float32))
    
    # Save metadata
    meta_path = output_dir / "import_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    
    return report
