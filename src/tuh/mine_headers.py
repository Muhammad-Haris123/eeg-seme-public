"""
Mine TUH EDF headers for AD drug / diagnosis keyword mentions.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

AD_DRUG_KEYWORDS = {
    "donepezil": ("donepezil", "aricept"),
    "memantine": ("memantine", "namenda", "namzaric"),
    "galantamine": ("galantamine", "razadyne", "reminyl"),
    "rivastigmine": ("rivastigmine", "exelon"),
}

AD_DIAGNOSIS_KEYWORDS = (
    "alzheimer",
    "dementia",
    "cognitive decline",
    "cognitive impairment",
    "memory loss",
    "memory impairment",
    "neurodegen",
    "mild cognitive",
)

GENERAL_MED_KEYWORDS = (
    "medication",
    "prescribed",
    "taking",
    "treatment",
    " mg",
    "dose",
    "daily",
)

NEURO_KEYWORDS = (
    "seizure",
    "epilepsy",
    "encephalopathy",
    "stroke",
    "tumor",
    "neuro",
    "abnormal",
    "spike",
    "slowing",
)


def _safe_str(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="ignore")
        except Exception:
            return obj.decode("latin-1", errors="ignore")
    if isinstance(obj, dict):
        return " ".join(_safe_str(v) for v in obj.values())
    return str(obj)


def read_raw_edf_header_bytes(edf_path: str, n_bytes: int = 512) -> str:
    """Read the leading ASCII header bytes of an EDF file."""
    with open(edf_path, "rb") as f:
        raw = f.read(n_bytes)
    return raw.decode("latin-1", errors="ignore")


def parse_filename_ids(filename: str) -> Tuple[str, str]:
    """Parse patient_id and session_id from TUH-style filenames."""
    stem = Path(filename).stem
    # aaaaabdo_s003_t000
    m = re.match(r"^([a-z]{8})_(s\d{3})", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    parts = stem.split("_")
    patient = parts[0].lower() if parts else stem.lower()
    session = "unknown"
    for p in parts[1:]:
        if re.match(r"^s\d{3}$", p, flags=re.IGNORECASE):
            session = p.lower()
            break
    return patient, session


def infer_label_from_path(edf_path: str) -> str:
    p = edf_path.replace("\\", "/").lower()
    if "/abnormal" in p or "\\abnormal" in edf_path.lower() or "abnormal_" in p:
        return "abnormal"
    if "/normal" in p or "\\normal" in edf_path.lower() or "normal_" in p:
        return "normal"
    return "unknown"


def collect_header_text(edf_path: str) -> Dict[str, str]:
    """Extract searchable text fields from an EDF without preloading signal data."""
    import mne

    mne.set_log_level("ERROR")
    header_bytes = read_raw_edf_header_bytes(edf_path, 512)
    subject_info = ""
    description = ""
    meas_date = ""
    extra = []

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        subject_info = _safe_str(raw.info.get("subject_info", {}))
        description = _safe_str(raw.info.get("description", ""))
        meas_date = _safe_str(raw.info.get("meas_date", ""))
        # Some TUH fields land in these keys
        for key in ("his_id", "last_name", "first_name", "birthday", "sex", "hand"):
            si = raw.info.get("subject_info") or {}
            if isinstance(si, dict) and key in si:
                extra.append(_safe_str(si.get(key)))
        # Local patient / recording identification are in the ASCII header
        del raw
    except Exception as exc:
        extra.append(f"mne_error:{exc}")

    combined = " ".join(
        [header_bytes, subject_info, description, meas_date] + extra
    )
    # Collapse whitespace for cleaner keyword search / CSV storage
    combined_clean = re.sub(r"\s+", " ", combined).strip()
    return {
        "header_bytes": header_bytes,
        "subject_info": subject_info,
        "description": description,
        "meas_date": meas_date,
        "combined": combined_clean,
    }


def _find_hits(text: str, keywords: Iterable[str]) -> List[str]:
    text_l = text.lower()
    hits = []
    for kw in keywords:
        if kw.lower() in text_l:
            hits.append(kw)
    return hits


def classify_header(text: str, label: str) -> Dict[str, str]:
    text_l = text.lower()
    drug_hits = {
        drug: _find_hits(text_l, kws) for drug, kws in AD_DRUG_KEYWORDS.items()
    }
    has_done = bool(drug_hits["donepezil"])
    has_mem = bool(drug_hits["memantine"])
    has_other = bool(drug_hits["galantamine"] or drug_hits["rivastigmine"])

    if has_done and has_mem:
        drug_status = "both"
    elif has_done:
        drug_status = "donepezil"
    elif has_mem:
        drug_status = "memantine"
    elif has_other:
        drug_status = "other_ad_drug"
    elif len(text.strip()) < 40:
        drug_status = "unknown"
    else:
        drug_status = "no_ad_drug"

    diag_hits = _find_hits(text_l, AD_DIAGNOSIS_KEYWORDS)
    neuro_hits = _find_hits(text_l, NEURO_KEYWORDS)
    med_hits = _find_hits(text_l, GENERAL_MED_KEYWORDS)

    if diag_hits:
        clinical_status = "ad_diagnosed"
    elif label == "normal":
        clinical_status = "normal"
    elif label == "abnormal" or neuro_hits:
        clinical_status = "neurological"
    else:
        clinical_status = "unknown"

    med_snippets = []
    for hits in drug_hits.values():
        med_snippets.extend(hits)
    med_snippets.extend(med_hits)

    return {
        "drug_status": drug_status,
        "clinical_status": clinical_status,
        "medication_text": ";".join(sorted(set(med_snippets))),
        "diagnosis_text": ";".join(sorted(set(diag_hits + neuro_hits))),
    }


def mine_edf_headers(
    edf_paths: List[str],
    output_csv: Optional[str] = None,
) -> List[Dict]:
    """Mine headers for a list of EDF paths and optionally save CSV."""
    from tqdm import tqdm

    rows: List[Dict] = []
    for path in tqdm(edf_paths, desc="Mining EDF headers"):
        filename = os.path.basename(path)
        patient_id, session_id = parse_filename_ids(filename)
        label = infer_label_from_path(path)
        try:
            texts = collect_header_text(path)
            cls = classify_header(texts["combined"], label)
            row = {
                "filename": filename,
                "patient_id": patient_id,
                "session_id": session_id,
                "label": label,
                "drug_status": cls["drug_status"],
                "clinical_status": cls["clinical_status"],
                "medication_text": cls["medication_text"],
                "diagnosis_text": cls["diagnosis_text"],
                "header_text_length": len(texts["combined"]),
                "edf_path": path,
            }
        except Exception as exc:
            row = {
                "filename": filename,
                "patient_id": patient_id,
                "session_id": session_id,
                "label": label,
                "drug_status": "unknown",
                "clinical_status": "unknown",
                "medication_text": "",
                "diagnosis_text": "",
                "header_text_length": 0,
                "edf_path": path,
                "error": str(exc),
            }
        rows.append(row)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        fieldnames = [
            "filename",
            "patient_id",
            "session_id",
            "label",
            "drug_status",
            "clinical_status",
            "medication_text",
            "diagnosis_text",
            "header_text_length",
            "edf_path",
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    return rows


def summarize_header_mining(rows: List[Dict]) -> Dict:
    n = len(rows)
    drug_counts = {
        "donepezil": 0,
        "memantine": 0,
        "both": 0,
        "other_ad_drug": 0,
        "no_ad_drug": 0,
        "unknown": 0,
    }
    ad_diag = 0
    clinical_text = 0
    for r in rows:
        ds = r.get("drug_status", "unknown")
        if ds in drug_counts:
            drug_counts[ds] += 1
        else:
            drug_counts["unknown"] += 1
        if r.get("clinical_status") == "ad_diagnosed":
            ad_diag += 1
        if int(r.get("header_text_length", 0)) >= 40:
            clinical_text += 1

    ad_drug_mentions = (
        drug_counts["donepezil"]
        + drug_counts["memantine"]
        + drug_counts["both"]
        + drug_counts["other_ad_drug"]
    )
    return {
        "n_files": n,
        "files_with_clinical_text": clinical_text,
        "ad_drug_mentions": ad_drug_mentions,
        "donepezil_aricept": drug_counts["donepezil"] + drug_counts["both"],
        "memantine_namenda": drug_counts["memantine"] + drug_counts["both"],
        "other_ad_drugs": drug_counts["other_ad_drug"],
        "ad_diagnosis_mentions": ad_diag,
        "minimal_or_no_header_text": drug_counts["unknown"] + max(0, n - clinical_text),
        "drug_status_counts": drug_counts,
    }
