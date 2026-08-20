"""
ADNI clinical cohort builder for Alzheimer's disease subjects.

Merges diagnosis, medications, FreeSurfer hippocampal volumes, and
neuropsychological composites into a longitudinal master cohort table.

Author: Research Team
Date: 2026
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.config import DATA_ROOT, PROJECT_ROOT

ADNI_DATA_DIR = PROJECT_ROOT / "adni_data"

DX_CSV = ADNI_DATA_DIR / "DXSUM_21Apr2026.csv"
MEDS_CSV = ADNI_DATA_DIR / "All_Subjects_RECCMEDS_20Apr2026.csv"
FS_CSV = ADNI_DATA_DIR / "UCSFFSX7_20Apr2026.csv"
UW_CSV = ADNI_DATA_DIR / "neuropsych" / "UWNPSYCHSUM_20Apr2026.csv"
MOCA_CSV = ADNI_DATA_DIR / "neuropsych" / "MOCA_20Apr2026.csv"
FAQ_CSV = ADNI_DATA_DIR / "neuropsych" / "FAQ_20Apr2026.csv"

COHORT_CSV = DATA_ROOT / "adni_cohort.csv"
SUMMARY_JSON = DATA_ROOT / "adni_cohort_summary.json"

DONEPEZIL_PATTERN = r"DONEPEZIL|ARICEPT"
MEMANTINE_PATTERN = r"MEMANTINE|NAMENDA|NAMZARIC"

DIAGNOSIS_MAP = {1: "Normal", 2: "MCI", 3: "AD"}


def _normalize_viscode(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()


def _read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required ADNI file not found: {path}")
    return pd.read_csv(path, low_memory=False, usecols=usecols)


def load_adni_tables() -> dict[str, pd.DataFrame]:
    """Load all required ADNI tables from adni_data/."""
    dx = _read_csv(
        DX_CSV,
        usecols=["RID", "PTID", "VISCODE", "VISCODE2", "EXAMDATE", "DIAGNOSIS"],
    )
    meds = _read_csv(MEDS_CSV, usecols=["RID", "PTID", "VISCODE", "CMMED"])
    fs = _read_csv(
        FS_CSV,
        usecols=["RID", "PTID", "VISCODE", "VISCODE2", "EXAMDATE", "ST29SV", "ST88SV"],
    )
    uw = _read_csv(
        UW_CSV,
        usecols=["RID", "VISCODE", "VISCODE2", "EXAMDATE", "ADNI_MEM", "ADNI_EF", "ADNI_LAN", "ADNI_VS"],
    )
    moca = _read_csv(MOCA_CSV, usecols=["RID", "PTID", "VISCODE", "VISCODE2", "MOCA"])
    faq = _read_csv(FAQ_CSV, usecols=["RID", "PTID", "VISCODE", "VISCODE2", "FAQTOTAL"])

    for df in (dx, meds, fs, uw, moca, faq):
        if "VISCODE" in df.columns:
            df["VISCODE_NORM"] = _normalize_viscode(df["VISCODE"])
        if "VISCODE2" in df.columns:
            df["VISCODE2_NORM"] = _normalize_viscode(df["VISCODE2"])

    return {"dx": dx, "meds": meds, "fs": fs, "uw": uw, "moca": moca, "faq": faq}


def classify_medication_status(meds: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each RID by AD medication exposure.

    Returns a DataFrame with columns: RID, on_donepezil, on_memantine, drug_status
    (drug_status filled later for AD-only subjects; here we store flags).
    """
    cmmed = meds["CMMED"].astype(str).str.upper()
    done_rids = set(
        meds.loc[cmmed.str.contains(DONEPEZIL_PATTERN, na=False, regex=True), "RID"].unique()
    )
    mem_rids = set(
        meds.loc[cmmed.str.contains(MEMANTINE_PATTERN, na=False, regex=True), "RID"].unique()
    )

    all_rids = sorted(done_rids | mem_rids | set(meds["RID"].dropna().unique()))
    rows = []
    for rid in all_rids:
        on_d = rid in done_rids
        on_m = rid in mem_rids
        if on_d and on_m:
            status = "combination"
        elif on_d:
            status = "donepezil"
        elif on_m:
            status = "memantine"
        else:
            status = "no_drug"
        rows.append(
            {
                "RID": rid,
                "on_donepezil": on_d,
                "on_memantine": on_m,
                "drug_status": status,
            }
        )
    return pd.DataFrame(rows)


def _score_at_visit(df: pd.DataFrame, rid: int, visit: str, col: str) -> float:
    """Return first non-null score for RID at VISCODE == visit."""
    subset = df.loc[
        (df["RID"] == rid) & (df["VISCODE_NORM"] == visit),
        col,
    ]
    if subset.empty:
        return np.nan
    vals = pd.to_numeric(subset, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.iloc[0])


def _hippocampal_volumes(fs: pd.DataFrame, rid: int) -> tuple[float, float, float]:
    """
    Baseline = earliest EXAMDATE with valid volumes;
    latest = most recent EXAMDATE with valid volumes.
    Returns (baseline, latest, atrophy_rate).
    """
    sub = fs.loc[fs["RID"] == rid].copy()
    if sub.empty:
        return np.nan, np.nan, np.nan

    sub["ST29SV"] = pd.to_numeric(sub["ST29SV"], errors="coerce")
    sub["ST88SV"] = pd.to_numeric(sub["ST88SV"], errors="coerce")
    sub["hippo"] = sub["ST29SV"] + sub["ST88SV"]
    sub["EXAMDATE"] = pd.to_datetime(sub["EXAMDATE"], errors="coerce")
    sub = sub.dropna(subset=["hippo", "EXAMDATE"]).sort_values("EXAMDATE")
    if sub.empty:
        return np.nan, np.nan, np.nan

    baseline = float(sub["hippo"].iloc[0])
    latest = float(sub["hippo"].iloc[-1])
    if baseline == 0 or np.isnan(baseline):
        rate = np.nan
    else:
        rate = (latest - baseline) / baseline
    # Single scan → no meaningful atrophy rate
    if len(sub) < 2:
        rate = np.nan
    return baseline, latest, rate


def build_ad_cohort(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """
    Build master AD cohort DataFrame for subjects with DIAGNOSIS=3 at VISCODE='bl'.
    """
    if tables is None:
        tables = load_adni_tables()

    dx = tables["dx"]
    meds = tables["meds"]
    fs = tables["fs"]
    uw = tables["uw"]
    moca = tables["moca"]
    # FAQ loaded for completeness / future extension; not required in master columns
    _ = tables["faq"]

    med_status = classify_medication_status(meds)
    med_lookup = med_status.set_index("RID")

    ad_bl = (
        dx.loc[(dx["DIAGNOSIS"] == 3) & (dx["VISCODE_NORM"] == "bl"), ["RID", "PTID", "DIAGNOSIS"]]
        .drop_duplicates(subset=["RID"], keep="first")
        .copy()
    )

    # PTID fallback from other tables if missing
    ptid_map = (
        pd.concat(
            [
                dx[["RID", "PTID"]],
                meds[["RID", "PTID"]],
                fs[["RID", "PTID"]],
                moca[["RID", "PTID"]],
            ],
            ignore_index=True,
        )
        .dropna(subset=["PTID"])
        .drop_duplicates(subset=["RID"], keep="first")
        .set_index("RID")["PTID"]
    )

    n_visits = uw.groupby("RID")["VISCODE_NORM"].nunique()

    rows: list[dict[str, Any]] = []
    for _, subj in ad_bl.iterrows():
        rid = int(subj["RID"])
        ptid = subj["PTID"]
        if pd.isna(ptid) and rid in ptid_map.index:
            ptid = ptid_map.loc[rid]

        if rid in med_lookup.index:
            drug_status = med_lookup.loc[rid, "drug_status"]
        else:
            drug_status = "no_drug"

        mem_bl = _score_at_visit(uw, rid, "bl", "ADNI_MEM")
        mem_m12 = _score_at_visit(uw, rid, "m12", "ADNI_MEM")
        ef_bl = _score_at_visit(uw, rid, "bl", "ADNI_EF")
        ef_m12 = _score_at_visit(uw, rid, "m12", "ADNI_EF")

        mem_change = (
            mem_m12 - mem_bl
            if not (np.isnan(mem_bl) or np.isnan(mem_m12))
            else np.nan
        )
        ef_change = (
            ef_m12 - ef_bl
            if not (np.isnan(ef_bl) or np.isnan(ef_m12))
            else np.nan
        )

        hippo_bl, hippo_latest, hippo_rate = _hippocampal_volumes(fs, rid)
        moca_bl = _score_at_visit(moca, rid, "bl", "MOCA")

        rows.append(
            {
                "RID": rid,
                "PTID": ptid,
                "diagnosis": DIAGNOSIS_MAP.get(int(subj["DIAGNOSIS"]), "AD"),
                "drug_status": drug_status,
                # Demographics not present in currently available ADNI extracts
                "age": np.nan,
                "sex": np.nan,
                "education": np.nan,
                "adni_mem_baseline": mem_bl,
                "adni_mem_12m": mem_m12,
                "adni_mem_change": mem_change,
                "adni_ef_baseline": ef_bl,
                "adni_ef_12m": ef_m12,
                "adni_ef_change": ef_change,
                "hippo_vol_baseline": hippo_bl,
                "hippo_vol_latest": hippo_latest,
                "hippo_atrophy_rate": hippo_rate,
                "n_cognitive_visits": int(n_visits.get(rid, 0)),
                "moca_baseline": moca_bl,
            }
        )

    cohort = pd.DataFrame(rows)
    cohort = cohort.sort_values("RID").reset_index(drop=True)
    return cohort


def compute_summary(cohort: pd.DataFrame) -> dict[str, Any]:
    """Compute per-drug-group counts and mean cognitive (ADNI_MEM) change."""

    def _group_stats(status: str) -> dict[str, Any]:
        g = cohort.loc[cohort["drug_status"] == status]
        changes = pd.to_numeric(g["adni_mem_change"], errors="coerce")
        return {
            "n_subjects": int(len(g)),
            "n_with_cognitive_change": int(changes.notna().sum()),
            "mean_adni_mem_change": float(changes.mean()) if changes.notna().any() else None,
            "mean_adni_ef_change": float(
                pd.to_numeric(g["adni_ef_change"], errors="coerce").mean()
            )
            if pd.to_numeric(g["adni_ef_change"], errors="coerce").notna().any()
            else None,
            "mean_hippo_atrophy_rate": float(
                pd.to_numeric(g["hippo_atrophy_rate"], errors="coerce").mean()
            )
            if pd.to_numeric(g["hippo_atrophy_rate"], errors="coerce").notna().any()
            else None,
        }

    longitudinal = cohort["adni_mem_change"].notna() | cohort["adni_ef_change"].notna()
    on_drug = cohort["drug_status"].isin(["donepezil", "memantine", "combination"])

    summary: dict[str, Any] = {
        "total_ad_baseline": int(len(cohort)),
        "ad_on_ad_drugs": int(on_drug.sum()),
        "total_ad_with_longitudinal_data": int(longitudinal.sum()),
        "groups": {
            "donepezil": _group_stats("donepezil"),
            "memantine": _group_stats("memantine"),
            "combination": _group_stats("combination"),
            "no_drug": _group_stats("no_drug"),
        },
        "notes": {
            "diagnosis_filter": "DIAGNOSIS==3 at VISCODE=='bl'",
            "cognitive_change": "score_12m - score_baseline (ADNI_MEM / ADNI_EF)",
            "hippocampal_volume": "ST29SV + ST88SV; atrophy_rate=(latest-baseline)/baseline",
            "demographics": "age/sex/education unavailable in current adni_data extracts",
        },
    }
    return summary


def format_summary_table(summary: dict[str, Any]) -> str:
    """Pretty-print cohort summary for the console."""
    groups = summary["groups"]

    def _line(label: str, key: str) -> str:
        g = groups[key]
        mean_c = g["mean_adni_mem_change"]
        mean_str = f"{mean_c:.4f}" if mean_c is not None else "N/A"
        return (
            f"{label:<24} {g['n_subjects']:>4} subjects, "
            f"mean cognitive change (ADNI_MEM) = {mean_str}"
        )

    lines = [
        "=== ADNI COHORT SUMMARY ===",
        _line("AD + Donepezil only:", "donepezil"),
        _line("AD + Memantine only:", "memantine"),
        _line("AD + Both drugs:", "combination"),
        _line("AD + No AD drug:", "no_drug"),
        f"Total AD with longitudinal data: {summary['total_ad_with_longitudinal_data']}",
        f"Total AD at baseline:            {summary['total_ad_baseline']}",
        f"AD subjects on AD drugs:         {summary['ad_on_ad_drugs']}",
    ]
    return "\n".join(lines)


def save_cohort(
    cohort: pd.DataFrame,
    summary: dict[str, Any],
    cohort_path: Path | None = None,
    summary_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write cohort CSV and summary JSON under data/."""
    cohort_path = cohort_path or COHORT_CSV
    summary_path = summary_path or SUMMARY_JSON
    cohort_path.parent.mkdir(parents=True, exist_ok=True)

    cohort.to_csv(cohort_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return cohort_path, summary_path


def build_and_save() -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Full pipeline: load → build → save → return cohort, summary, printed table."""
    tables = load_adni_tables()
    cohort = build_ad_cohort(tables)
    summary = compute_summary(cohort)
    save_cohort(cohort, summary)
    table = format_summary_table(summary)
    return cohort, summary, table


if __name__ == "__main__":
    _, _, table = build_and_save()
    print(table)
    print(f"\nSaved: {COHORT_CSV}")
    print(f"Saved: {SUMMARY_JSON}")
