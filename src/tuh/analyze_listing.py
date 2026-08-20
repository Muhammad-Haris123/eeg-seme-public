"""
Analyze TUH EEG corpus directory listing to understand
dataset structure and plan targeted downloads.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# rsync list-only line:
# -rwxrwxr-x     17,543,622 2024/02/23 21:03:35 path/to/file.edf
LINE_RE = re.compile(
    r"^(?P<perms>[dl\-rwxstST]+)\s+"
    r"(?P<size>[\d,]+)\s+"
    r"(?P<date>\d{4}/\d{2}/\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<path>.+)$"
)

MAIN_EDF_RE = re.compile(
    r"^(?P<corpus>tuh_eeg)/(?P<version>v[\d.]+)/edf/"
    r"(?P<group>\d{3})/(?P<patient_id>[a-z]{8})/"
    r"(?P<session_id>s\d{3})_(?P<year>\d{4})/"
    r"(?P<montage>[\w]+)/(?P<filename>[a-z]{8}_s\d{3}_t\d{3}\.edf)$",
    re.IGNORECASE,
)

ABNORMAL_EDF_RE = re.compile(
    r"^(?P<corpus>tuh_eeg_abnormal)/(?P<version>v[\d.]+)/edf/"
    r"(?P<split>train|eval)/(?P<label>normal|abnormal)/"
    r"(?P<montage>[\w]+)/(?P<filename>(?P<patient_id>[a-z]{8})_(?P<session_id>s\d{3})_t(?P<segment>\d{3})\.edf)$",
    re.IGNORECASE,
)

SEIZURE_EDF_RE = re.compile(
    r"^(?P<corpus>tuh_eeg_seizure)/(?P<version>v[\d.]+)/edf/"
    r".*?/(?P<patient_id>[a-z]{8})/"
    r"(?P<session_id>s\d{3})_(?P<year>\d{4})/"
    r"(?P<montage>[\w]+)/(?P<filename>[a-z]{8}_s\d{3}_t\d{3}\.edf)$",
    re.IGNORECASE,
)

GENERIC_EDF_RE = re.compile(
    r"^(?P<corpus>[^/]+)/(?P<version>v[\d.]+)/.*?"
    r"(?P<patient_id>[a-z]{8})_(?P<session_id>s\d{3})_t(?P<segment>\d{3})\.edf$",
    re.IGNORECASE,
)

FILENAME_RE = re.compile(
    r"(?P<patient_id>[a-z]{8})_(?P<session_id>s\d{3})_t(?P<segment>\d{3})\.edf$",
    re.IGNORECASE,
)


def _parse_size(size_str: str) -> int:
    return int(size_str.replace(",", ""))


def _corpus_from_path(path: str) -> str:
    return path.split("/", 1)[0] if path else "unknown"


def parse_tuh_listing(listing_path: str | Path) -> dict:
    """
    Parse the rsync directory listing file (streaming, line-by-line).
    """
    listing_path = Path(listing_path)
    edf_files: List[dict] = []
    all_files: List[dict] = []
    patients: Set[str] = set()
    sessions_per_patient: Dict[str, Set[str]] = defaultdict(set)
    subcorpora: Counter = Counter()
    total_size = 0

    with listing_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("receiving ") or line.startswith("sent ") or line.startswith("total "):
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            perms = m.group("perms")
            path = m.group("path").strip()
            if path in (".",):
                continue
            is_dir = perms.startswith("d")
            size = _parse_size(m.group("size"))
            date = m.group("date")
            entry = {
                "full_path": path,
                "file_size_bytes": size,
                "date": date,
                "is_dir": is_dir,
                "perms": perms,
            }
            all_files.append(entry)

            if is_dir or not path.lower().endswith(".edf"):
                continue

            corpus = _corpus_from_path(path)
            subcorpora[corpus] += 1
            total_size += size

            info = {
                "full_path": path,
                "file_size_bytes": size,
                "date": date,
                "corpus": corpus,
                "version": None,
                "patient_id": None,
                "session_id": None,
                "year": None,
                "montage": None,
                "segment": None,
                "label": None,
                "split": None,
            }

            mm = MAIN_EDF_RE.match(path)
            if mm:
                info.update(
                    {
                        "version": mm.group("version"),
                        "patient_id": mm.group("patient_id").lower(),
                        "session_id": mm.group("session_id").lower(),
                        "year": int(mm.group("year")),
                        "montage": mm.group("montage"),
                        "segment": FILENAME_RE.search(mm.group("filename")).group("segment")
                        if FILENAME_RE.search(mm.group("filename"))
                        else None,
                    }
                )
            else:
                am = ABNORMAL_EDF_RE.match(path)
                if am:
                    info.update(
                        {
                            "version": am.group("version"),
                            "patient_id": am.group("patient_id").lower(),
                            "session_id": am.group("session_id").lower(),
                            "montage": am.group("montage"),
                            "segment": am.group("segment"),
                            "label": am.group("label").lower(),
                            "split": am.group("split").lower(),
                        }
                    )
                else:
                    sm = SEIZURE_EDF_RE.match(path)
                    if sm:
                        info.update(
                            {
                                "version": sm.group("version"),
                                "patient_id": sm.group("patient_id").lower(),
                                "session_id": sm.group("session_id").lower(),
                                "year": int(sm.group("year")),
                                "montage": sm.group("montage"),
                            }
                        )
                        fn = FILENAME_RE.search(path)
                        if fn:
                            info["segment"] = fn.group("segment")
                    else:
                        gm = GENERIC_EDF_RE.match(path)
                        if gm:
                            info.update(
                                {
                                    "version": gm.group("version"),
                                    "patient_id": gm.group("patient_id").lower(),
                                    "session_id": gm.group("session_id").lower(),
                                    "segment": gm.group("segment"),
                                }
                            )
                        # montage from path token if present
                        for tok in path.split("/"):
                            if tok.startswith("01_tcp") or tok.startswith("02_tcp") or tok.startswith("03_tcp"):
                                info["montage"] = tok
                            if tok in ("normal", "abnormal") and info["label"] is None:
                                info["label"] = tok
                            if tok in ("train", "eval") and info["split"] is None:
                                info["split"] = tok
                        # year from session folder like s001_2015
                        for tok in path.split("/"):
                            ym = re.match(r"s\d{3}_(\d{4})$", tok)
                            if ym:
                                info["year"] = int(ym.group(1))
                                break

            if info["patient_id"]:
                patients.add(info["patient_id"])
                if info["session_id"]:
                    sessions_per_patient[info["patient_id"]].add(info["session_id"])
            edf_files.append(info)

    sess_counts = {pid: len(s) for pid, s in sessions_per_patient.items()}
    return {
        "edf_files": edf_files,
        "all_files": all_files,
        "total_edf_count": len(edf_files),
        "total_size_gb": total_size / (1024**3),
        "subcorpora": dict(subcorpora),
        "patients": patients,
        "sessions_per_patient": sess_counts,
    }


def analyze_corpus_structure(parsed_data: dict) -> dict:
    """Compute detailed corpus statistics and print summary."""
    edfs: List[dict] = parsed_data["edf_files"]
    patients: Set[str] = parsed_data["patients"]
    sess = parsed_data["sessions_per_patient"]

    years = [e["year"] for e in edfs if e.get("year")]
    sizes = [e["file_size_bytes"] for e in edfs]
    montages = Counter(e.get("montage") or "unknown" for e in edfs)

    by_corpus: Dict[str, List[dict]] = defaultdict(list)
    for e in edfs:
        by_corpus[e["corpus"]].append(e)

    def _corpus_block(name: str) -> dict:
        items = by_corpus.get(name, [])
        pids = {e["patient_id"] for e in items if e.get("patient_id")}
        size_tb = sum(e["file_size_bytes"] for e in items) / (1024**4)
        sess_counts = []
        if name == "tuh_eeg":
            # sessions among patients in this corpus only
            for pid in pids:
                sess_counts.append(sess.get(pid, 0))
        years_c = Counter(e["year"] for e in items if e.get("year"))
        mont_c = Counter(e.get("montage") or "unknown" for e in items)
        out = {
            "n_files": len(items),
            "n_patients": len(pids),
            "size_tb": size_tb,
            "years": dict(sorted(years_c.items())),
            "montages": dict(mont_c),
        }
        if sess_counts:
            out["sessions_per_patient"] = {
                "mean": float(statistics.mean(sess_counts)),
                "median": float(statistics.median(sess_counts)),
                "max": int(max(sess_counts)),
            }
        if name == "tuh_eeg_abnormal":
            labels = Counter(e.get("label") or "unknown" for e in items)
            splits = Counter(e.get("split") or "unknown" for e in items)
            out["labels"] = dict(labels)
            out["splits"] = dict(splits)
            out["patients_per_label"] = {
                lab: len({e["patient_id"] for e in items if e.get("label") == lab and e.get("patient_id")})
                for lab in labels
            }
        return out

    multi_2 = sum(1 for v in sess.values() if v >= 2)
    multi_5 = sum(1 for v in sess.values() if v >= 5)
    multi_10 = sum(1 for v in sess.values() if v >= 10)
    max_sess = max(sess.values()) if sess else 0
    max_pid = max(sess, key=sess.get) if sess else None

    n_ar = montages.get("01_tcp_ar", 0)
    n_le = montages.get("02_tcp_le", 0)
    n_total = max(len(edfs), 1)
    n_gt5 = sum(1 for s in sizes if s > 5 * 1024 * 1024)
    n_lt3 = sum(1 for s in sizes if s < 3 * 1024 * 1024)
    year_hist = Counter(years)

    known = {"tuh_eeg", "tuh_eeg_abnormal", "tuh_eeg_seizure"}
    other_n = sum(len(v) for k, v in by_corpus.items() if k not in known)

    analysis = {
        "overall": {
            "total_edf_files": len(edfs),
            "total_unique_patients": len(patients),
            "total_size_tb": parsed_data["total_size_gb"] / 1024.0,
            "total_size_gb": parsed_data["total_size_gb"],
            "date_range_years": [min(years), max(years)] if years else None,
            "subcorpora_counts": parsed_data["subcorpora"],
        },
        "per_subcorpus": {
            "tuh_eeg": _corpus_block("tuh_eeg"),
            "tuh_eeg_abnormal": _corpus_block("tuh_eeg_abnormal"),
            "tuh_eeg_seizure": _corpus_block("tuh_eeg_seizure"),
            "other_files": other_n,
        },
        "multi_session": {
            "ge_2": multi_2,
            "ge_5": multi_5,
            "ge_10": multi_10,
            "max_sessions": max_sess,
            "max_sessions_patient": max_pid,
        },
        "montage": {
            "01_tcp_ar": n_ar,
            "02_tcp_le": n_le,
            "other": n_total - n_ar - n_le,
            "pct_ar": 100.0 * n_ar / n_total,
            "pct_le": 100.0 * n_le / n_total,
        },
        "file_sizes": {
            "mean_mb": float(statistics.mean(sizes) / 1e6) if sizes else 0.0,
            "median_mb": float(statistics.median(sizes) / 1e6) if sizes else 0.0,
            "gt_5mb": n_gt5,
            "gt_5mb_pct": 100.0 * n_gt5 / n_total,
            "lt_3mb": n_lt3,
            "lt_3mb_pct": 100.0 * n_lt3 / n_total,
        },
        "year_distribution": dict(sorted(year_hist.items())),
    }

    o = analysis["overall"]
    main = analysis["per_subcorpus"]["tuh_eeg"]
    abn = analysis["per_subcorpus"]["tuh_eeg_abnormal"]
    seiz = analysis["per_subcorpus"]["tuh_eeg_seizure"]
    print("=" * 60)
    print("TUH EEG CORPUS ANALYSIS (from directory listing)")
    print("=" * 60)
    print(f"Total EDF files:           {o['total_edf_files']}")
    print(f"Total unique patients:     {o['total_unique_patients']}")
    print(f"Total data size:           {o['total_size_tb']:.2f} TB")
    if o["date_range_years"]:
        print(f"Recording date range:      {o['date_range_years'][0]} - {o['date_range_years'][1]}")
    print()
    print("Sub-corpus breakdown:")
    print(f"  tuh_eeg (main):          {main['n_files']} files, {main['n_patients']} patients, {main['size_tb']:.2f} TB")
    labs = abn.get("labels", {})
    print(
        f"  tuh_eeg_abnormal:        {abn['n_files']} files "
        f"({labs.get('normal', 0)} normal, {labs.get('abnormal', 0)} abnormal)"
    )
    print(f"  tuh_eeg_seizure:         {seiz['n_files']} files, {seiz['n_patients']} patients")
    print(f"  Other:                   {other_n} files")
    print()
    print("Multi-session patients:")
    print(f"  2+ sessions:             {multi_2} patients")
    print(f"  5+ sessions:             {multi_5} patients")
    print(f"  10+ sessions:            {multi_10} patients")
    print(f"  Max sessions:            {max_sess} (patient {max_pid})")
    print()
    print("Montage distribution:")
    print(f"  01_tcp_ar (avg ref):     {n_ar} files ({analysis['montage']['pct_ar']:.1f}%)")
    print(f"  02_tcp_le (linked ear):  {n_le} files ({analysis['montage']['pct_le']:.1f}%)")
    print()
    fs = analysis["file_sizes"]
    print("File sizes:")
    print(f"  Mean:                    {fs['mean_mb']:.1f} MB")
    print(f"  Median:                  {fs['median_mb']:.1f} MB")
    print(f"  Files > 5MB (usable):   {fs['gt_5mb']} ({fs['gt_5mb_pct']:.1f}%)")
    print(f"  Files < 3MB (too short): {fs['lt_3mb']} ({fs['lt_3mb_pct']:.1f}%)")
    print()
    print("Recording year distribution:")
    for y, c in sorted(year_hist.items()):
        print(f"  {y}: {c} files")
    print("=" * 60)
    return analysis


def _usable(e: dict, prefer_ar: bool = True) -> bool:
    if e["file_size_bytes"] <= 5 * 1024 * 1024:
        return False
    mont = e.get("montage") or ""
    if prefer_ar:
        return mont == "01_tcp_ar" or mont == "02_tcp_le"
    return True


def _mb(nbytes: int) -> float:
    return nbytes / (1024**2)


def identify_download_candidates(parsed_data: dict, output_dir: str | Path | None = None) -> dict:
    """
    Identify best EDF files to download (~200 files target).
    """
    edfs = parsed_data["edf_files"]
    sess = parsed_data["sessions_per_patient"]
    output_dir = Path(output_dir) if output_dir else Path("data/tuh_mining")
    output_dir.mkdir(parents=True, exist_ok=True)

    abnormal = [
        e
        for e in edfs
        if e["corpus"] == "tuh_eeg_abnormal" and e.get("label") == "abnormal" and _usable(e)
    ]
    normal = [
        e
        for e in edfs
        if e["corpus"] == "tuh_eeg_abnormal" and e.get("label") == "normal" and _usable(e)
    ]
    main = [e for e in edfs if e["corpus"] == "tuh_eeg" and _usable(e) and e.get("patient_id")]

    # Prefer AR montage first
    def _sort_key(e):
        ar_rank = 0 if e.get("montage") == "01_tcp_ar" else 1
        return (ar_rank, -e["file_size_bytes"], e.get("full_path", ""))

    abnormal = sorted(abnormal, key=_sort_key)
    normal = sorted(normal, key=_sort_key)

    # Tier 3: multi-session patients (3+), first session only
    multi_pids = {pid for pid, n in sess.items() if n >= 3}
    by_patient: Dict[str, List[dict]] = defaultdict(list)
    for e in main:
        if e["patient_id"] in multi_pids:
            by_patient[e["patient_id"]].append(e)
    tier3 = []
    for pid, items in by_patient.items():
        # pick earliest session file (by session_id then path)
        items_sorted = sorted(items, key=lambda x: (x.get("session_id") or "", x.get("year") or 9999, x["full_path"]))
        # first session group
        first_sess = items_sorted[0]["session_id"]
        first_files = [x for x in items_sorted if x["session_id"] == first_sess]
        first_files = sorted(first_files, key=_sort_key)
        tier3.append(first_files[0])
    tier3 = sorted(tier3, key=_sort_key)

    selected = []

    def _add(items: List[dict], tier: str, limit: int):
        count = 0
        seen_paths = {s["full_remote_path"] for s in selected}
        for e in items:
            if count >= limit:
                break
            if e["full_path"] in seen_paths:
                continue
            selected.append(
                {
                    "patient_id": e.get("patient_id"),
                    "session_id": e.get("session_id"),
                    "full_remote_path": e["full_path"],
                    "file_size_mb": round(_mb(e["file_size_bytes"]), 2),
                    "selection_tier": tier,
                    "montage": e.get("montage"),
                    "label": e.get("label"),
                    "split": e.get("split"),
                    "year": e.get("year"),
                    "corpus": e.get("corpus"),
                    "version": e.get("version"),
                }
            )
            seen_paths.add(e["full_path"])
            count += 1

    _add(abnormal, "tier1_abnormal", 100)
    _add(normal, "tier2_normal", 50)
    _add(tier3, "tier3_multisession", 50)

    # Summaries
    def _tier_stats(tier: str) -> dict:
        rows = [r for r in selected if r["selection_tier"] == tier]
        return {
            "n_files": len(rows),
            "size_gb": round(sum(r["file_size_mb"] for r in rows) / 1024.0, 3),
        }

    summary = {
        "tier1_abnormal": _tier_stats("tier1_abnormal"),
        "tier2_normal": _tier_stats("tier2_normal"),
        "tier3_multisession": _tier_stats("tier3_multisession"),
        "total": {
            "n_files": len(selected),
            "size_gb": round(sum(r["file_size_mb"] for r in selected) / 1024.0, 3),
        },
    }
    # ~10 Mbps = 1.25 MB/s
    summary["estimated_minutes_at_10mbps"] = round(
        (summary["total"]["size_gb"] * 1024) / 1.25 / 60.0, 1
    )

    # CSV
    csv_path = output_dir / "download_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "selection_tier",
                "patient_id",
                "session_id",
                "label",
                "split",
                "montage",
                "year",
                "file_size_mb",
                "corpus",
                "version",
                "full_remote_path",
            ],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(row)

    (output_dir / "download_candidates_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Remote layout for this account nests corpora under data/tuh_eeg/
    # Listing paths like tuh_eeg_abnormal/... map to data/tuh_eeg/tuh_eeg_abnormal/...
    remote_root = "data/tuh_eeg"

    # download_selected.sh — one rsync per file (targeted)
    sel_lines = [
        "#!/bin/bash",
        "# TUH EEG selected download — targeted files only",
        f"# Total: ~{summary['total']['n_files']} files, ~{summary['total']['size_gb']} GB estimated",
        "# Correct remote root: data/tuh_eeg/ (listing paths are relative to that)",
        'DEST="/mnt/c/Users/UC/Desktop/my_fyp/tug_eeg_corpus/selected_edfs"',
        'mkdir -p "$DEST"',
        'SSH_OPTS=\'ssh -i ~/.ssh/id_ed25519\'',
        'REMOTE="nedc-tuh-eeg@www.isip.piconepress.com"',
        f'ROOT="{remote_root}"',
        "",
    ]
    for row in selected:
        remote = row["full_remote_path"]
        tier_dir = row["selection_tier"]
        sel_lines.append(f'mkdir -p "$DEST/{tier_dir}"')
        sel_lines.append(
            f'rsync -avL -e "$SSH_OPTS" "$REMOTE:$ROOT/{remote}" "$DEST/{tier_dir}/" || echo "FAILED: {remote}"'
        )
    sel_lines.append("")
    sel_lines.append(f'echo "Downloaded selected files to $DEST/"')
    sel_path = output_dir / "download_selected.sh"
    sel_path.write_text("\n".join(sel_lines) + "\n", encoding="utf-8")

    # alternative: try abnormal directory bulk (may bypass symlink issue)
    alt_lines = [
        "#!/bin/bash",
        "# Alternative: try tuh_eeg_abnormal under corrected remote root",
        "# Listing paths nest under data/tuh_eeg/ for this account",
        'DEST="/mnt/c/Users/UC/Desktop/my_fyp/tug_eeg_corpus/selected_edfs"',
        'mkdir -p "$DEST"',
        'SSH_OPTS=\'ssh -i ~/.ssh/id_ed25519\'',
        'REMOTE="nedc-tuh-eeg@www.isip.piconepress.com"',
        f'ROOT="{remote_root}"',
        "",
        'echo "Trying tuh_eeg_abnormal eval/abnormal AR montage directory..."',
        'mkdir -p "$DEST/abnormal_eval"',
        'rsync -avL -e "$SSH_OPTS" \\',
        '  "$REMOTE:$ROOT/tuh_eeg_abnormal/v3.0.1/edf/eval/abnormal/01_tcp_ar/" \\',
        '  "$DEST/abnormal_eval/" \\',
        '  && echo "SUCCESS: abnormal eval AR directory downloaded" \\',
        '  || echo "FAILED: abnormal eval AR directory"',
        "",
        'echo "Trying a small sample of individual abnormal files..."',
        'mkdir -p "$DEST/abnormal_sample"',
    ]
    # add up to 5 tier1 files as individual probes
    n_probe = 0
    for row in selected:
        if row["selection_tier"] != "tier1_abnormal":
            continue
        alt_lines.append(
            f'rsync -avL -e "$SSH_OPTS" "$REMOTE:$ROOT/{row["full_remote_path"]}" "$DEST/abnormal_sample/" '
            f'&& echo "OK: {row["full_remote_path"]}" || echo "FAIL: {row["full_remote_path"]}"'
        )
        n_probe += 1
        if n_probe >= 5:
            break
    alt_lines.extend(
        [
            "",
            'echo "Check if any files downloaded. If yes, the abnormal subcorpus works!"',
            'find "$DEST" -type f -name "*.edf" | wc -l',
            'du -sh "$DEST" 2>/dev/null || true',
        ]
    )
    alt_path = output_dir / "download_alternative.sh"
    alt_path.write_text("\n".join(alt_lines) + "\n", encoding="utf-8")

    # Make executable where possible
    try:
        os.chmod(sel_path, 0o755)
        os.chmod(alt_path, 0o755)
    except OSError:
        pass

    print("=" * 60)
    print("DOWNLOAD PLAN")
    print("=" * 60)
    print(
        f"Tier 1 (abnormal EEGs):    {summary['tier1_abnormal']['n_files']} files, "
        f"~{summary['tier1_abnormal']['size_gb']:.2f} GB"
    )
    print(
        f"Tier 2 (normal controls):  {summary['tier2_normal']['n_files']} files, "
        f"~{summary['tier2_normal']['size_gb']:.2f} GB"
    )
    print(
        f"Tier 3 (multi-session):    {summary['tier3_multisession']['n_files']} files, "
        f"~{summary['tier3_multisession']['size_gb']:.2f} GB"
    )
    print(f"Total:                     {summary['total']['n_files']} files, ~{summary['total']['size_gb']:.2f} GB")
    print()
    print(f"Estimated download time at 10 Mbps: {summary['estimated_minutes_at_10mbps']} minutes")
    print(f"Download scripts saved to: {output_dir}/")
    print()
    print("NEXT STEPS:")
    print("1. Use ROOT=data/tuh_eeg (already baked into scripts)")
    print("2. Run: bash data/tuh_mining/download_alternative.sh")
    print("   or full set: bash data/tuh_mining/download_selected.sh")
    print("3. After download: python run_tuh_pipeline.py --stage 3")
    print("=" * 60)

    return {"selected": selected, "summary": summary}


def extract_edf_header_info_from_listing(parsed_data: dict) -> dict:
    """Identify supplementary clinical/metadata files in the listing."""
    all_files = parsed_data.get("all_files", [])
    interesting_ext = {".txt", ".csv", ".lbl", ".tsv", ".json", ".md", ".pdf", ".gz", ".html"}
    keywords = ("readme", "docs", "header", "clinical", "report", "annotation", "label")

    hits = []
    for e in all_files:
        if e.get("is_dir"):
            continue
        path = e["full_path"]
        lower = path.lower()
        ext = Path(lower).suffix
        if ext in interesting_ext or any(k in lower for k in keywords):
            # skip huge raw archives unless docs/headers
            hits.append(
                {
                    "path": path,
                    "size_bytes": e["file_size_bytes"],
                    "size_mb": round(_mb(e["file_size_bytes"]), 2),
                    "date": e.get("date"),
                }
            )

    # Categorize
    docs = [h for h in hits if "/docs/" in h["path"].lower() or "readme" in h["path"].lower()]
    txt = [h for h in hits if h["path"].lower().endswith(".txt")]
    lbl = [h for h in hits if h["path"].lower().endswith(".lbl")]
    csvs = [h for h in hits if h["path"].lower().endswith(".csv")]
    headers = [h for h in hits if "header" in h["path"].lower()]

    out = {
        "n_supplementary_files": len(hits),
        "n_docs_readme": len(docs),
        "n_txt": len(txt),
        "n_lbl": len(lbl),
        "n_csv": len(csvs),
        "n_header_related": len(headers),
        "examples_docs": docs[:20],
        "examples_txt": txt[:20],
        "examples_lbl": lbl[:20],
        "examples_csv": csvs[:20],
        "examples_headers": headers[:20],
    }

    print(f"Supplementary-like files found: {len(hits)}")
    print(f"  Docs/README: {len(docs)}")
    print(f"  .txt: {len(txt)}")
    print(f"  .lbl: {len(lbl)}")
    print(f"  .csv: {len(csvs)}")
    print(f"  header-related: {len(headers)}")
    if headers:
        print("  Header examples:")
        for h in headers[:5]:
            print(f"    {h['path']} ({h['size_mb']} MB)")
    if docs:
        print("  Docs/README examples:")
        for h in docs[:5]:
            print(f"    {h['path']} ({h['size_mb']} MB)")
    return out
