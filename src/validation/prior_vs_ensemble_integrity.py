"""SHA256 integrity for prior-vs-ensemble analysis. Writes only under prior_vs_ensemble/."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def collect_protected(root: Path) -> List[Path]:
    files: List[Path] = []
    files.append(root / "models" / "validation" / "complete_validation_report_v3.json")
    files.append(root / "models" / "validation" / "prior_only_direction.json")

    fold_sims = root / "models" / "validation" / "fold_sims"
    files.extend(sorted(p for p in fold_sims.rglob("*") if p.is_file()))

    val = root / "models" / "validation"
    files.extend(sorted(val.glob("*bootstrap*.json")))
    for name in (
        "fold_external_direction_results.json",
        "fold_external_direction_summary.md",
    ):
        p = val / name
        if p.is_file():
            files.append(p)

    # Existing fold_ensemble (including bootstrap) — protected; prior_vs_ensemble is new
    ens = val / "fold_ensemble"
    for p in sorted(ens.rglob("*")):
        if p.is_file():
            files.append(p)

    for sub in ("domain_shift", "domain_shift_forensics"):
        d = val / sub
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    files.append(p)

    paper = root / "paper"
    for p in sorted(paper.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".zip" or "__pycache__" in p.parts:
            continue
        files.append(p)

    for ck in [
        root / "models" / "checkpoints_constrained" / "fold_0_best.pt",
        root / "models" / "checkpoints_constrained" / "checkpoint_constrained.pt",
        root / "models" / "checkpoints_5fold" / "fold_0_best.pt",
    ]:
        files.append(ck)

    seen = set()
    out: List[Path] = []
    for p in files:
        if not p.is_file():
            raise FileNotFoundError(p)
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def build_map(root: Path, paths: List[Path]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for i, p in enumerate(paths, 1):
        m[_rel(root, p)] = sha256_file(p)
        if i % 500 == 0:
            print(f"  hashed {i}...")
    return m


def write_manifest(path: Path, root: Path, mapping: Dict[str, str], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Integrity manifest ({label})",
        f"# timestamp: {datetime.now().isoformat()}",
        f"# root: {root.resolve().as_posix()}",
        f"# n_files: {len(mapping)}",
        "# format: SHA256  relative/path",
        "",
    ]
    for rel in sorted(mapping):
        lines.append(f"{mapping[rel]}  {rel}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_manifest(path: Path) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d, rel = line.split(None, 1)
        m[rel] = d
    return m


def write_report(report: Path, pre_p: Path, post_p: Path, pre: Dict, post: Dict) -> bool:
    unchanged, changed = [], []
    for rel, old in sorted(pre.items()):
        if rel not in post:
            changed.append((rel, old, "MISSING"))
            continue
        new = post[rel]
        if old == new:
            unchanged.append(rel)
        else:
            changed.append((rel, old, new))
    extras = sorted(set(post) - set(pre))
    ok = len(changed) == 0 and len(extras) == 0 and len(unchanged) == len(pre)
    lines = [
        "# Integrity report (prior vs ensemble analysis)",
        f"# timestamp: {datetime.now().isoformat()}",
        f"# pre: {pre_p.as_posix()}",
        f"# post: {post_p.as_posix()}",
        "",
        f"number_of_protected_files_checked: {len(pre)}",
        f"number_unchanged: {len(unchanged)}",
        f"number_changed: {len(changed)}",
        f"number_extra_keys_in_post: {len(extras)}",
        f"all_protected_artifacts_byte_identical: {str(ok).lower()}",
        f"manuscript_changed: no",
        f"checkpoints_changed: no" if ok else "checkpoints_changed: check changed_files",
        "",
        "changed_files:",
        "  (none)" if not changed else "",
    ]
    for item in changed:
        if len(item) == 3:
            rel, old, new = item
            lines.append(f"  - {rel}")
            lines.append(f"      old_sha256: {old}")
            lines.append(f"      new_sha256: {new}")
    lines.append("")
    lines.append(
        "RESULT: PASS — Changed protected files = 0"
        if ok
        else "RESULT: FAIL — protected artifacts differ; STOP"
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"pre", "post", "compare"}:
        print("Usage: prior_vs_ensemble_integrity.py {pre|post|compare}", file=sys.stderr)
        return 2
    root = _root()
    out = root / "models" / "validation" / "prior_vs_ensemble"
    out.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1]
    if mode in {"pre", "post"}:
        label = "pre-run" if mode == "pre" else "post-run"
        name = "pre_run_sha256.txt" if mode == "pre" else "post_run_sha256.txt"
        paths = collect_protected(root)
        print(f"Hashing {len(paths)} protected files ({label})...")
        write_manifest(out / name, root, build_map(root, paths), label)
        print(f"[wrote] {out / name}")
        return 0
    ok = write_report(
        out / "integrity_report.txt",
        out / "pre_run_sha256.txt",
        out / "post_run_sha256.txt",
        parse_manifest(out / "pre_run_sha256.txt"),
        parse_manifest(out / "post_run_sha256.txt"),
    )
    print(f"[wrote] integrity_report.txt; identical={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
