"""
Integrity manifests for the ensemble-bootstrap run.

Writes ONLY under models/validation/fold_ensemble/bootstrap/.
Does not overwrite fold_ensemble/* integrity files from the prior run.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def collect_protected(root: Path) -> List[Path]:
    files: List[Path] = []

    v3 = root / "models" / "validation" / "complete_validation_report_v3.json"
    if not v3.is_file():
        raise FileNotFoundError(v3)
    files.append(v3)

    fold_sims = root / "models" / "validation" / "fold_sims"
    files.extend(sorted(p for p in fold_sims.rglob("*") if p.is_file()))

    boot_dir = root / "models" / "validation"
    files.extend(sorted(boot_dir.glob("*bootstrap*.json")))

    paper = root / "paper"
    for p in sorted(paper.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".zip":
            continue
        if "__pycache__" in p.parts:
            continue
        files.append(p)

    for ck in [
        root / "models" / "checkpoints_constrained" / "fold_0_best.pt",
        root / "models" / "checkpoints_constrained" / "checkpoint_constrained.pt",
        root / "models" / "checkpoints_5fold" / "fold_0_best.pt",
    ]:
        if not ck.is_file():
            raise FileNotFoundError(ck)
        files.append(ck)

    ens = (
        root
        / "models"
        / "validation"
        / "fold_ensemble"
        / "fold_ensemble_external_direction_results.json"
    )
    if not ens.is_file():
        raise FileNotFoundError(ens)
    files.append(ens)

    # Existing ensemble summary is also a point-estimate output; protect it.
    ens_md = (
        root
        / "models"
        / "validation"
        / "fold_ensemble"
        / "fold_ensemble_external_direction_summary.md"
    )
    if ens_md.is_file():
        files.append(ens_md)

    seen = set()
    out: List[Path] = []
    for p in files:
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def build_map(root: Path, paths: List[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for i, p in enumerate(paths, 1):
        mapping[_rel(root, p)] = sha256_file(p)
        if i % 500 == 0:
            print(f"  hashed {i}...")
    return mapping


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
    for rel in sorted(mapping.keys()):
        lines.append(f"{mapping[rel]}  {rel}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_manifest(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        mapping[rel] = digest
    return mapping


def compare(
    pre: Dict[str, str], post: Dict[str, str]
) -> Tuple[List[str], List[Tuple[str, str, str]], List[str], List[str]]:
    unchanged: List[str] = []
    changed: List[Tuple[str, str, str]] = []
    missing: List[str] = []
    for rel, old in sorted(pre.items()):
        if rel not in post:
            missing.append(rel)
            continue
        new = post[rel]
        if old == new:
            unchanged.append(rel)
        else:
            changed.append((rel, old, new))
    extras = sorted(set(post) - set(pre))
    return unchanged, changed, missing, extras


def write_report(
    report_path: Path,
    pre_path: Path,
    post_path: Path,
    pre: Dict[str, str],
    post: Dict[str, str],
) -> bool:
    unchanged, changed, missing, extras = compare(pre, post)
    ok = (
        len(changed) == 0
        and len(missing) == 0
        and len(extras) == 0
        and len(unchanged) == len(pre)
    )
    lines = [
        "# Integrity report (ensemble bootstrap run)",
        f"# timestamp: {datetime.now().isoformat()}",
        f"# pre: {pre_path.as_posix()}",
        f"# post: {post_path.as_posix()}",
        "",
        f"number_of_protected_files_checked: {len(pre)}",
        f"number_unchanged: {len(unchanged)}",
        f"number_changed: {len(changed)}",
        f"number_missing_in_post: {len(missing)}",
        f"number_extra_keys_in_post: {len(extras)}",
        f"all_protected_artifacts_byte_identical: {str(ok).lower()}",
        "",
        "changed_files:",
    ]
    if not changed:
        lines.append("  (none)")
    else:
        for rel, old, new in changed:
            lines.append(f"  - {rel}")
            lines.append(f"      old_sha256: {old}")
            lines.append(f"      new_sha256: {new}")
    lines.append("")
    lines.append("missing_in_post:")
    lines.append("  (none)" if not missing else "\n".join(f"  - {m}" for m in missing))
    lines.append("")
    lines.append("extra_keys_in_post_not_in_pre:")
    lines.append("  (none)" if not extras else "\n".join(f"  - {e}" for e in extras))
    lines.append("")
    lines.append(
        "RESULT: PASS — Changed protected files = 0"
        if ok
        else "RESULT: FAIL — protected artifacts differ; STOP"
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"pre", "post", "compare"}:
        print("Usage: ensemble_bootstrap_integrity.py {pre|post|compare}", file=sys.stderr)
        return 2
    root = _root()
    out = root / "models" / "validation" / "fold_ensemble" / "bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1]
    if mode in {"pre", "post"}:
        label = "pre-run" if mode == "pre" else "post-run"
        name = "pre_run_sha256.txt" if mode == "pre" else "post_run_sha256.txt"
        paths = collect_protected(root)
        print(f"Hashing {len(paths)} protected files ({label})...")
        mapping = build_map(root, paths)
        write_manifest(out / name, root, mapping, label)
        print(f"[wrote] {out / name} ({len(mapping)} files)")
        return 0

    pre_path = out / "pre_run_sha256.txt"
    post_path = out / "post_run_sha256.txt"
    report_path = out / "integrity_report.txt"
    pre = parse_manifest(pre_path)
    post = parse_manifest(post_path)
    ok = write_report(report_path, pre_path, post_path, pre, post)
    print(f"[wrote] {report_path}; identical={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
