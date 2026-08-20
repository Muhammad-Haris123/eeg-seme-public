"""
SHA-256 integrity manifests for protected CBM artifacts.

Read-only hashing of protected files. Writes ONLY under
models/validation/fold_ensemble/ when invoked via CLI.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


def collect_protected_files(root: Path) -> List[Path]:
    """Deterministic ordered list of protected artifacts."""
    files: List[Path] = []

    v3 = root / "models" / "validation" / "complete_validation_report_v3.json"
    if not v3.is_file():
        raise FileNotFoundError(v3)
    files.append(v3)

    fold_sims = root / "models" / "validation" / "fold_sims"
    if not fold_sims.is_dir():
        raise FileNotFoundError(fold_sims)
    files.extend(sorted(p for p in fold_sims.rglob("*") if p.is_file()))

    boot_dir = root / "models" / "validation"
    files.extend(sorted(boot_dir.glob("*bootstrap*.json")))

    paper = root / "paper"
    if not paper.is_dir():
        raise FileNotFoundError(paper)
    for p in sorted(paper.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".zip":
            continue
        if "__pycache__" in p.parts:
            continue
        files.append(p)

    checkpoints = [
        root / "models" / "checkpoints_constrained" / "fold_0_best.pt",
        root / "models" / "checkpoints_constrained" / "checkpoint_constrained.pt",
        root / "models" / "checkpoints_5fold" / "fold_0_best.pt",
    ]
    for ck in checkpoints:
        if not ck.is_file():
            raise FileNotFoundError(ck)
        files.append(ck)

    seen = set()
    out: List[Path] = []
    for p in files:
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def build_manifest_map(root: Path, paths: Iterable[Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for i, p in enumerate(paths, 1):
        mapping[_rel(root, p)] = sha256_file(p)
        if i % 500 == 0:
            print(f"  hashed {i} files...")
    return mapping


def write_manifest_txt(
    out_path: Path,
    root: Path,
    mapping: Dict[str, str],
    label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_manifest_txt(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        mapping[rel] = digest
    return mapping


def compare_manifests(
    pre: Dict[str, str], post: Dict[str, str]
) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str, str]], List[str]]:
    unchanged: List[str] = []
    changed_detail: List[Tuple[str, str, str]] = []
    missing: List[str] = []
    for rel, old in sorted(pre.items()):
        if rel not in post:
            missing.append(rel)
            continue
        new = post[rel]
        if old == new:
            unchanged.append(rel)
        else:
            changed_detail.append((rel, old, new))
    extras = sorted(set(post) - set(pre))
    changed_names = [c[0] for c in changed_detail]
    return unchanged, changed_names, missing, changed_detail, extras


def write_comparison_report(
    report_path: Path,
    pre_path: Path,
    post_path: Path,
    pre: Dict[str, str],
    post: Dict[str, str],
) -> bool:
    unchanged, changed_names, missing, changed_detail, extras = compare_manifests(
        pre, post
    )
    all_identical = (
        len(changed_names) == 0
        and len(missing) == 0
        and len(extras) == 0
        and len(unchanged) == len(pre)
    )
    lines = [
        "# Integrity check report (pre-run vs post-run)",
        f"# timestamp: {datetime.now().isoformat()}",
        f"# pre_manifest: {pre_path.as_posix()}",
        f"# post_manifest: {post_path.as_posix()}",
        "",
        f"number_of_protected_files_checked: {len(pre)}",
        f"number_unchanged: {len(unchanged)}",
        f"number_changed: {len(changed_names)}",
        f"number_missing_in_post: {len(missing)}",
        f"number_extra_keys_in_post: {len(extras)}",
        f"all_protected_artifacts_byte_identical: {str(all_identical).lower()}",
        "",
        "changed_files:",
    ]
    if not changed_detail:
        lines.append("  (none)")
    else:
        for rel, old, new in changed_detail:
            lines.append(f"  - {rel}")
            lines.append(f"      old_sha256: {old}")
            lines.append(f"      new_sha256: {new}")
    lines.append("")
    lines.append("missing_in_post:")
    if not missing:
        lines.append("  (none)")
    else:
        for rel in missing:
            lines.append(f"  - {rel}")
    lines.append("")
    lines.append("extra_keys_in_post_not_in_pre:")
    if not extras:
        lines.append("  (none)")
    else:
        for rel in extras:
            lines.append(f"  - {rel}")
    lines.append("")
    if all_identical:
        lines.append("RESULT: PASS — Changed protected files = 0")
    else:
        lines.append(
            "RESULT: FAIL — protected artifacts differ; STOP manuscript integration"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_identical


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"pre", "post", "compare"}:
        print("Usage: integrity_manifest.py {pre|post|compare}", file=sys.stderr)
        return 2
    root = _root()
    out_dir = root / "models" / "validation" / "fold_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = argv[0]
    if mode in {"pre", "post"}:
        label = "pre-run" if mode == "pre" else "post-run"
        out_name = (
            "pre_run_integrity_sha256.txt"
            if mode == "pre"
            else "post_run_integrity_sha256.txt"
        )
        paths = collect_protected_files(root)
        print(f"Hashing {len(paths)} protected files ({label})...")
        mapping = build_manifest_map(root, paths)
        out_path = out_dir / out_name
        write_manifest_txt(out_path, root, mapping, label)
        print(f"[wrote] {out_path} ({len(mapping)} files)")
        return 0

    pre_path = out_dir / "pre_run_integrity_sha256.txt"
    post_path = out_dir / "post_run_integrity_sha256.txt"
    report_path = out_dir / "integrity_check_report.txt"
    if not pre_path.is_file() or not post_path.is_file():
        raise FileNotFoundError("Need both pre and post manifests for compare")
    pre = parse_manifest_txt(pre_path)
    post = parse_manifest_txt(post_path)
    ok = write_comparison_report(report_path, pre_path, post_path, pre, post)
    print(f"[wrote] {report_path}; identical={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
