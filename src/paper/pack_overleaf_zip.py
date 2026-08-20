"""
Rebuild paper/overleaf_cbm_elsarticle.zip from the Overleaf package folder.

Default source: paper/overleaf (or paper/overleaf_staging if overleaf is missing).
Does NOT overwrite overleaf from paper/latex (that used to wipe fresh builds).

    .\\eeg_twin\\Scripts\\python.exe src\\paper\\pack_overleaf_zip.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
ZIP_PATH = PAPER / "overleaf_cbm_elsarticle.zip"

SKIP_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".log",
    ".out",
    ".toc",
    ".lof",
    ".lot",
    ".synctex.gz",
    ".fls",
    ".fdb_latexmk",
    ".spl",
    ".pdf",
}
SKIP_PREFIXES = ("_", "compile", "c1", "c2", "c3", "bib")


def _package_root() -> Path:
    for candidate in (PAPER / "overleaf", PAPER / "overleaf_staging"):
        if (candidate / "main.tex").exists():
            return candidate
    raise SystemExit("No Overleaf package found. Run build_overleaf_elsarticle.py first.")


def _should_skip(path: Path) -> bool:
    name = path.name
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    if name.startswith(SKIP_PREFIXES):
        return True
    if name.startswith(("_preview", "_page", "_chk", "_t1", "_fix")):
        return True
    if path.suffix.lower() == ".svg" and "figures" in path.parts:
        return True
    return False


def build_zip(src: Path) -> int:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    n = 0
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if not path.is_file() or _should_skip(path):
                continue
            zf.write(path, arcname=path.relative_to(src).as_posix())
            n += 1
    return n


def main() -> None:
    src = _package_root()
    n = build_zip(src)
    print(f"Zipped {n} files from {src}")
    print(f"Wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
