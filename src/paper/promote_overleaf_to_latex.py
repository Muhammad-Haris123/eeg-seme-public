"""After build_overleaf_elsarticle.py, promote overleaf -> latex then pack zip."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OVERLEAF = ROOT / "paper" / "overleaf"
LATEX = ROOT / "paper" / "latex"


def main() -> None:
    if not (OVERLEAF / "main.tex").exists():
        raise SystemExit("Run build_overleaf_elsarticle.py first")
    LATEX.mkdir(parents=True, exist_ok=True)
    for name in ("main.tex", "references.bib", "README.md"):
        src = OVERLEAF / name
        if src.exists():
            shutil.copy2(src, LATEX / name)
    for sub in ("sections", "figures", "tables"):
        src_dir = OVERLEAF / sub
        dst_dir = LATEX / sub
        if not src_dir.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in src_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(src_dir)
                out = dst_dir / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)
    print(f"Promoted {OVERLEAF} -> {LATEX}")


if __name__ == "__main__":
    main()
