# Overleaf / Elsevier `elsarticle` package

Upload this **entire folder** to Overleaf:

1. Zip the `overleaf` directory (contents: `main.tex`, `sections/`, `figures/`, `tables/`, `references.bib`).
2. Overleaf → **New Project** → **Upload Project** → select the zip.
3. Set compiler to **pdfLaTeX** (default).
4. Click **Recompile**.

## Notes

- Document class: `elsarticle` (preprint, 12pt, 3p, times), bibliography `elsarticle-num`.
- Target journal: Computer Methods and Programs in Biomedicine (CMPB; ISSN 0169-2607).
- Graphical abstract: `figures/graphical_abstract.png` (Elsevier front matter; not Fig. N).
- Source of truth for prose remains `paper/sections/*.md`; re-run `src/paper/build_overleaf_elsarticle.py` after major edits.

## If `highlights` / `graphicalabstract` errors

Some older TeX Live installs need:
```latex
\usepackage{framed} % rarely needed
```
Or comment out the `highlights` / `graphicalabstract` environments and paste those assets in the Elsevier submission form separately.
