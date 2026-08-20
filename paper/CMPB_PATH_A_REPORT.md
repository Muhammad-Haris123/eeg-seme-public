# Path A completion report (CMPB 6.8 → target 8.0–8.3)

**Date:** 2026-08-20  
**Status:** Path A Weeks 1–4 executed in-repo (public GitHub/Zenodo URL still pending human push)

## Week 1 — Identity
- Protocol name locked: **SEME** (`paper/SEME_PROTOCOL.md`)
- Applied title: **SEME: A Decoupled Evaluation Protocol for Literature-Constrained EEG–Drug Generators**
- Cover letter: `paper/CMPB_COVER_LETTER.md`
- Hostile title/abstract pass applied in `00_frontmatter.md` (Results lead with constrained≫unconstrained; prior-only as circularity budget; locked high‑r demoted)

## Week 2 — Software packaging
- Regenerator: `src/paper/regenerate_cmpb_artifacts.py`
- `LICENSE` (MIT), `CITATION.cff`, expanded `paper/CODE_AVAILABILITY.md`
- **Still required from author:** create public repo/Zenodo, paste URL + commit hash

## Week 3 — Incremental value
- Results/Discussion lead with constrained≫unconstrained and prior-only circularity budget
- Incremental-value box in Discussion
- Encoding bake-off figure: `paper/figures/fig_bakeoff_encoding.png` (Fig. 6); α_c sweep renumbered Fig. 7

## Week 4 — Hygiene
- Ethics / Data availability already present in `07_backmatter.md`
- Code availability updated to point at regenerator
- Authors remain placeholder (complete at Editorial Manager)
- Overleaf package synced; run `pack_overleaf_zip.py` after PDF compile

## Hostile one-sentence test
Title+Abstract should read: evaluation protocol (SEME) with a constrained CVAE case study — **not** clinical PD twin validation.

## Estimated score impact (judgment)
Path A packaging/framing complete in manuscript: **~7.6–8.1** conditional on public URL at submission. Without public URL, expect **~7.3–7.7**. True **8.0–8.3** needs the public tag.
