# GPT / ChatGPT Image Prompt — CBM Graphical Abstract

Use this in ChatGPT (DALL·E / GPT Image) with **exact numbers**. Prefer **landscape / wide** (≈ 16:9 or 1200×340–400 px feel). Ask for **PNG**, high resolution, white background.

---

## Primary prompt (copy all of this)

```text
Create a publication-quality graphical abstract for a Computers in Biology and Medicine paper. Wide landscape banner (about 16:9), clean white background, Elsevier/scientific journal style. Flat vector illustration, not photorealistic, not 3D glossy AI chrome, no purple neon gradients, no stock-photo brains, no human faces, no watermarks, no decorative icons clutter.

Layout: three equal panels left → right connected by simple dark gray arrows. Thin teal/navy/amber color accents only. Sans-serif scientific typography; all text sharp and readable at small print size. Keep generous margins. No emoji.

Top banner title (one line, bold dark gray):
External validation: constraint-consistent direction preserved; diagnostic magnitude fails

PANEL 1 (left, soft teal border / pale teal fill) — title: Constrained EEG–drug CVAE
Show a minimal schematic: stacked EEG feature block → drug embedding chip labeled ChemBERTa → CVAE oval with μ → reconstructed features. Small constraint badge: PD band + connectivity, αc = 0.1.
Exact text under schematic:
2185-D resting EEG
Donepezil / Memantine
Train N = 66 (AD 37 / HC 29)

PANEL 2 (center, soft navy border / pale blue fill) — title: Direction preserved
Show three small cohort chips labeled TUH, OSF, P-ADIC, each with a green check and “10/10”.
Exact text:
Constraint-consistent signs vs training signature
TUH / OSF / P-ADIC: 10/10 each
r = 0.908 / 0.851 / 0.918
Tiny footnote style: no paired post-dose EEG

PANEL 3 (right, soft amber border / pale amber fill) — title: Diagnosis fails externally
Show a simple bar comparison: taller amber bar labeled θ/α 0.675; shorter teal bar labeled latent probe 0.579; thin navy tick for best head OOF 0.593.
Exact text:
Magnitude nulls across 4 external cohorts
Encoding: partial survival below spectral reference
Latent probe AUC = 0.579
θ/α mean-fold AUC = 0.675
Best head OOF AUC = 0.593 | CAUEEG n = 727

Bottom strip (optional, very small):
Mechanism transfers under domain shift · magnitude does not diagnose · loss localized at encoding

Critical: every number above must appear exactly as written. Do not invent extra statistics. Do not write “placeholder”. Do not crop text. High-resolution PNG suitable for journal graphical abstract.
```

---

## Short follow-up if text is messy (second message)

```text
Regenerate the same three-panel graphical abstract. Keep identical numbers. Fix typography: increase font size, no overlapping labels, no cut-off text, equal panel widths, white background, print-ready PNG.
```

---

## Negative constraints (add if the model ignores style)

```text
Avoid: purple/violet theme, dark mode, glassmorphism, glowing neurons, realistic brain MRI photos, cartoon scientists, QR codes, fake DOI, extra AUCs, extra datasets, watermark, logo soup.
```

---

## Numbers checklist (must match manuscript)

| Item | Exact value |
|------|-------------|
| Train N | 66 (AD 37 / HC 29) |
| Direction | 10/10 on TUH, OSF, P-ADIC |
| r | 0.908 / 0.851 / 0.918 |
| Latent probe AUC | 0.579 |
| θ/α mean-fold AUC | 0.675 |
| Best head OOF AUC | 0.593 |
| CAUEEG n (Dementia vs Normal probe) | 727 |
| αc | 0.1 |

After download: save as `paper/figures/graphical_abstract.png` (overwrite), then rebuild Overleaf zip.
