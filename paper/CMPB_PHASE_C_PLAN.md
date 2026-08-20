# Phase C — CMPB polish (DeLong + ChemBERTa ablation + packaging)

**Status:** COMPLETE  
**Authority:** Phase A contribution contract; Phase B compressed manuscript  
**Hard rules honored:**
- Did NOT overwrite locked CVAE checkpoint or locked Layer-5* validation JSONs.
- Did NOT reprocess TUH.
- Did NOT run Phase 2 clinical-outcome pipelines.
- New analyses are **non-locked** secondary artifacts under `models/validation/`.

## Deliverables
1. **DeLong** — `src/validation/run_delong_oof_probes.py` → `models/validation/delong_oof_probe_*.json/md`
2. **ChemBERTa vs padded one-hot** — `src/validation/run_chemberta_onehot_ablation.py` → `models/validation/chemberta_onehot_ablation/`
3. **Packaging** — expanded `paper/CODE_AVAILABILITY.md`
4. **Manuscript** — Methods / Results §3.4–3.6 / Discussion / S1 / Table A.3 / ledger / bib `DeLong1988`; Overleaf PDF rebuilt (`paper/overleaf/main.pdf`)

## Headline secondary findings (non-locked)
- DeLong (OOF n=727): latent and best head significantly below θ/α and 2185-D; θ/α vs 2185-D n.s.
- One-hot: 10/10 signs on TUH/OSF/P-ADIC at low \(r\); rematched ChemBERTa: 9/10 at low \(r\) (neither recovers locked high-\(r\)).

## Claim boundaries
- Locked fold-0 ChemBERTa twin remains primary historical checkpoint.
- Ablation does not claim large-library ChemBERTa generalization.
- DeLong does not change locked CV-mean AUCs (0.579 / 0.675 / 0.699).
