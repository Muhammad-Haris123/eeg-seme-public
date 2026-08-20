# Phase A — CMPB Journal Strategy

**Status:** LOCKED TARGET  
**Date:** 2026-08-20  
**Decision:** Proceed with **Computer Methods and Programs in Biomedicine (CMPB)** only.  
**CBM:** On hold; not pursued in this track.

---

## 1. Target lock

| Item | Decision |
|------|----------|
| Primary journal | **Computer Methods and Programs in Biomedicine** (Elsevier; ISSN 0169-2607) |
| Backup | Deferred (CBM on hold). Revisit only if CMPB desk-rejects on scope/length |
| Article type | Original Research Manuscript |
| Identity to sell | Formal **evaluation / measurement methodology** for constrained chemistry-conditioned EEG generators, demonstrated on a CVAE twin |
| Identity not to sell | Clinically validated AD drug-response digital twin |

### Why CMPB (given CBM hold)

CMPB wants formal computing methods, biomedical informatics principles, reusable methodology/software, and clear algorithmic utility. Our strongest publishable asset is not a clinically usable twin; it is a **decoupled evaluation protocol** (signs vs continuous \(r\) vs magnitude vs encoding) with hard negative controls. That maps to CMPB if framed correctly. It maps poorly if framed as “validated pharmacodynamic twin.”

---

## 2. CMPB submission constraints (must drive Phase B)

Sources: CMPB Guide for Authors summaries / special-issue author instructions / Livewrite journal profile (verify on Elsevier GFA before upload).

| Requirement | Spec | Current manuscript | Gap |
|-------------|------|--------------------|-----|
| Structured abstract | 4 headings: Background and Objective; Methods; Results; Conclusions; ≤350 words | Unstructured ~230 words | **Must rewrite** |
| Main text length | Normally ≤ **3500 words** excluding abstract | ≈ **12,400 words** (sections 1–5 + conclusion) | **Critical: must compress ~3.5×** |
| Sections | Intro, Methods, Results, Discussion (+ Acknowledgements) | Present (plus heavy S1) | Move detail to Appendix/S1 |
| Keywords | 3–6 | 6 | OK |
| References | Original research often capped ~**50** | 35 in `references.bib` | OK |
| Cover letter | Required: fit, known, new, not published elsewhere | Missing | **Must write** |
| Highlights | Elsevier-style 3–5 bullets, ≤85 chars each | Present | Keep; retune to framework |
| Software / reproducibility | Strongly rewarded by CMPB culture | Dense S1; no public release package yet | Phase C packaging |
| First decision timing (reported) | ~40 days to first decision (profile estimate) | — | Plan rebuttal bank early |

**Strategic implication:** Phase B is not only “sell the framework.” It is also a **controlled compression** of the manuscript into CMPB length while preserving the locked scientific spine.

---

## 3. Similar-paper survey (10 anchors)

These are the nearest published patterns for positioning. Where a paper is adjacent (not CMPB), it is marked and used only for competitive landscape.

### A. CMPB / CMPB-core methods papers

| # | Paper | Why it matters for us |
|---|-------|------------------------|
| 1 | **SpaOmicsVAE** (CMPB, 2025; VAE + dual GNN for spatial multi-omics) | CMPB accepts VAE **frameworks** when the computing method and evaluation are explicit. Contribution is method + benchmarks, not clinical diagnosis alone. |
| 2 | **TransVAE-DTA** (CMPB, 2023; Transformer + VAE for drug–target affinity) | CMPB already publishes chemistry-conditioned generative/VAE hybrids. Supports ChemBERTa-conditioned modeling as in-scope **if** contribution is computational. |
| 3 | **MEDUSA** (CMPB, 2023; Python EEG/BCI software ecosystem) | Gold-standard CMPB EEG software paper: reusable tools, pipelines, community use. Shows CMPB rewards **software/method ecosystems**, not only accuracy claims. |
| 4 | **Şeker et al.** (CMPB, 2021; EEG complexity / permutation entropy for early AD) | CMPB EEG–AD precedent exists, but as biomarker/method papers with clear metrics—not digital-twin overclaim. |
| 5 | Recent CMPB method/software articles (2026 examples: path-planning HS-RRT*; AneuSI isolation tool) | Editorial preference visible: **named method/tool + evaluation**, concise application story. |

### B. Close competitive landscape (often not CMPB; still what referees compare against)

| # | Paper / type | Positioning lesson |
|---|--------------|--------------------|
| 6 | EEG–AD deep classifiers (ensemble / CNN–BiLSTM / spectrogram transformers in biomedical signal journals) | Field is flooded with **high-accuracy diagnosis** papers. Competing on AUC alone is a trap for us (our diagnostic story is attenuation, not SOTA). |
| 7 | Domain-shift / cross-dataset EEG transfer papers | External multi-cohort testing is valued; our escalation design is a strength **if** not sold as diagnosis win. |
| 8 | AD digital-twin / progression VAE work (cLVAE, SimulAD-style) | Neighbors claim trajectory/digital-twin utility. We must differentiate: **signature concordance under constraint**, not disease-progression forecasting. |
| 9 | DADD / EEG mechanism twins for diagnosis–prognosis | Closest scientific cousins. Our niche is chemistry-conditioned **constraint evaluation**, not CSF/conversion prediction. |
| 10 | SelfEEG / TorchEEG / BioFoundation-style EEG toolkits | CMPB culture increasingly expects reusable code. Our S1 is good internally; public packaging is still a gap vs these precedents. |

### Patterns that win at CMPB

1. **Name the method** (framework / protocol / tool), then show a biomedical case study.
2. State **what is known / what is added** in one paragraph (cover letter + intro).
3. Prefer **controls, ablations, external tests** over single-dataset accuracy theater.
4. Provide **software/reproducibility** as a first-class product.
5. Keep main text short; put configs and paths in supplements.

### Patterns that lose at CMPB (and fit our risks)

1. “Novel digital twin validates pharmacology” without paired outcome data.
2. Leading with a single locked high-\(r\) checkpoint while seed/fold sensitivity collapses.
3. Diagnostic SOTA framing when the paper’s honest result is encoding attenuation.
4. Encyclopedia Methods in the main text (~5k words Methods alone in our draft).

---

## 4. Contribution contract (1 page)

### One sentence — what the paper IS

This paper contributes a **formal, externally stress-tested evaluation methodology** for literature-constrained, chemistry-conditioned EEG generative models that separately scores (i) signed drug-response signature concordance, (ii) continuous effect-vector fidelity under fold/seed variation, (iii) diagnostic magnitude transfer, and (iv) residual encoding of diagnostic label information, demonstrated on a pharmacodynamically penalized CVAE twin for Donepezil/Memantine.

### Three sentences — what the paper is NOT

1. It is **not** a clinically validated pharmacodynamic digital twin and does **not** claim empirical post-dose EEG drug-effect validation (external cohorts lack paired pre/post-drug recordings).  
2. It is **not** a demonstration that the CVAE independently discovered drug direction or achieved seed-stable continuous external fidelity; prior-only already reaches 10/10 signs, and locked high continuous \(r\) is checkpoint-specific.  
3. It is **not** an Alzheimer’s diagnostic system, ChemBERTa chemical-generalization study, or claim of constraint-caused diagnostic compression; feature references retain more diagnostic signal than tested latent readouts, without establishing a simple direction–diagnosis trade-off.

### Contract addendum — mandatory claim boundaries

| Allowed claim | Forbidden claim |
|---------------|-----------------|
| Constraint-consistent signature concordance under domain shift (locked protocol) | External pharmacodynamic validation |
| Signs more stable than continuous \(r\) in this battery | Architecture-wide continuous-\(r\) robustness |
| Prior-only / unconstrained / multi-seed / imitation as core method | “Model discovered pharmacology” |
| Encoding attenuation under tested heads (0.579 / 0.593 vs 0.675 / 0.699) | Diagnostic superiority or ceiling theorems |
| Measurement framework for constrained EEG–drug generators | Standalone clinical decision support |

### Cover-letter skeleton (Phase D will expand)

- **Fit:** CMPB methods/software + biomedical computing evaluation.  
- **Known:** EEG–AD classifiers and generative twins often conflate reconstruction, direction, magnitude, and diagnosis.  
- **Added:** Decoupled endpoint protocol + hard controls on a constrained CVAE case study.  
- **Not elsewhere:** Confirm unpublished / not under review elsewhere.

---

## 5. Positioning vs CMPB expectations

| CMPB ask | Our answer under the contract |
|----------|-------------------------------|
| What computing method is new? | Decoupled evaluation protocol + constrained generative case study with signed ReLU penalties and locked external assessment rules |
| Why not just another EEG–AD CNN? | We do not compete on diagnosis AUC; we measure dissociation under constraint |
| Why not reject for circularity? | Circularity is named; prior-only and imitation controls quantify how much “success” is prior-shaped |
| Where is the software? | Must be elevated in Phase C (public tag, README, regenerate tables) |
| Can it fit 3500 words? | Only after aggressive Phase B compression |

---

## 6. Phase A decision outputs (actionable)

1. **Journal:** CMPB only for this track.  
2. **Contribution identity:** Evaluation methodology first; twin second (worked example).  
3. **Hard packaging constraint:** Structured abstract + ~3500-word main text + cover letter.  
4. **Do not change locked science in Phase A.** No Phase 2, no DeLong yet, no ChemBERTa ablation yet, no TUH reprocessing, no number edits.  
5. **Phase B should start from this contract**, not from CBM storyline language.

### Recommended Phase B order (next, when authorized)

1. Rewrite Abstract into CMPB 4-section structure under the contract.  
2. Rewrite Intro contribution paragraph + Conclusion to lead with framework.  
3. Demote locked \(r\) optically in Highlights/Abstract.  
4. Compress Methods/Results toward ≤3500 words; park provenance in S1.  
5. Keep all locked numbers unchanged.

### Explicitly deferred to later phases

- ChemBERTa vs one-hot (Phase C science polish)  
- DeLong on stored OOF (Phase C)  
- Public code release packaging (Phase C)  
- Submission zip / Editorial Manager upload (Phase D)

---

## 7. Phase A completion checklist

- [x] Target locked: CMPB  
- [x] CBM hold respected  
- [x] Similar-paper landscape surveyed (10 anchors)  
- [x] Contribution contract written (1-sentence IS + 3-sentence IS NOT)  
- [x] CMPB format constraints captured (structured abstract; ~3500 words; cover letter; ≤50 refs OK)  
- [x] Critical length gap quantified (~12.4k → ~3.5k)  
- [ ] Author confirmation that Phase B may begin from this contract

---

## 8. One-page printable contract (copy block)

> **Contribution contract — CMPB track**  
>  
> **IS:** A formal evaluation methodology for literature-constrained, chemistry-conditioned EEG generative models that separately scores signed signature concordance, continuous effect-vector fidelity (fold/seed sensitivity), diagnostic magnitude transfer, and encoding retention of diagnostic label information, demonstrated on a pharmacodynamically penalized Donepezil/Memantine CVAE.  
>  
> **IS NOT (1):** A clinically validated pharmacodynamic digital twin or post-dose EEG drug-effect validation.  
> **IS NOT (2):** Evidence that the CVAE independently discovered drug direction or achieved seed-stable continuous external fidelity.  
> **IS NOT (3):** An Alzheimer’s diagnostic product, ChemBERTa generalization claim, or proof of constraint-caused diagnostic compression.  
>  
> **Primary success metric for revision quality:** A CMPB referee can quote the IS sentence and never be misled by the Abstract into believing any IS-NOT claim.

---

*End of Phase A memo. No manuscript science was modified in this phase.*
