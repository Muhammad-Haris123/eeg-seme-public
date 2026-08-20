# Plan to ~8.3 (CBM-ready, no post-dose EEG)

**Status:** Active execution plan  
**Target score:** ~8.0–8.3 (strong, claim-aligned submission)  
**Not the goal:** ~9 empirical pharmacodynamic twin (requires paired pre/post drug EEG)  
**Date:** 2026-08-09  

---

## 0. Target definition

### What ~8.3 means here

A Computers in Biology and Medicine reviewer can accept the paper as:

> A **pharmacodynamically constrained EEG–drug CVAE simulation framework** that (i) preserves **constraint-consistent, model-generated** signed response direction under external EEG domain shift, (ii) does **not** yield generalizable diagnostic magnitude, and (iii) shows a **quantified** direction↔diagnosis trade-off via unconstrained / \(\alpha_c\) experiments.

### What would block ~8.3

- Title/abstract still read as “pharmacodynamic effects transfer empirically”
- No matched unconstrained external battery
- No prior-only baseline
- No \(\alpha_c\) curve (or causal trade-off language without it)
- OSF oversold; missing CIs / checkpoint clarity / code+ethics / placeholders

### Explicitly out of scope for this plan

- Waiting for real post-dose Donepezil/Memantine EEG
- Full ChemBERTa/fingerprint/shuffle zoo
- Full MMD / domain-classifier suite (optional stretch only)
- GNN re-introduction

---

## 1. Revised central claim (lock before more prose)

**Use this spine (replace current causal overclaim):**

1. **Constraint-consistent direction:** Model-generated Donepezil/Memantine band/connectivity signs agree with the **training constrained signature** on TUH, OSF, and P-ADIC (currently 10/10; \(r \approx 0.85\)–\(0.92\)).
2. **Diagnostic magnitude fails externally:** Especially P-ADIC and CAUEEG; OSF is exploratory (HC \(n=12\)).
3. **Encoding gap:** CAUEEG theta/alpha reference AUC ≈ 0.675; latent probe ≈ 0.579; best head ≈ 0.593.
4. **Trade-off (to be demonstrated in this plan):** Unconstrained vs constrained and/or \(\alpha_c\) sweep show how constraint strength moves direction fidelity vs latent diagnostic AUC.

**Forbidden wording until Workstream B gates pass:** “pharmacodynamic direction transfers,” “validated drug-response twin,” “constraints cause compression” (use “associated with / consistent with” until B1–B3 results exist).

**Decision:** Treat `STORYLINE_LOCKED.md` as needing a controlled reopen for claim language only; do not change spine numbers unless new JSON changes them.

---

## 2. Workstreams and exact tasks

### Workstream A — Claim & manuscript hygiene (Days 1–3)

| ID | Task | How / where | Done when |
|----|------|-------------|-----------|
| A1 | Retitle | Options in §6 below | Title matches constraint-consistent framing |
| A2 | Rewrite Abstract, Highlights, Graphical abstract text, Conclusion | `paper/sections/00_frontmatter.md`, `07_conclusion.md`, GA caption | No empirical-transfer reading |
| A3 | Methods: endpoint definition | `02_methods.md` — primary mechanistic endpoint = signature agreement; **no paired post-dose EEG** stated here | Endpoint cannot be misread |
| A4 | Downgrade OSF | Results diagnosis + Discussion | “Exploratory; not replicated on P-ADIC” |
| A5 | Relabel TUH role | Methods/Results | Domain-shift / proxy labels, not AD diagnosis |
| A6 | “Feature ceiling” → “spectral reference” | Encoding results + Discussion | Consistent wording |
| A7 | Purge jargon | Methods/Results/backmatter | No Layer 5b / project config / Phase-2 in prose |
| A8 | Authors, CRediT, funding | `main.tex`, backmatter | No placeholders |
| A9 | Ethics + code availability | New short sections | Provider-true ethics; repo+commit+env commands |
| A10 | Rebuild Overleaf zip | `python src/paper/pack_overleaf_zip.py` | Fresh `paper/overleaf_cbm_elsarticle.zip` |

**Exit A:** Hostile read of title+abstract cannot claim you proved real drug EEG transfer.

---

### Workstream B — Experiments that create ~8.3 (Days 3–18)

Existing assets:

- Variants: `MODEL_VARIANTS['baseline_mlp']` / `constrained_mlp` in `src/models/config.py`
- Trainer: `train_upgraded_model(..., constraint_weight=...)` in `src/models/train_phase2_upgraded.py`
- External runners: `src/tuh/run_layer5e_caueeg.py`, `run_layer5d_padic.py`, `run_layer5b_ad_labeled.py`, `run_latent_probe_caueeg.py`, `run_latent_classifier_head_caueeg.py`
- In-sample unconstrained metrics already exist: `models/evaluation/constrained_vs_unconstrained_report.json` (MSE/Pearson/TVB only — **not** external Layer-5e battery)

#### B1. Matched unconstrained twin (mandatory)

| Step | Action |
|------|--------|
| B1.1 | Train `baseline_mlp` / `use_constrained_loss=False` with **same** data, folds, seed, epochs as constrained |
| B1.2 | Save frozen checkpoint e.g. `models/checkpoints_unconstrained/checkpoint_unconstrained.pt` |
| B1.3 | Document selection rule (must match constrained: which fold / retrain-on-all / seed) |
| B1.4 | Point external runners at unconstrained ckpt (or add `--checkpoint` CLI) |
| B1.5 | Run: direction on TUH + OSF + P-ADIC; magnitude on P-ADIC + CAUEEG Dem vs Nor; CAUEEG latent probe + nested head |
| B1.6 | Write `models/validation/unconstrained_external_battery.json` + short `.md` summary |
| B1.7 | Update `paper/citation_ledger.md` for every new number |

**Pre-registered interpretation (write before reading results):**

| Pattern | Manuscript consequence |
|---------|------------------------|
| Constrained ≫ unconstrained on direction; unconstrained ≥ constrained on latent AUC | Keep trade-off as main result |
| Both ~10/10 direction | Soften “constraints cause direction”; emphasize domain-shift preservation + encoding |
| Both weak latent AUC | Soften “constraint compresses diagnosis” |

#### B2. Prior-only baseline (mandatory)

| Step | Action |
|------|--------|
| B2.1 | Script: apply literature signed band/connectivity deltas to baseline 2185-D features (from `get_eeg_effect_prior`) |
| B2.2 | Score same 10-block direction metric vs training constrained mean effect vector |
| B2.3 | Save `models/validation/prior_only_direction.json` |

**Point:** How much of “mechanistic transfer” needs a CVAE at all?

#### B3. Constraint-strength sweep (hero experiment for 8.3)

| \(\alpha_c\) | Train + freeze + evaluate |
|--------------|---------------------------|
| 0, 0.01, 0.05, **0.1**, 0.2, 0.5 | Direction (min: TUH + P-ADIC; prefer all 3), CAUEEG latent AUC, reconstruction MSE |

| Step | Action |
|------|--------|
| B3.1 | Loop `constraint_weight` in trainer; save ckpts under `models/checkpoints_alpha_c/{weight}/` |
| B3.2 | Short external eval per weight (direction + CAUEEG probe) |
| B3.3 | Figure: \(x=\alpha_c\), dual axis direction score/\(r\) vs latent AUC |
| B3.4 | JSON: `models/validation/constraint_strength_sweep.json` |

**This figure is the difference between “interpreted trade-off” and “demonstrated trade-off.”**

#### B4. Extras that push 8.0 → 8.3 (do after B1–B3)

| ID | Task | Why |
|----|------|-----|
| B4.1 | ChemBERTa vs one-hot drug ID (same constrained setup) | Stops “ChemBERTa is cosmetic” |
| B4.2 | Formal disease-label table (actual / all-0 / all-1) for primary magnitude contrasts | Sensitivity already partial; make primary |
| B4.3 | One alternate magnitude (decoder-space or band-block \(\ell_2\)) on CAUEEG Dem vs Nor | Null not metric-specific |
| B4.4 | 2185-D logistic (or HistGB) nested CV on CAUEEG Dem vs Nor | Stronger than theta/alpha-only “reference” |

---

### Workstream C — Statistics (parallel, Days 2–5 and after B)

| ID | Task | Done when |
|----|------|-----------|
| C1 | Primary hierarchy in Methods | P1 direction (3 cohorts); P2 CAUEEG magnitude; P3 latent AUC; rest exploratory |
| C2 | 95% CI for \(r\) (Fisher \(z\)) | In Table 2 + text |
| C3 | 95% CI for Cohen’s \(d\) | Table 3 / forest (already approximate in fig; lock exact method) |
| C4 | AUC CI / bootstrap; label mean-fold vs OOF | Encoding section + Table 4 |
| C5 | CAUEEG null: CI on \(d\) + SESOI/TOST **or** powered-for-small-effects sentence | Discussion/Results |
| C6 | Multiplicity statement | Escalation = exploratory design, not confirmatory MCT |

---

### Workstream D — Reproducibility & submission pack (Days 14–20)

| ID | Task |
|----|------|
| D1 | Checkpoint/CV diagram in Methods |
| D2 | Parameter count; seed; fold \(n\); unique subjects (66 files vs unique BIDS) |
| D3 | Preprocessing table by cohort |
| D4 | Public code repo + commit hash + `environment`/requirements + commands to regenerate validation JSON |
| D5 | Ethics/data-use statements from real provider text |
| D6 | Sync `paper/latex` → zip via `pack_overleaf_zip.py` |
| D7 | Hostile internal review against ChatGPT circularity critique |

---

## 3. Week-by-week schedule

| Week | Focus | Deliverables |
|------|--------|--------------|
| **Week 1** | A + C start + B1 train | Claim rewrite draft; unconstrained training started/finished; CI scripts |
| **Week 2** | B1 eval + B2 | Unconstrained external JSON; prior-only JSON; gate decision (§2 B1 table) |
| **Week 3** | B3 + B4.1–B4.2 | \(\alpha_c\) sweep figure; ChemBERTa one-hot; disease-label table |
| **Week 4** | Rewrite + D + B4.3–B4.4 if time | Full Results/Discussion; ethics/code; Overleaf zip; submit checklist |

---

## 4. Manuscript structure after experiments

1. **Intro** — dual validation problem; no post-dose EEG stated early  
2. **Methods** — twin; constraints; endpoints; **ablations**; stats hierarchy; checkpoint rule  
3. **Results**  
   - 3.1 Unconstrained vs constrained (+ prior-only)  
   - 3.2 External constraint-consistent direction  
   - 3.3 Magnitude escalation (OSF exploratory)  
   - 3.4 Encoding gap (+ feature-space reference if B4.4)  
   - 3.5 \(\alpha_c\) trade-off curve (**hero**)  
4. **Discussion** — what was / was not validated; clinical = hypothesis simulation  
5. **Limitations** — no post-dose EEG; \(N=66\); dementia labels; metric choices  

---

## 5. Title options (pick after B1 gate)

**Preferred before gate (safe):**  
`External validation of a pharmacodynamically constrained EEG-drug CVAE: constraint-consistent direction preservation and diagnostic information loss`

**If B1–B3 support trade-off:**  
`Constraint strength trades signed response-direction preservation against latent diagnostic information in an EEG-drug CVAE`

**Avoid:**  
`Pharmacodynamic Direction Transfers, Diagnostic Magnitude Does Not...` (current — too strong)

---

## 6. Definition of done (~8.3 checklist)

### Science
- [x] Title/abstract = constraint-consistent simulation, not empirical drug transfer  
- [x] Unconstrained matched external battery in paper + JSON  
- [x] Prior-only baseline in paper + JSON  
- [x] \(\alpha_c\) sweep figure in paper + JSON *(Fig. 6 + `constraint_strength_sweep.json`)*  
- [x] OSF exploratory; P-ADIC/CAUEEG carry diagnostic null *(prose updated; keep reinforcing)*  
- [x] Encoding claim scoped to tested readouts + spectral/feature reference  
- [x] ChemBERTa vs one-hot **or** ChemBERTa de-emphasized in pitch *(de-emphasized: two-drug conditioner)*  
- [x] Citation ledger updated for every new number  

### Submission
- [x] Real authors / CRediT / funding  
- [x] Ethics + code availability  
- [x] CIs on \(r\) *(Fisher-z JSON done; d/AUC CI pending)*  
- [x] Checkpoint/CV unambiguous *(SELECTION.md + Methods)*  
- [x] No pipeline jargon *(purged Layer/Phase narrative; JSON keys kept for provenance)*  
- [x] Fresh `paper/overleaf_cbm_elsarticle.zip`  

### Gate
- [x] B1 interpretation table applied; title finalized accordingly *(Option A kept; see `B1_GATE_DECISION.md`)*  

---

## 7. Risk register

| Risk | Mitigation |
|------|------------|
| Unconstrained also 10/10 | Pivot: domain-shift preservation + encoding; drop constraint-causal direction story |
| Unconstrained also low AUC | Soften compression-from-constraint; keep magnitude null + encoding gap |
| \(\alpha_c\) sweep flat | Report honestly; trade-off may be bottleneck/KL not \(\alpha_c\) — optional \(\beta\) note in limitations |
| Train time / GPU | Prioritize B1 full battery; sweep on TUH+P-ADIC+CAUEEG probe only |
| Scope creep | Freeze ChatGPT wishlist; only B4 extras after B1–B3 |

---

## 8. First actions tomorrow (ordered)

1. Open storyline reopen note: adopt §1 claim language.  
2. Start unconstrained training (`baseline_mlp`, `constraint_weight=0`).  
3. Draft prior-only script sketch against `get_eeg_effect_prior` + `BAND_SLICES`.  
4. Patch external runners to accept checkpoint path argument.  
5. Rewrite title/abstract/endpoint paragraph (Workstream A) in parallel.  
6. Add Fisher-\(z\) CI helper for existing \(r\) values (quick win for Table 2).

---

## 9. One-line summary

**To hit ~8.3 without post-dose EEG:** lock honest claims → prove the trade-off with **unconstrained + prior-only + \(\alpha_c\) sweep** on the **same** external endpoints → add ChemBERTa/one-hot and a stronger feature reference → ship stats/repro hygiene → submit as a **constraint-consistent simulation / information-trade-off** paper.
