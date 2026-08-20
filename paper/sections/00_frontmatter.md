# Front matter (Computer Methods and Programs in Biomedicine)

## Title

**Applied:**  
SEME: A Decoupled Evaluation Protocol for Literature-Constrained EEG–Drug Generators

**Protocol expansion (first Methods/Intro use):** SEME = Signed concordance, Effect-vector fidelity, Magnitude transfer, Encoding retention.

**Case-study framing (not in title line):** demonstrated on a pharmacodynamically penalized Donepezil/Memantine CVAE.

Alternate titles (not applied): see `paper/SEME_PROTOCOL.md`.

---

## Highlights

- SEME protocol separates signs, continuous r, magnitude, and encoding.
- Constrained signs beat unconstrained; locked high r is checkpoint-specific.
- Prior-only reaches 10/10 signs (mean r=0.636); circularity budget made explicit.
- No paired post-dose EEG; endpoint is signature concordance, not PD validation.
- Latent AUC 0.579 vs theta/alpha 0.675 and 2185-D 0.699; head OOF 0.593.

---

## Graphical abstract

File: `paper/figures/graphical_abstract.png`

Direction scoring uses the 2185-D feature-space drug-minus-baseline effect vector, not CVAE latent means. SEME treats signs, continuous \(r\), magnitude, and encoding as separate axes.

---

## Abstract (CMPB structured; ≤350 words)

**Background and Objective:** Chemistry-conditioned EEG generators are often scored with conflated endpoints. We introduce SEME (Signed concordance, Effect-vector fidelity, Magnitude transfer, Encoding retention), a formal evaluation protocol that scores those axes separately for literature-constrained EEG–drug generators, demonstrated on a pharmacodynamically penalized CVAE for Donepezil and Memantine.

**Methods:** Under SEME, a constrained CVAE (\(N=66\); 2185-D EEG features; frozen ChemBERTa embeddings) used subject-level five-fold splits. External cohorts lacked paired pre/post-drug EEG, so signed concordance and continuous effect-vector \(r\) were scored against the training constrained signature on TUH, OSF, and P-ADIC. Core controls were matched unconstrained twins, a prior-only signed effect (circularity budget), linear imitation on CVAE-simulated targets, fold/seed sensitivity, and CAUEEG encoding probes against theta/alpha and packed 2185-D logistic references, with paired DeLong tests on stored out-of-fold scores.

**Results:** On the locked sign endpoint, constrained twins reached 10/10 agreement on TUH, OSF, and P-ADIC while matched unconstrained twins failed (3/10, 4/10, 4/10). Prior-only already reached 10/10 signs at mean \(r=0.636\), so binary signs are partly prior-recoverable. Continuous \(r\) at the locked fold-0 seed-42 checkpoint was high (\(0.908/0.851/0.918\)) but checkpoint-specific (five-fold means \(0.385/0.371/0.372\); non-42 seeds typically 0.08 to 0.43). Magnitude did not diagnose reliably. Encoding bake-off placed latent ROC-AUC \(0.579\) below theta/alpha \(0.675\) and a 2185-D logistic reference of \(0.699\) (best nested-head OOF AUC \(0.593\); DeLong \(p<0.001\) vs 2185-D).

**Conclusions:** SEME shows that constrained sign concordance is the more reproducible external finding in this battery, that continuous fidelity is checkpoint-specific, that prior-only bounds circularity for signs, and that residual latent diagnostic signal sits below classical feature references. The study supports a scoped measurement framework, not post-dose pharmacodynamic validation, seed-stable continuous fidelity, or standalone external diagnosis.

---

## Keywords

SEME evaluation protocol; pharmacodynamically constrained CVAE; EEG-drug simulation; external signature assessment; encoding-level attenuation; reproducibility
