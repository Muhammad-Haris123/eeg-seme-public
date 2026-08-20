# SEME protocol (locked identity for CMPB Path A)

**Status:** LOCKED for Path A (2026-08-20)  
**Full name:** **SEME** — Signed concordance, Effect-vector fidelity, Magnitude transfer, Encoding retention  
**One-line:** A decoupled evaluation protocol for literature-constrained, chemistry-conditioned EEG generative models.

## What SEME scores (four axes)

| Axis | Letter | Endpoint family in this paper |
|------|--------|--------------------------------|
| **S** | Signed concordance | Block-level sign agreement vs training constrained signature |
| **E** | Effect-vector fidelity | Continuous Pearson \(r\) on cohort-mean drug−baseline vectors (+ fold/seed sensitivity) |
| **M** | Magnitude transfer | Latent shift magnitude \(\bar{s}\) as diagnostic discriminator |
| **E** | Encoding retention | Residual label information in \(\mu_{\mathrm{base}}\) vs classical feature references |

## Applied title (methods first)

**SEME: A Decoupled Evaluation Protocol for Literature-Constrained EEG–Drug Generators**

Case-study subtitle (not required in title line): demonstrated on a pharmacodynamically penalized Donepezil/Memantine CVAE.

## Alternate titles considered (not applied)

1. Decoupled Endpoint Evaluation of Constrained EEG–Drug Generators: The SEME Protocol with a Donepezil/Memantine CVAE Case Study  
2. Separating Signature Concordance, Continuous Fidelity, Magnitude, and Encoding in Constrained EEG–Drug Simulation: SEME and Case Study  

## Contribution contract (unchanged science)

Matches `paper/CMPB_PHASE_A_JOURNAL_STRATEGY.md` §4. SEME is the **product**; the CVAE is the **worked example**.

## Hostile-read pass criteria

Title + Abstract + Conclusion must not support: post-dose PD validation; seed-stable continuous \(r\); diagnostic product; ChemBERTa chemical generalization; “model discovered pharmacology.”
