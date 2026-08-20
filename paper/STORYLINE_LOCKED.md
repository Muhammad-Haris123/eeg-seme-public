# Locked storyline (CBM manuscript)

**Status:** CONTROLLED REOPEN (2026-08-09) for claim *language* only under `paper/STORYLINE_REOPEN_8_3.md` / `paper/PLAN_TO_8_3.md`. Spine numbers unchanged until new JSON updates them.
**Verified against:** `complete_validation_report_v3.json`, `layer5e_caueeg_latent_probe_results.json`, `layer5e_caueeg_classifier_head_results.json` (2026-08-09).
**Backup before reopen:** `backups/pre_plan_8_3_20260809_194305/` (+ `.zip`).

---

## Central claim (quoteable spine)

A pharmacodynamically constrained CVAE digital twin produces **constraint-consistent**, model-generated Donepezil/Memantine band/connectivity signs that agree with the **training constrained signature** at **10/10** on **three** external EEG cohorts (TUH, OSF AD/HC, P-ADIC). This is signature preservation under domain shift, **not** empirical pharmacodynamic transfer (no paired post-dose EEG in these cohorts). The same model's drug-response-**magnitude** endpoint does **not** carry external diagnostic signal across an escalating sequence of four external tests (TUH proxy fail; OSF positive only with HC n caveats; P-ADIC AD/HC null; CAUEEG Dementia vs Normal null). A targeted encoding-level analysis shows this is not a data-scarcity problem and not a readout-complexity problem: measurable clinical signal survives encoding (latent probe ROC-AUC **0.579**, permutation p **0.010**) but less than a simple spectral marker (theta/alpha ROC-AUC **0.675**), and nested stronger classifiers do not close that gap (best head OOF AUC **0.593**; stored artifact tag is readout saturation under tested heads, not a theoretical ceiling). Whether pharmacodynamic constraints **trade** diagnostic information for mechanistic consistency is tested with matched unconstrained and \(\alpha_c\) experiments (Plan-to-8.3). B1 gate (2026-08-09): matched unconstrained fails direction (3–4/10) while constrained is 10/10, but unconstrained latent AUC is **not** higher than constrained (0.539 vs 0.579); do **not** claim constraint-caused diagnostic compression from B1 alone.


### Precision notes (do not blur when drafting)

| Claim in prose | Exact support | Do not say |
|----------------|---------------|------------|
| Direction transfers, replicated 3x | TUH, OSF (L5b), P-ADIC each `direction_agreement_total` = `10/10` in v3 | Do not claim CAUEEG also has a 10/10 direction result unless that analysis is added and logged |
| Four independent EEG datasets | TUH, OSF, P-ADIC, CAUEEG in the escalating diagnosis/magnitude investigation | Do not imply all four have identical endpoints |
| Latent probe AUC 0.579 | `layer5e_caueeg_latent_probe_results.json` → `latent_probe_dementia_vs_normal.cv.roc_auc.mean` | Do not round to 0.58 in the locked claim; manuscript may use 0.579 |
| perm p = 0.010 | same file → `permutation_test_latent.permutation_p_value` ≈ 0.00995 | Report as p = 0.010 |
| theta/alpha AUC 0.675 | same file → `feature_probe_theta_alpha_dementia_vs_normal.cv.roc_auc.mean` | |
| Readout does not close gap | `layer5e_caueeg_classifier_head_results.json` best OOF AUC 0.593; `verdict_tag` records readout saturation under tested heads, not a theoretical ceiling | Do not claim MLP/HGB recovered theta/alpha performance |

---

## Narrative arc (section jobs)

### Introduction
Motivate why a mechanistic, interpretable digital twin matters for AD drug-response simulation. Argue that "does it discriminate AD from controls on latent magnitude" is the wrong first question; "does it simulate the constrained drug-response mechanism correctly" is the right one. End by previewing the trade-off the paper will measure.

### Methods
Build the twin (EEG features, ChemBERTa drug conditioning, CVAE, pharmacodynamic constraints). State the constraints as a **design decision**, not an afterthought. Define endpoints clearly: direction agreement vs magnitude discrimination vs encoding probes.

### Results Part A (mechanism)
Direction agreement, **10/10**, replicated on **three** external cohorts (TUH, OSF, P-ADIC). This is the twin doing what it was built to do. Open by linking from Methods' endpoint definitions; close by asking whether magnitude also carries diagnostic signal.

### Results Part B (diagnosis)
External magnitude-discrimination nulls, escalating across four datasets. Each dataset answers a doubt left by the previous one (proxy labels → AD labels small HC → large NIA-AA AD/HC → large hospital Normal/MCI/Dementia). Report nulls with the same rigor as positives. Close by asking whether the null means no clinical EEG signal, or loss inside the twin.

### Results Part C (why) — climax
Encoding-level dissection on CAUEEG: theta/alpha proves clinical signal in inputs; baseline μ probe shows partial survival (0.579 vs chance); nested heads show readout is not the bottleneck (best head OOF AUC 0.593; readout saturation under tested heads, not a theoretical ceiling). Treat this as the intellectual climax of Results. Close by naming the trade-off for Discussion.

### Discussion
Name the trade-off explicitly (mechanistic transfer vs diagnostic compression). Situate against DADD-class twins and AChEI-EEG response literature. State what would need to change architecturally to close the gap (and what would not: more complex readout alone). Limitations: encoding loss, dementia-label heterogeneity, no drug-outcome ground truth in CAUEEG.

### Conclusion
One paragraph a reviewer could quote: what this paper establishes as fact (mechanism transfers; magnitude does not diagnose externally; encoding analysis localizes the information loss and quantifies the trade-off).

---

## Escalation map (Part B datasets)

| Dataset | Question it was brought in to resolve | What prior left open |
|---------|----------------------------------------|----------------------|
| TUH | Does magnitude discriminate on independent hospital EEG? | Training-only / in-distribution checks |
| OSF AD/HC | Was TUH null only because labels were abnormal/normal proxies? | Need true AD vs HC |
| P-ADIC | Does a larger NIA-AA AD/HC set confirm OSF, or was OSF HC n=12 fragile? | OSF imbalance / caveats |
| CAUEEG | On large clinical Normal/MCI/Dementia EEG with good scale/PCA, is magnitude still null? | Need large-N clinical strata |

---

## Out of scope for this paper (do not smuggle in)

- Claiming clinically validated treatment-response prediction
- High-tier Path 1 Rx-outcome results (future work only, specific)
- Retraining or architecture changes as completed experiments
- Presenting layers as a flat "5 / 5b / 5d / 5e" dump without the escalation prose

---

## Reopen criteria

Only reopen this lock if a new validation artifact changes a spine number (direction counts, probe AUCs, head verdict) or if coauthors explicitly revise the central claim.
