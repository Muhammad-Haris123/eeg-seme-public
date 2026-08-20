# Table 3. Diagnostic magnitude discrimination results

**Caption (place ABOVE the table, Elsevier):** Twin latent drug-response magnitude discrimination across escalating external cohorts. Status strings are taken from the validation JSON (`FAIL`, `PASS_WITH_CAVEATS`). OSF primary uses `honest_verdict.status` (`PASS_WITH_CAVEATS`); OSF disease-label sensitivity rows have no status field (shown as —). Kruskal–Wallis row reports H instead of Cohen's d and Kruskal p in the p column.

**Placement judgment:** Prefer **Supplementary Table** (or main-text Table with primary contrasts only). Full table has 14 contrast rows + 4 dataset section breaks; dense for a single-column main text.

| Dataset | Contrast | N (group A / group B) | Mann-Whitney p | Cohen's d | Status |
|---------|----------|----------------------|----------------|-----------|--------|
| **TUH** | | | | | |
| TUH | Abnormal vs normal | 131 / 69 | 0.272 | -0.13 | FAIL |
| **OSF** | | | | | |
| OSF | AD vs HC | 80 / 12 | p < 0.001 | 0.44 | PASS_WITH_CAVEATS |
| OSF | AD vs HC (disease=0) | 80 / 12 | p < 0.001 | 0.64 | — |
| OSF | AD vs HC (disease=1) | 80 / 12 | p < 0.001 | 0.34 | — |
| **P-ADIC** | | | | | |
| P-ADIC | AD vs HC | 49 / 96 | 0.968 | -0.07 | FAIL |
| P-ADIC | AD vs HC (disease=0) | 49 / 96 | 0.527 | -0.16 | FAIL |
| P-ADIC | AD vs HC (disease=1) | 49 / 96 | 0.702 | -0.14 | FAIL |
| **CAUEEG** | | | | | |
| CAUEEG | Dementia vs Normal | 291 / 436 | 0.516 | -0.00 | FAIL |
| CAUEEG | Dementia vs MCI | 291 / 395 | 0.877 | 0.07 | FAIL |
| CAUEEG | MCI vs Normal | 395 / 436 | 0.363 | -0.07 | FAIL |
| CAUEEG | AD-tagged Dementia vs Normal | 214 / 436 | 0.624 | -0.01 | FAIL |
| CAUEEG | Normal / MCI / Dementia (Kruskal–Wallis) | 436 / 395 / 291 | Kruskal 0.637 | H = 0.90 | FAIL |
| CAUEEG | Dementia vs Normal (disease=0) | 291 / 436 | 0.411 | -0.04 | FAIL |
| CAUEEG | Dementia vs Normal (disease=1) | 291 / 436 | 0.776 | 0.00 | FAIL |

## Source notes

- TUH: `complete_validation_report_v3.json` → `layer5.abnormal_vs_normal`
- OSF: `layer5b_ad_labeled_external.discrimination`; `data/ad_labeled_external/validation/layer5b_results.json` (`honest_verdict`, `sensitivity_disease_label_fixed`)
- P-ADIC: `layer5d_padic_external.discrimination` + `disease_label_ablation`
- CAUEEG: `layer5e_caueeg_external.discrimination` (pairwise + `kruskal_three_group`) + `disease_label_ablation`
