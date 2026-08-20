# FINAL MULTI-SEED ROBUSTNESS REPORT

Generated: 2026-08-11  
Experiment: locked fold-0 constrained CVAE training-seed sensitivity  
Manuscript edits: **NONE**  
Protected science changes: **0**

---

## 1. Backup

`backups/pre_multiseed_robustness_20260810_235807/`

## 2. Seeds

Prespecified: **42 (locked reference, not retrained), 7, 21, 123, 2024**  
All completed. Failures: **0**

## 3. Training protocol

**Held fixed**

- Dataset, preprocessing, architecture, αc=0.1, ChemBERTa, disease handling
- Optimizer / LR / batch / epochs / constraint loss
- **Fold assignment + fold-0 train/val split frozen at `split_seed=42`**
- Checkpoint rule: min internal validation total loss on fold 0
- Fixed training constrained signature for scoring
- External cohorts: TUH / OSF / P-ADIC; identical direction endpoint

**Only intended change**

- `train_seed` controlling torch/numpy initialization and DataLoader shuffle

**Critical STOP-gate finding (resolved by design)**

Naive change of `DATA_CONFIG['random_seed']` would also change subject fold membership. This experiment therefore **does not** change `DATA_CONFIG.random_seed`. It freezes splits at seed 42 and varies training stochasticity only.

Outputs live only under `models/validation/multiseed_robustness/`.

## 4. Results

| Cohort | Seed | Agreement | r | Cosine |
| --- | ---: | ---: | ---: | ---: |
| TUH | 42 | 10/10 | 0.908024 | 0.927613 |
| OSF | 42 | 10/10 | 0.851003 | 0.884206 |
| P-ADIC | 42 | 10/10 | 0.917640 | 0.942843 |
| TUH | 7 | 10/10 | 0.407892 | 0.592086 |
| OSF | 7 | 10/10 | 0.407638 | 0.588463 |
| P-ADIC | 7 | 10/10 | 0.427859 | 0.608313 |
| TUH | 21 | 10/10 | 0.154262 | 0.054284 |
| OSF | 21 | 10/10 | 0.159845 | 0.117509 |
| P-ADIC | 21 | 8/10 | 0.105660 | -0.046439 |
| TUH | 123 | 9/10 | 0.146906 | 0.025395 |
| OSF | 123 | 9/10 | 0.230114 | 0.209951 |
| P-ADIC | 123 | 9/10 | 0.080451 | -0.124799 |
| TUH | 2024 | 10/10 | 0.163031 | 0.018633 |
| OSF | 2024 | 10/10 | 0.182212 | 0.088491 |
| P-ADIC | 2024 | 10/10 | 0.146115 | 0.016763 |

Seed-42 r/agreement are the **locked** `complete_validation_report_v3.json` values. Cosine for seed 42 was recomputed from the locked checkpoint for metric parity only.

## 5. Aggregate robustness

| Cohort | Mean r | Median r | SD | Min | Max | 10/10 seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TUH | 0.356 | 0.163 | 0.328 | 0.147 | 0.908 | 4/5 |
| OSF | 0.366 | 0.230 | 0.288 | 0.160 | 0.851 | 4/5 |
| P-ADIC | 0.336 | 0.146 | 0.354 | 0.080 | 0.918 | 3/5 |

Agreement summary:

- Often near-perfect signs (many 10/10; some 9/10; one 8/10)
- Continuous r collapses for non-42 seeds toward the same low range seen in five-fold sensitivity (~0.1–0.4)

## 6. Seed-42 comparison

Mean absolute Δr vs locked seed 42 (new seeds only):

- TUH ≈ 0.690
- OSF ≈ 0.606
- P-ADIC ≈ 0.728

No new seed reproduces the locked high-r regime (~0.85–0.92).

## 7. Interpretation

**Evidence suggests substantial training-seed sensitivity**

Answer to the scientific question:

> The locked fold-0 high continuous external correlations appear **seed-specific** under ordinary training stochasticity with fixed fold assignment. Directional **sign** agreement is much more stable than continuous r, but the headline high-r result does **not** persist across the prespecified additional seeds.

This does **not** erase:

- fold dependence
- prior-only baseline
- no paired post-dose EEG
- soft historical seed-42 provenance
- small N / domain-shift limitations

## 8. Provenance

Every new seed has documented:

- training start/end
- checkpoint selection time / rule / epoch
- pre-external freeze
- config hash
- checkpoint SHA256
- external scoring start/end

See `seed_manifest.json`.

This establishes forward provenance for the robustness experiment. It does **not** prove historical seed-42 pre-registration.

## 9. Integrity

- Protected files checked: 10
- Changed: **0**
- Manuscript changed: **NO**
- Existing locked checkpoint changed: **NO**
- Status: CLEAN (`integrity_report.txt`)

## 10. Recommendation (next scientific step only)

**Treat the locked seed-42 high-r fold-0 result as a seed-sensitive observation; if the manuscript is updated later, disclose that continuous external fidelity was not reproduced under additional training seeds with fixed fold-0 splits (analysis already complete; do not change locked numbers).**

Do not integrate into the manuscript in this run.

---

## Artifacts

- `models/validation/multiseed_robustness/aggregate_results.csv`
- `models/validation/multiseed_robustness/aggregate_results.json`
- `models/validation/multiseed_robustness/seed_level_summary.md`
- `models/validation/multiseed_robustness/multiseed_config.json`
- `models/validation/multiseed_robustness/seed_manifest.json`
- `models/validation/multiseed_robustness/integrity_report.txt`
- `models/validation/multiseed_robustness/seed_{7,21,123,2024}/checkpoint.pt`
- `models/validation/multiseed_robustness/run_log.txt`
