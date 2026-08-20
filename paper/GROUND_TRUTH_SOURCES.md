# Ground-truth sources for the CBM manuscript

**Rule:** Do not invent numbers. Every quantitative claim must cite a path below
(or a figure/table derived from them). Prefer `complete_validation_report_v3.json`
as the master roll-up; use layer-specific JSON/MD for keys not duplicated there.

**Storyline:** `paper/STORYLINE_LOCKED.md` is locked. Every section must serve that spine.

Primary index for agents: read this file and `STORYLINE_LOCKED.md` before drafting
any manuscript sentence.

---

## Master validation roll-up

| Path | Role |
|------|------|
| `models/validation/complete_validation_report_v3.json` | Authoritative multi-layer report (L1–L5e). Prefer this for status, p, d, n. |
| `models/validation/complete_validation_report_v2.json` | Prior roll-up (historical only; do not prefer over v3). |
| `models/validation/complete_validation_report.json` | Oldest roll-up (historical only). |
| `models/validation/adni_correlation_results.json` | ADNI ecological / Layer 4 detail. |

---

## External clinical layers (escalating narrative)

### TUH proxy (Layer 5)

| Path | Role |
|------|------|
| `models/validation/complete_validation_report_v3.json` → key `layer5` | Primary numbers. |
| `models/validation/layer5_domain_shift_diagnosis.md` | Narrative diagnosis. |

### OSF AD/HC (Layer 5b)

| Path | Role |
|------|------|
| `models/validation/complete_validation_report_v3.json` → key `layer5b_ad_labeled_external` | Primary numbers. |
| `models/validation/layer5b_ad_labeled_external_diagnosis.md` | Narrative diagnosis. |
| `data/ad_labeled_external/validation/layer5b_results.json` | Detailed Layer 5b run output. |

### TUEG mining (Layer 5c; stopped)

| Path | Role |
|------|------|
| `models/validation/layer5c_tueg_report_mining_diagnosis.md` | Stopped; optional methods footnote only. |

### P-ADIC AD/HC (Layer 5d)

| Path | Role |
|------|------|
| `models/validation/layer5d_padic_results.json` | Primary Layer 5d numbers. |
| `models/validation/complete_validation_report_v3.json` → key `layer5d_padic_external` | Annotated copy in v3. |
| `models/validation/layer5d_padic_external_diagnosis.md` | Narrative diagnosis. |
| `models/validation/figures/layer5d_padic_pca.png` | PCA figure source. |

### CAUEEG Normal/MCI/Dementia (Layer 5e)

| Path | Role |
|------|------|
| `models/validation/layer5e_caueeg_results.json` | Twin magnitude discrimination + scale/PCA. |
| `models/validation/complete_validation_report_v3.json` → key `layer5e_caueeg_external` | Annotated copy in v3. |
| `models/validation/layer5e_caueeg_external_diagnosis.md` | Narrative diagnosis (incl. feature-space θ/α). |
| `models/validation/layer5e_caueeg_latent_probe_results.json` | Baseline μ linear probe + permutation + θ/α probe. |
| `models/validation/layer5e_caueeg_latent_probe_summary.md` | Probe interpretation. |
| `models/validation/layer5e_caueeg_classifier_head_results.json` | Nested-CV stronger heads + perm + μ+logvar. |
| `models/validation/layer5e_caueeg_classifier_head_summary.md` | Head bake-off verdict. |
| `models/validation/figures/layer5e_caueeg_pca.png` | PCA figure source. |
| `data/caueeg_external/processing_report.json` | Feature extraction n_ok / class counts / epochs. |
| `data/caueeg_external/validation/baseline_latents.npz` | Frozen μ/logvar used by probes (not prose-citable alone). |

---

## Planning / readiness (not primary numeric sources)

| Path | Role |
|------|------|
| `models/validation/publication_readiness_assessment.md` | Venue strategy context. |
| `models/validation/high_tier_upgrade_plan.md` | Future Path 1 plan; do not treat as results. |

---

## Architecture and training (CVAE)

| Path | Role |
|------|------|
| `src/models/config.py` | `MODEL_CONFIG`, `TRAINING_CONFIG`, constrained_mlp flags. |
| `src/models/losses.py` | Pharmacodynamic constrained loss definitions. |
| `src/models/train_phase2_upgraded.py` | Constrained training loop. |
| `models/checkpoints_constrained/checkpoint_constrained.pt` | Frozen twin used for L5e / probes (epoch 99). |
| `models/checkpoints_constrained/fold_*_best.pt` | 5-fold constrained checkpoints (if reporting CV). |

Supporting modules (cite only if methods detail needs them):

- `src/models/inference.py`
- `src/tuh/process_tuh_eeg.py` (shared feature path, `TARGET_N_EPOCHS_FOR_PSD=864`)
- `src/tuh/process_caueeg_eeg.py`
- `src/tuh/run_layer5e_caueeg.py`
- `src/tuh/run_latent_probe_caueeg.py`
- `src/tuh/run_latent_classifier_head_caueeg.py`

---

## Paper bookkeeping (this folder)

| Path | Role |
|------|------|
| `paper/citation_ledger.md` | Every manuscript number → source key. |
| `paper/figures_tables_manifest.md` | Fig/Table order and first-cite sentences. |
| `paper/abbreviations.md` | Abbreviation list. |
| `paper/references.bib` | BibTeX for elsarticle-num. |
| `paper/sections/` | Draft section files. |
| `paper/figures/` | Manuscript figure copies (final numbering). |
| `paper/tables/` | Manuscript table sources. |
