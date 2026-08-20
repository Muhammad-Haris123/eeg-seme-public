# Citation ledger

Maps every quantitative claim in the manuscript to a ground-truth artifact.
Update this file whenever a number is added to any section under `paper/sections/`.

Format:

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| (section:line or quote) | … | `models/validation/…` | `path.to.key` | YYYY-MM-DD |

---

## Spine numbers (locked storyline; verified 2026-08-09)

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| `STORYLINE_LOCKED.md` | Direction agreement 10/10 (TUH) | `complete_validation_report_v3.json` | `layer5.cross_dataset.direction_agreement_total` | 2026-08-09 |
| `STORYLINE_LOCKED.md` | Direction agreement 10/10 (OSF L5b) | `complete_validation_report_v3.json` | `layer5b_ad_labeled_external.cross_dataset.direction_agreement_total` | 2026-08-09 |
| `STORYLINE_LOCKED.md` | Direction agreement 10/10 (P-ADIC) | `complete_validation_report_v3.json` | `layer5d_padic_external.cross_dataset.direction_agreement_total` | 2026-08-09 |
| `STORYLINE_LOCKED.md` | Latent probe ROC-AUC 0.579 | `layer5e_caueeg_latent_probe_results.json` | `latent_probe_dementia_vs_normal.cv.roc_auc.mean` | 2026-08-09 |
| `STORYLINE_LOCKED.md` | Latent probe perm p = 0.010 | `layer5e_caueeg_latent_probe_results.json` | `permutation_test_latent.permutation_p_value` (0.00995) | 2026-08-09 |
| `STORYLINE_LOCKED.md` | Theta/alpha probe ROC-AUC 0.675 | `layer5e_caueeg_latent_probe_results.json` | `feature_probe_theta_alpha_dementia_vs_normal.cv.roc_auc.mean` | 2026-08-09 |
| `STORYLINE_LOCKED.md` | Best head OOF AUC 0.593; readout saturation under tested heads; historical artifact `verdict_tag` retained for provenance | `layer5e_caueeg_classifier_head_results.json` | `best_head.oof_roc_auc`, `verdict_tag` | 2026-08-18 |

## `00_frontmatter.md` (Highlights + Abstract)

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| Highlights bullet 1 | Locked seed-42 fold-0: 10/10; high r checkpoint-specific 0.908/0.851/0.918 | `complete_validation_report_v3.json` | `layer5` / `layer5b` / `layer5d` `.cross_dataset.{direction_agreement_total,effect_magnitude_correlation}` | 2026-08-11 |
| Highlights bullet 2 | Continuous r fold- and train-seed sensitive; signs more stable | `fold_external_direction_results.json`; `multiseed_robustness/FINAL_MULTISEED_REPORT.md` | five-fold means; multi-seed non-42 r range | 2026-08-11 |
| Highlights bullet 3 | Prior-only 10/10 mean r=0.636; ensemble does not clearly exceed prior | `prior_only_direction.json`; `prior_vs_ensemble/` | mean r; verdict | 2026-08-11 |
| Highlights bullet 4 | No paired post-dose EEG (endpoint framing) | Methods / storyline | design statement | 2026-08-11 |
| Highlights bullet 5 | Latent 0.579; θ/α 0.675; 2185-D 0.699; head OOF 0.593 | probe/head/raw2185 JSON | CV/OOF AUC keys | 2026-08-11 |
| Abstract | Feature dim 2185 | `src/models/config.py` / feature pipeline; also twin flat size in validation loaders | architecture convention (PSD+bands+conn) | 2026-08-09 |
| Abstract | TUH n=200 | `complete_validation_report_v3.json` | `layer5.cross_dataset.n_tuh_used` | 2026-08-09 |
| Abstract | TUH r=0.908 | same | `layer5.cross_dataset.effect_magnitude_correlation` | 2026-08-09 |
| Abstract | OSF n=92 | same | `layer5b_ad_labeled_external.cross_dataset.n_used` | 2026-08-09 |
| Abstract | OSF r=0.851 | same | `layer5b_ad_labeled_external.cross_dataset.effect_magnitude_correlation` | 2026-08-09 |
| Abstract | P-ADIC n=145 | same | `layer5d_padic_external.cross_dataset.n_used` | 2026-08-09 |
| Abstract | P-ADIC r=0.918 | same | `layer5d_padic_external.cross_dataset.effect_magnitude_correlation` | 2026-08-09 |
| Abstract | TUH d=-0.13 | same | `layer5.abnormal_vs_normal.cohens_d` (-0.1273) | 2026-08-09 |
| Abstract | OSF p=0.000278, d=0.44, AD n=80, HC n=12 | same | `layer5b_ad_labeled_external.discrimination.{mannwhitney_p,cohens_d,n_ad,n_hc}` | 2026-08-09 |
| Abstract | P-ADIC d=-0.07 | same | `layer5d_padic_external.discrimination.cohens_d` (-0.0652) | 2026-08-09 |
| Abstract | CAUEEG good PCA overlap | same | `layer5e_caueeg_external.pca.pca_overlap` (= good); fraction `overlap_fraction_within_train_r95`=0.611 | 2026-08-09 |
| Abstract | calibrated scale (implied) | same | `layer5e_caueeg_external.scale_check.bandpower_ratio_caueeg_over_train` (=1.104) | 2026-08-09 |
| Abstract | locked fold-0 framing | Methods / checkpoint alias | fold-0 epoch-99 constrained alias | 2026-08-10 |
| Abstract | five-fold mean r 0.385 / 0.371 / 0.372 | `fold_external_direction_results.json` | per-cohort five-fold mean r | 2026-08-10 |
| Abstract | prior-only mean r=0.636 | `prior_only_direction.json` | mean effect-vector correlation | 2026-08-10 |
| Abstract | ensemble r 0.642 / 0.626 / 0.627 | `fold_ensemble/fold_ensemble_external_direction_results.json` | per-cohort `effect_magnitude_correlation` | 2026-08-10 |
| Abstract | ensemble does not clearly exceed prior | `prior_vs_ensemble/` summary | verdict: evidence does not clearly establish incremental agreement | 2026-08-10 |
| Abstract | multi-seed non-42 r typically 0.08 to 0.43; signs mostly 9/10–10/10 | `multiseed_robustness/FINAL_MULTISEED_REPORT.md` / `aggregate_results.csv` | per-seed r and agreement | 2026-08-11 |
| Abstract | strongest claim = directional/sign concordance, not seed-stable continuous r | same + Discussion reframe | interpretation | 2026-08-11 |

## `01_introduction.md`

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| Intro para 1 | Donepezil response 20% to 60% | literature `references.bib` `Lu2020DonepezilGenetics` | Lu et al., Front. Pharmacol. 2020 (review statement) | 2026-08-09 |
| Intro contrib 1 | Direction 10/10; r = 0.908 / 0.851 / 0.918 | `complete_validation_report_v3.json` | `layer5` / `layer5b_ad_labeled_external` / `layer5d_padic_external` `.cross_dataset.{direction_agreement_total,effect_magnitude_correlation}` | 2026-08-09 |
| Intro contrib 2 | TUH p=0.272, d=-0.13 | same | `layer5.abnormal_vs_normal.{mannwhitney_p,cohens_d}` | 2026-08-09 |
| Intro contrib 2 | OSF p=0.000278, d=0.44, HC n=12 | same | `layer5b_ad_labeled_external.discrimination.*` | 2026-08-09 |
| Intro contrib 2 | P-ADIC p=0.968, d=-0.07 | same | `layer5d_padic_external.discrimination.*` | 2026-08-09 |
| Intro contrib 2 | CAUEEG p=0.516, d=0.00, n=727 | same | `layer5e_caueeg_external.discrimination.primary_dementia_vs_normal.*` | 2026-08-09 |
| Intro contrib 3 | theta/alpha AUC=0.675; latent AUC=0.579; perm p=0.010 | `layer5e_caueeg_latent_probe_results.json` | feature / latent CV AUC means; `permutation_test_latent.permutation_p_value` | 2026-08-09 |
| Intro contrib 3 | nested heads OOF AUC=0.593 | `layer5e_caueeg_classifier_head_results.json` | `best_head.oof_roc_auc` | 2026-08-09 |
| Intro para 3 | Feature dim 2185 | feature pipeline / twin loaders | flat 2185-D representation | 2026-08-09 |

## `02_methods.md`

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| §2.1 | latent dims 128/64/256/128; ChemBERTa 384; hidden dims | `src/models/config.py` | `MODEL_CONFIG` | 2026-08-09 |
| §2.1 | constrained_mlp flags | same | `MODEL_VARIANTS['constrained_mlp']` | 2026-08-09 |
| §2.2 | Donepezil/Memantine eeg_effects priors | `src/drugs/pharmacological_embedding.py` | `DRUG_PHARMACOLOGY[*]['eeg_effects']` | 2026-08-09 |
| §2.2 | Donepezil priors: delta -0.15, theta -0.20, alpha +0.25, beta +0.10, connectivity +0.20 | `src/drugs/pharmacological_embedding.py` | `DRUG_PHARMACOLOGY['donepezil']['eeg_effects']` (exact floats) | 2026-08-09 |
| §2.2 | Memantine priors: delta -0.10, theta -0.15, alpha +0.10, beta +0.05, connectivity +0.15 | `src/drugs/pharmacological_embedding.py` | `DRUG_PHARMACOLOGY['memantine']['eeg_effects']` (exact floats) | 2026-08-09 |
| §2.2 | band/connectivity index ranges; warmup 5; weight 0.1; beta_kl 0.01 | `src/models/losses.py`; `config.py`; checkpoint | `BAND_INDICES`, `compute_constrained_total_loss`, ckpt `constraint_weight`/`warmup_epochs` | 2026-08-09 |
| §2.3 | AD 37 / HC 29 | `data/eeg_features/{AD,HC}_psd.npy` shapes | `(37,19,20)`, `(29,19,20)` | 2026-08-09 |
| §2.3 | OpenNeuro accession ds004504 | user-confirmed; also `layer5b_ad_labeled_external_diagnosis.md`, `src/tuh/run_layer5b_ad_labeled.py` | Miltiadous OpenNeuro ds004504 = training cohort | 2026-08-09 |
| §2.3 | Inclusion: HC sub-037..065 (29); AD sub-001..036 plus sub-001 cleaned `.set` (37 files / 36 BIDS folders) | `data/raw_eeg/{AD,HC}/`; `EEGLoader.find_eeg_files` | live file discovery list | 2026-08-09 |
| §2.3 | Training GPU: NVIDIA GeForce RTX 5060 Ti (16 GB) | user-confirmed; `nvidia-smi` on workstation; `models/logs/training_history.json` | `config.training.device` = `cuda`; `nvidia-smi` name/memory | 2026-08-09 |
| §2.3 | 5-fold, seed 42, batch 16, lr 1e-4, wd 1e-5, 100 epochs | `train_phase2_upgraded.py`; `TRAINING_CONFIG`; `DATA_CONFIG` | | 2026-08-09 |
| §2.3 | checkpoint epoch 99 | `models/checkpoints_constrained/checkpoint_constrained.pt` | `epoch` | 2026-08-09 |
| §2.3 | Layer 5e num_samples=5 | `src/tuh/run_layer5e_caueeg.py` | `simulate_post_drug_eeg(..., num_samples=5)` | 2026-08-09 |
| §2.4 | TUH 200; abn 131 / nor 69 | `complete_validation_report_v3.json` | `layer5.processing`; `layer5.abnormal_vs_normal` | 2026-08-09 |
| §2.4 | TUH processing 200/200 | `complete_validation_report_v3.json` | `layer5.processing.total_edf_files`=200; `layer5.processing.successfully_processed`=200 | 2026-08-09 |
| §2.4 | OSF AD 80 / HC 12 | same | `layer5b_ad_labeled_external.discrimination` | 2026-08-09 |
| §2.4 | P-ADIC AD 49 / HC 96 | same | `layer5d_padic_external.discrimination` | 2026-08-09 |
| §2.4 | CAUEEG 436/395/291; ratio 1.104; crop 320; notch 60; target epochs 864 | `layer5e_caueeg_external`; `process_caueeg_eeg.py`; `process_tuh_eeg.py` | `n_by_class`; `scale_check`; `CROP_SECONDS`; `CAUEEG_NOTCH_HZ`; `TARGET_N_EPOCHS_FOR_PSD` | 2026-08-09 |
| §2.4 | CAUEEG 1122/1122 features OK | `complete_validation_report_v3.json` | `layer5e_caueeg_external.n_features`=1122; `n_by_class` sums to 1122 (Normal 436 + MCI 395 + Dementia 291) | 2026-08-09 |
| §2.5 | BAND_SLICES; mean_drug_response formula; probe CV 5 / perm 200 / nested 5×3 | `tuh_validation.py`; Layer 5e runners; probe/head scripts | | 2026-08-09 |
| §2.6 | no multiplicity correction (stated) | Layer 5/5b/5d/5e + probe scripts (absence of correction) | unadjusted p-values only; no `multipletests`/FDR/Bonferroni | 2026-08-09 |
| §2.6 | Python 3.10 environment (`eeg_twin`); RTX 5060 Ti 16 GB for CAUEEG twin pass | project venv; user-confirmed same workstation GPU | hardware sentence in Methods §2.6 | 2026-08-09 |

## `03_results_mechanism.md`

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| §3.1 primary | Direction 10/10 on three cohorts; r = 0.851 to 0.918 | `complete_validation_report_v3.json` | `layer5` / `layer5b` / `layer5d` `.cross_dataset.direction_agreement_total` and `.effect_magnitude_correlation` | 2026-08-09 |
| §3.1 TUH | n=200; 5/5+5/5=10/10; r=0.908 | same | `layer5.cross_dataset` (`n_tuh_used`, agreements, `effect_magnitude_correlation`) | 2026-08-09 |
| §3.1 OSF | n=92; 10/10; r=0.851 | same | `layer5b_ad_labeled_external.cross_dataset` | 2026-08-09 |
| §3.1 P-ADIC | n=145; 10/10; r=0.918 | same | `layer5d_padic_external.cross_dataset` | 2026-08-09 |
| §3.1 Fig. 2 / Table 2 | first cites | `fig2_direction_agreement.*`; `table2_direction_agreement.*` | generated from same JSON keys | 2026-08-09 |
| §3.2 five-fold | mean r 0.385±0.308 / 0.371±0.294 / 0.372±0.310 | `fold_external_direction_results.json` | per-cohort mean±SD r | 2026-08-10 |
| §3.2 multi-seed Table 6 | seed 42 locked r 0.908/0.851/0.918 (10/10); seed 7 0.408/0.408/0.428 (10/10); seed 21 0.154/0.160/0.106 (10/10,10/10,8/10); seed 123 0.147/0.230/0.080 (9/10); seed 2024 0.163/0.182/0.146 (10/10); means 0.356/0.366/0.336 | `multiseed_robustness/aggregate_results.csv` | per-seed r/agreement; aggregate mean/median/SD | 2026-08-11 |
| §3.2 multi-seed range | non-42 TUH 0.147–0.408; OSF 0.160–0.408; P-ADIC 0.080–0.428 | same | min/max among seeds 7/21/123/2024 | 2026-08-11 |
| §3.2 ensemble | r 0.642 / 0.626 / 0.627; bootstrap CIs | `fold_ensemble/bootstrap/ensemble_subject_bootstrap_B2000.json` | locked_ensemble_r; percentile intervals | 2026-08-10 |
| §3.2 ensemble vs prior | does not clearly exceed prior-only 0.636 | `prior_vs_ensemble/` | verdict summary | 2026-08-10 |

## `04_results_diagnosis.md`

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| §3.2 TUH | p=0.272, d=-0.13, n=131/69, FAIL | `complete_validation_report_v3.json` | `layer5.abnormal_vs_normal` | 2026-08-09 |
| §3.2 OSF | p=0.000278, d=0.44, n=80/12, PASS_WITH_CAVEATS | same + `layer5b_results.json` | `layer5b_ad_labeled_external.discrimination`; `honest_verdict.status` | 2026-08-09 |
| §3.2 OSF ablation | disease0 p<0.001 d=0.64; disease1 p<0.001 d=0.34 | `layer5b_results.json` | `sensitivity_disease_label_fixed` | 2026-08-09 |
| §3.2 P-ADIC | p=0.968, d=-0.07, n=49/96, FAIL | `complete_validation_report_v3.json` | `layer5d_padic_external.discrimination` | 2026-08-09 |
| §3.2 P-ADIC ablation | p=0.527 d=-0.16; p=0.702 d=-0.14 | same | `disease_label_ablation` | 2026-08-09 |
| §3.2 P-ADIC ages | AD ~70 vs HC ~52 | `layer5d_padic_external_diagnosis.md` | age table | 2026-08-09 |
| §3.2 CAUEEG primary | p=0.516, d=-0.00, n=291/436, FAIL | `complete_validation_report_v3.json` | `…primary_dementia_vs_normal` | 2026-08-09 |
| §3.2 CAUEEG secondary | Dem/MCI, MCI/Nor, AD-tag, Kruskal H=0.90 p=0.637 | same | `discrimination.*` | 2026-08-09 |
| §3.2 CAUEEG Dem vs MCI | p=0.877, d=0.07 | `complete_validation_report_v3.json` | `layer5e_caueeg_external.discrimination.dementia_vs_mci.mannwhitney_p`=0.8770108…; `.cohens_d`=0.065867… (reported d=0.07) | 2026-08-09 |
| §3.2 CAUEEG MCI vs Normal | p=0.363, d=-0.07 | same | `…discrimination.mci_vs_normal.mannwhitney_p`=0.362825…; `.cohens_d`=-0.072095… (reported d=-0.07) | 2026-08-09 |
| §3.2 CAUEEG AD-tagged Dem vs Normal | n=214 / 436; p=0.624, d=-0.01 | same | `…ad_tagged_dementia_vs_normal.n_dementia_ad_tag`=214; `.n_normal`=436; `.mannwhitney_p`=0.623793…; `.cohens_d`=-0.012750… (reported d=-0.01) | 2026-08-09 |
| §3.2 CAUEEG Kruskal | H=0.90, p=0.637 | same | `…discrimination.kruskal_three_group.H`=0.902321…; `.p`=0.636889… | 2026-08-09 |
| §3.2 CAUEEG ablation | p=0.411 d=-0.04; p=0.776 d=0.00 | same | `disease_label_ablation` | 2026-08-09 |
| §3.2 CAUEEG domain | ratio 1.104; PCA overlap 0.611 | same | `scale_check`; `pca.overlap_fraction_within_train_r95` | 2026-08-09 |
| §3.2 theta/alpha | p<0.001 (1.18e-15), d=0.47 | same | `feature_space_discrimination.theta_alpha_ratio.dementia_vs_normal` | 2026-08-09 |
| §3.2 Fig. 3 / Table 3 / Fig. 4 | first cites | generated figures/tables | manifest | 2026-08-09 |

## `05_results_encoding_analysis.md`

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| §3.3 latent probe | AUC=0.579, perm p=0.010, n=727 | `layer5e_caueeg_latent_probe_results.json` | `latent_probe_dementia_vs_normal.cv.roc_auc.mean`; `permutation_test_latent.permutation_p_value` | 2026-08-09 |
| §3.3 theta/alpha | AUC=0.675 | same | `feature_probe_theta_alpha_dementia_vs_normal.cv.roc_auc.mean` | 2026-08-09 |
| §3.3 gap | ~0.096 (0.675-0.579) | derived from above CV means | | 2026-08-09 |
| §3.3 tuned LogReg | OOF AUC=0.593, perm p=0.005 | `layer5e_caueeg_classifier_head_results.json` | `best_head.oof_roc_auc`; `permutation_best_head.permutation_p_value` | 2026-08-09 |
| §3.3 MLP | OOF AUC=0.543 | same | `heads.mlp.oof.roc_auc` | 2026-08-09 |
| §3.3 HGB | OOF AUC=0.556 | same | `heads.hist_gradient_boosting.oof.roc_auc` | 2026-08-09 |
| §3.3 mu+logvar | OOF AUC=0.587 | same | `mu_logvar_control.oof.roc_auc` | 2026-08-09 |
| §3.3 verdict | readout saturation under tested heads (stored `verdict_tag` string is historical; not a theoretical ceiling) | same | `verdict_tag` | 2026-08-12 |
| §3.3 null band | p5=0.449, p95=0.559 | `layer5e_caueeg_latent_probe_results.json` | `permutation_test_latent.null_auc_percentiles` | 2026-08-09 |
| §3.3 Fig. 5 / Table 4 | first cites | `fig5_encoding_analysis.*`; `table4_encoding_results.*` | manifest | 2026-08-09 |

## `06_discussion.md`

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| §4 para 1 | Direction fold-0; five-fold means; ensemble 0.642/0.626/0.627; prior 0.636; probes 0.579/0.675/0.593 | Results + fold_ensemble + prior_only JSON | same as Results ledger rows | 2026-08-10 |
| §4 para 2 | domain OOF AUC 0.984 / 1.000 / 1.000; broad family separability; PCA overlap 0.611 | `domain_shift/domain_shift_results.json`; `domain_shift_forensics/domain_shift_forensic_summary.md`; v3 PCA | pooled_oof_auc; family table; overlap_fraction | 2026-08-10 |
| §4 para 2 | DADD / personalized EEG twin situating cites | `references.bib` | Amato2025DADD, Amato2026DADDnpj, Amato2025BNM, Simpraga2017mAChR, Johannsson2015AChIndex | 2026-08-09 |
| §4 para 3 | Reconstruction vs classification dissociation precedent | `references.bib` | Rampasek2019DrVAE (Bioinformatics 2019; DOI 10.1093/bioinformatics/btz158) | 2026-08-09 |
| §4 limitations update | OOF DeLong and ChemBERTa/one-hot ablation completed as non-locked Phase C; linear control remains imitation-only | `delong_oof_probe_results.json`; `chemberta_onehot_ablation/chemberta_onehot_results.json`; `linear_direction_baseline/` | comparisons; direction_contrast; scientific_role | 2026-08-20 |
| §4 limitations | rematched arms do not recover locked high-r; single-seed ablation; lock chronology not timestamp-proven | Phase C JSON + audit | design gaps | 2026-08-20 |

## Phase C secondary (2026-08-20)

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| §2.4 / §3.4 | DeLong latent vs θ/α: z=-3.63, p=0.000285 | `delong_oof_probe_results.json` | comparisons latent_probe vs theta_alpha | 2026-08-20 |
| §3.4 | DeLong latent vs 2185-D: z=-4.64, p=3.50e-6 | same | latent_probe vs packed_2185 | 2026-08-20 |
| §3.4 | DeLong best head vs θ/α: z=-3.32, p=0.000909 | same | theta_alpha vs best_nested_head (sign flipped in prose) | 2026-08-20 |
| §3.4 | DeLong best head vs 2185-D: z=-4.09, p=4.34e-5 | same | packed_2185 vs best_nested_head | 2026-08-20 |
| §3.4 | DeLong θ/α vs 2185-D: z=-1.11, p=0.268 | same | theta_alpha vs packed_2185 | 2026-08-20 |
| §3.6 | one-hot signs 10/10; r=0.176/0.194/0.174 (TUH/OSF/P-ADIC) | `chemberta_onehot_ablation/chemberta_onehot_results.json` | arms.onehot.*_direction | 2026-08-20 |
| §3.6 | rematched ChemBERTa signs 9/10; r=0.104/0.110/0.076 | same | arms.chemberta.*_direction | 2026-08-20 |
| §3.6 | latent AUC mean 0.540 vs 0.559 | same | latent_probe_contrast | 2026-08-20 |

## `07_backmatter.md`

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| Conclusion | Direction 10/10; r=0.908/0.851/0.918; magnitude nulls; probe 0.579/0.675/0.593 | Results Sections 3.1 to 3.3 / v3 + probe/head JSON | same as Results ledger | 2026-08-09 |
| Data availability | CAUEEG non-redistribution; Min-jae Kim access | `layer5e_caueeg_external_diagnosis.md` §0 | Access / provenance | 2026-08-09 |
| Data availability | P-ADIC Dryad DOI | `layer5d_padic_external_diagnosis.md` | DOI 10.5061/dryad.8gtht76pw | 2026-08-09 |

## Plan-to-8.3 new artifacts (2026-08-09)

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| B1 gate / Results (pending prose) | Unconstrained TUH direction 3/10 | `unconstrained_external_battery.json` | `cohorts.tuh.direction.direction_agreement_total` | 2026-08-09 |
| B1 gate / Results (pending prose) | Unconstrained OSF/P-ADIC/CAUEEG direction 4/10 | same | `cohorts.{osf,padic,caueeg}.direction.direction_agreement_total` | 2026-08-09 |
| B1 gate / Results (pending prose) | Unconstrained CAUEEG latent AUC mean 0.539 | same | `cohorts.caueeg.encoding.latent_probe.roc_auc_mean` | 2026-08-09 |
| B1 gate / Results (pending prose) | Unconstrained theta/alpha AUC mean 0.675 | same | `cohorts.caueeg.encoding.theta_alpha_probe.roc_auc_mean` | 2026-08-09 |
| B2 prior-only | Prior-only direction 10/10; mean r=0.636 | `prior_only_direction.json` | `direction_agreement_total`; `effect_magnitude_correlation_mean` | 2026-08-09 |
| B2 prior-only | Donepezil prior r=0.572; Memantine r=0.700 | same | `donepezil.effect_corr`; `memantine.effect_corr` | 2026-08-09 |
| C2 Fisher CI | TUH r=0.908 (95% CI 0.900 to 0.915) | `direction_r_fisher_ci.json` | `cohorts.tuh.effect_magnitude_correlation_formatted` | 2026-08-09 |
| C2 Fisher CI | OSF r=0.851 (95% CI 0.839 to 0.862) | same | `cohorts.osf.effect_magnitude_correlation_formatted` | 2026-08-09 |
| C2 Fisher CI | P-ADIC r=0.918 (95% CI 0.911 to 0.924) | same | `cohorts.padic.effect_magnitude_correlation_formatted` | 2026-08-09 |
| Checkpoint rule | Unconstrained = fold_0_best of 5-fold unconstrained | `models/checkpoints_unconstrained/CHECKPOINT_SELECTION.md` | selection rule | 2026-08-09 |
| Backup | Pre-plan restore point | `backups/pre_plan_8_3_20260809_194305/` (+ `.zip`) | RESTORE_README.md | 2026-08-09 |
| B3 \(\alpha_c\) | Direction 10/10 for \(\alpha_c\ge0.05\); peak r at 0.1; AUC non-monotone | `constraint_strength_sweep.json` + `_summary.md` | `points[*].external` | 2026-08-09 |
| B1 gate note | Unconstrained does not beat constrained latent AUC | `paper/B1_GATE_DECISION.md` | gate table | 2026-08-09 |
| §3.1 | Unconstrained TUH/OSF/P-ADIC direction 3/10, 4/10, 4/10 | `unconstrained_external_battery.json` | `cohorts.*.direction.direction_agreement_total` | 2026-08-09 |
| §3.1 | Prior-only 10/10; mean r=0.636 | `prior_only_direction.json` | `direction_agreement_total`; `effect_magnitude_correlation_mean` | 2026-08-09 |
| §3.1 linear imitation control | fold-0 vs locked signature: TUH r=0.851 (10/10), OSF r=0.639 (10/10), P-ADIC r=0.775 (10/10); targets are locked CVAE-simulated Δx | `linear_direction_baseline/linear_direction_baseline_results.json` | `folds[0].primary.{TUH,OSF,P-ADIC}.effect_magnitude_correlation` + `direction_agreement_total`; `model.target_delta` | 2026-08-18 |
| §3.1 linear vs locked CVAE fold-0 (descriptive) | CVAE r=0.908/0.851/0.918 vs linear imitation r=0.851/0.639/0.775 | `complete_validation_report_v3.json`; `linear_direction_baseline/linear_direction_baseline_results.json` | v3 `layer5* cross_dataset.effect_magnitude_correlation`; linear `folds[0].primary.*.effect_magnitude_correlation` | 2026-08-18 |
| §3.2 | Fisher CIs for constrained r | `direction_r_fisher_ci.json` | `cohorts.*.effect_magnitude_correlation_formatted` | 2026-08-09 |
| §3.5 / Fig.6 | \(\alpha_c\) sweep seeded metrics | `constraint_strength_sweep.json` | `points[*].external` | 2026-08-09 |
| Table 5 | Constrained vs unconstrained direction rows | `table5_ablation_direction.md` | generated from battery + v3 | 2026-08-09 |
| §3.1 unconst mag | TUH p=0.124 d=-0.04; OSF p=0.124 d=-0.16; P-ADIC p=0.791 d=0.03; CAUEEG p=0.088 d=-0.07 | `unconstrained_external_battery.json` | `cohorts.*.magnitude` | 2026-08-09 |
| §3.1 unconst r | TUH r=-0.309; OSF r=-0.331; P-ADIC r=-0.327 | same | `cohorts.*.direction.effect_magnitude_correlation` | 2026-08-09 |
| §3.5 sweep | \(\alpha_c\)=0.01 agree 8/10; TUH r=0.096; \(\alpha_c\)=0.05 TUH r=0.299 AUC=0.527; \(\alpha_c\)=0.1 TUH r=0.897; \(\alpha_c\)=0.2 r=0.220 AUC=0.514; \(\alpha_c\)=0.5 r=0.412 AUC=0.578; \(\alpha_c\)=0 TUH r=-0.350 P-ADIC r=-0.364 | `constraint_strength_sweep.json` | `points[*].external` | 2026-08-09 |

## 2026-08-09 must-do statistical / robustness updates

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| §3.2 / Methods | Subject-bootstrap r intervals TUH 0.874–0.900 (boot mean 0.888); OSF 0.790–0.837 (0.815); P-ADIC 0.871–0.904 (0.889); B=2000 | `direction_r_subject_bootstrap_ci.json` | `cohorts.*.ci.effect_magnitude_correlation` | 2026-08-09 |
| §3.2 | Cosine means 0.936 / 0.896 / 0.942 | same | `cohorts.*.point.cosine_mean` | 2026-08-09 |
| §3.2 five-fold | TUH r mean 0.385±0.308; OSF 0.371±0.294; P-ADIC 0.372±0.310; agreements 10/10,10/10,8/10,10/10,9/10 | `fold_external_direction_results.json` | `cohorts.*.*` | 2026-08-09 |
| §3.2 five-fold fold0 re-sim | TUH r=0.897; OSF 0.850; P-ADIC 0.900 | same | `folds.fold_0.cohorts.*.effect_magnitude_correlation` | 2026-08-09 |
| §3.4 / Intro | Raw 2185-D logistic CV mean AUC 0.699 (OOF 0.699) | `layer5e_caueeg_raw2185_probe_results.json` | `raw2185_logistic_dementia_vs_normal.cv.roc_auc.mean` / `oof.roc_auc` | 2026-08-09 |
| Methods lock | α_c=0.1 / fold-0 / epoch-99 locked before external scoring | Methods §2.3 lock statement | protocol statement | 2026-08-09 |

| Manuscript location | Claim / number (exact) | Source file | JSON key / MD heading | Verified |
|---------------------|------------------------|-------------|------------------------|----------|
| Methods §2.3 | Train N=66 small vs 2185-D / latent 128 / widths to 512 | `src/models/config.py`; training cohort counts | architecture + AD 37 / HC 29 | 2026-08-09 |
| Methods §2.6 / Fig.3 | Hedges-Olkin approx 95% CI on Cohen's d | Fig. 3 construction (`make_fig3_diagnostic_null.py`) | SE from d,n1,n2 (not in JSON) | 2026-08-09 |
| §3.4 / Table 5 | θ/α mean-fold AUC 0.675 vs OOF 0.673 | `layer5e_caueeg_latent_probe_results.json` | `feature_probe_...cv.roc_auc.mean` vs `cv.oof.roc_auc` | 2026-08-09 |
| Discussion Limitations | α_c sweep = fold-0 + TUH/P-ADIC only | `constraint_strength_sweep.json` | design note / points cohorts | 2026-08-09 |
| Graphical abstract | Direction 10/10; r triad; latent 0.579; θ/α 0.675; head 0.593; N=66 | spine artifacts above | same keys | 2026-08-09 |
| Title | SEME: A Decoupled Evaluation Protocol for Literature-Constrained EEG–Drug Generators | `00_frontmatter.md` / `SEME_PROTOCOL.md` | Path A applied title | 2026-08-20 |
| Methods / Table A.1 | Parameter count 2,648,777 | `models/validation/computational_requirements.json` | `parameter_count` | 2026-08-11 |
| Methods / Table A.2 | Checkpoint 31,876,749 bytes; dummy inference 0.602 ms/subject; peak alloc 50.2 MiB | same | `checkpoint_bytes`, `ms_per_subject`, `peak_gpu_memory_allocated_mib` | 2026-08-11 |
| Methods / Table A.2 | Python 3.10.0; PyTorch 2.10.0+cu128; NumPy 1.26.4; SciPy 1.15.3; sklearn 1.7.2; MNE 1.11.0; Transformers 5.0.0 | same | package version keys | 2026-08-11 |
| Table 5 | 2185-D logistic OOF AUC 0.699, bal acc 0.642 | `layer5e_caueeg_raw2185_probe_results.json` | `raw2185_logistic_dementia_vs_normal.cv.oof` | 2026-08-11 |
| Fig. 2 caption | Bootstrap percentile intervals TUH 0.874–0.900; OSF 0.790–0.837; P-ADIC 0.871–0.904 | `direction_r_subject_bootstrap_ci.json` | `cohorts.*.ci.effect_magnitude_correlation` | 2026-08-11 |
| Fig. A.1 | Fold-0 train/val loss curves | `constrained_evaluation_report.json` | `constraint_curves.fold_0` | 2026-08-11 |
| Methods §2.3 | Checkpoint epoch 99; val total 0.0241 | `checkpoint_constrained.pt`; `constraint_curves.fold_0` | `epoch`; min `val.total` | 2026-08-11 |
| Discussion | TUH family OOF: connectivity 0.977, band-power 0.742; OSF families \(\ge 0.999\); P-ADIC connectivity 1.000, band 0.981 | `domain_shift_forensic_summary.md` | Feature-family pooled OOF AUC table | 2026-08-11 |
