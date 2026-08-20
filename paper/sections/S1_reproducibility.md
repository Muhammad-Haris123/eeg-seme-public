# Supplementary Material S1. Reproducibility appendix

This appendix lists configuration keys, scripts, and artifact paths used to regenerate the manuscript numbers. It is not required to follow the scientific argument in the main text. Table A.3 lists methodological safeguards and remaining scope boundaries, including non-locked Phase C DeLong and ChemBERTa-versus-one-hot rows.

## Reproduction contract

- Environment: Python 3.10.0 in `eeg_twin/`; PyTorch 2.10.0+cu128; NumPy 1.26.4; SciPy 1.15.3; scikit-learn 1.7.2; MNE 1.11.0; Transformers 5.0.0 (`computational_requirements.json`).
- Hardware: NVIDIA GeForce RTX 5060 Ti (16 GB).
- Split: subject-level `StratifiedKFold` (`n_splits=5`, `random_state=42`); within-fold validation `random_state=42+fold_idx`.
- Locked checkpoint: `models/checkpoints_constrained/checkpoint_constrained.pt` = `fold_0_best.pt`, epoch 99, `constraint_weight=0.1`, selection = minimum fold-0 validation total loss (0.0241).
- Matched unconstrained: `models/checkpoints_unconstrained/checkpoint_unconstrained.pt` from `checkpoints_5fold/fold_0_best.pt`.
- Variant: `constrained_mlp`; 2,648,777 trainable parameters; ChemBERTa embeddings frozen `.npy` files.
- Preprocessing: FIR 0.5–40 Hz, training notch 50 Hz, FastICA with empty default exclusion, 4 s epochs at 90% overlap, per-subject z-score after epoching, Welch `n_per_seg=256` at 500 Hz.
- Feature map: 2185-D, 0-based slices PSD `[0,380)`, bands `[380,475)`, interleaved connectivity `[475,2185)`; \(L_{\mathrm{conn}}\) window `[475,1330)`.
- Dummy-batch inference (not clinical real-time or end-to-end latency): 0.602 ms/subject; dummy throughput 1660 subjects/s; peak dummy-forward GPU memory 50.2 MiB; checkpoint 31,876,749 bytes. Training wall-clock for the locked run is not logged.
- Expected locked direction outputs (fold-0 seed-42): \(r = 0.908 / 0.851 / 0.918\) and 10/10 signs on TUH / OSF / P-ADIC against the training constrained signature.
- Master roll-up: `models/validation/complete_validation_report_v3.json`.

## Model configuration

- Variant: `constrained_mlp` in `src/models/config.py` (`MODEL_VARIANTS['constrained_mlp']`).
- ChemBERTa input dim: `MODEL_CONFIG['drug_encoder']['input_dim'] = 384`.
- CVAE latent dim: `MODEL_CONFIG['cvae']['latent_dim'] = 128`.
- Hidden widths: EEG encoder [256, 512], drug encoder [256, 128], CVAE [512, 256]; `use_attention: False`.
- KL weight: `TRAINING_CONFIG['beta_kl'] = 0.01`.
- Constraint schedule: `warmup_epochs = 5`, paper `constraint_weight = 0.1` in `run_constrained_training.py`. Implemented \(\alpha_c(t)=\alpha_c^{\star}\min(1,(t-T_w)/5)\) for \(t\ge T_w\) with 0-indexed \(t\), so \(\alpha_c(T_w)=0\).
- Parameter count: 2,648,777 (`models/validation/computational_requirements.json`).
- Fold-0 loss histories: `models/evaluation/constrained_evaluation_report.json` → `constraint_curves.fold_0` (Fig. A.1).

## Pharmacodynamic priors and loss helpers

- Priors store: `src/drugs/pharmacological_embedding.py` (`DRUG_PHARMACOLOGY[*]['eeg_effects']`).
- Prior loader: `get_eeg_effect_prior(drug_name)`.
- Band constraint: `compute_band_direction_constraint` in `src/models/losses.py`.
- Connectivity constraint: `compute_connectivity_constraint`.
- Band / connectivity index slices used by external direction scoring: `BAND_SLICES` in `src/tuh/tuh_validation.py` (delta [380:399], theta [399:418], alpha [418:437], beta [437:456], connectivity [475:1330]).
- Flatten order (`api/utils/feature_processor.py` `flatten_features`): PSD [0:380), band powers [380:475), then for each of five bands coherence upper-triangle (171) then PLV (171), so full connectivity is interleaved on [475:2185). Coherence and PLV are **not** contiguous pure blocks. The training connectivity constraint and direction-scoring window `[475:1330)` match `CONNECTIVITY_INDICES` / `BAND_SLICES['connectivity']`; that window mixes interleaved coh/PLV from early bands and must not be cited as an all-coherence block. Do not use `feature_block_ranges()['coherence_block']` as a pure-coherence map (it conflicts with interleaving).

## Training data paths and checkpoint aliases

- Training arrays: `data/eeg_features/{AD,HC}_*.npy` (PSD shapes `(37, 19, 20)` and `(29, 19, 20)`).
- Raw EEGLAB discovery: `EEGLoader.find_eeg_files` under `data/raw_eeg/{AD,HC}/`.
- Subject-level CV helper: `get_subject_level_kfold` (`n_splits = 5`, `random_seed = 42` from `DATA_CONFIG`).
- Constrained evaluation alias: `models/checkpoints_constrained/checkpoint_constrained.pt` (fold-0 best).
- Matched unconstrained alias: `models/checkpoints_unconstrained/checkpoint_unconstrained.pt` (from `checkpoints_5fold/fold_0_best.pt`).
- Simulation draws: `num_samples = 5` in `simulate_post_drug_eeg` for reported external runs.

## External processing and validation artifacts

- TUH processing: `process_tuh_eeg.py` (`CROP_SECONDS = 320`, `TARGET_N_EPOCHS_FOR_PSD = 864`). For locked TUH results, the stored feature provenance in `data/tuh_features/processing_report.json` records `crop_seconds=60.0` and `apply_ica=false`; the 320-s value reflects the current script default and was not used to regenerate the locked TUH artifacts.
- CAUEEG processing: `src/tuh/process_caueeg_eeg.py`.
- Master report: `models/validation/complete_validation_report_v3.json`.
- Unconstrained battery: `models/validation/unconstrained_external_battery.json`.
- Prior-only baseline: `models/validation/prior_only_direction.json`.
- Fisher-\(z\) CIs for \(r\): `models/validation/direction_r_fisher_ci.json`.
- \(\alpha_c\) sweep: `models/validation/constraint_strength_sweep.json`.
- Latent probe: `models/validation/layer5e_caueeg_latent_probe_results.json`.
- Nested heads: `models/validation/layer5e_caueeg_classifier_head_results.json`.

## Endpoint implementation notes

- Direction correlation key in JSON: `effect_magnitude_correlation`.
- Subject-bootstrap CIs: `models/validation/direction_r_subject_bootstrap_ci.json` (B=2000).
- Five-fold external direction: `models/validation/fold_external_direction_results.json`.
- Equal-weight fold ensemble: `models/validation/fold_ensemble/fold_ensemble_external_direction_results.json`.
- Ensemble subject bootstrap (B=2000, seed 42): `models/validation/fold_ensemble/bootstrap/ensemble_subject_bootstrap_B2000.json`.
- Prior vs ensemble incremental-value note: `models/validation/prior_vs_ensemble/` (paired \(\Delta r\) not supportable from stored prior-only artifacts).
- Domain-shift LR AUCs (2185-D): `models/validation/domain_shift/domain_shift_results.json`.
- Domain-shift forensics (feature families): `models/validation/domain_shift_forensics/domain_shift_forensic_summary.md`.
- Feature dimension map (interleaved connectivity): `models/validation/domain_shift_forensics/feature_dimension_mapping.md`.
- Raw 2185-D probe: `models/validation/layer5e_caueeg_raw2185_probe_results.json`.
- Magnitude score helper: `mean_drug_response`.
- Cohen's d helper: `_cohens_d` (sample variance, `ddof = 1`).
- Encoding probe scripts: `run_latent_probe_caueeg.py` (`PROBE_DISEASE_LABEL = 0.0`), `run_latent_classifier_head_caueeg.py`.
- Probe CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`; `N_PERM = 200`.
- Theta/alpha mean-fold AUC 0.675 vs pooled OOF AUC 0.673: `feature_probe_theta_alpha_dementia_vs_normal.cv.roc_auc.mean` vs `.cv.oof.roc_auc`.
- Raw 2185-D mean-fold AUC 0.699: `raw2185_logistic_dementia_vs_normal.cv.roc_auc.mean`.

## Architecture, compute, and training curves

Layer widths and the measured parameter count (2,648,777) are listed in Table A.1. Dummy-batch inference time, checkpoint size, and package versions are listed in Table A.2. Fold-0 constrained train/validation total loss, MSE, KL, and unweighted \(L_{\mathrm{band}}\)/\(L_{\mathrm{conn}}\) with \(\alpha_c(t)\) are stored in `constraint_curves.fold_0`. Fig. A.1 plots those logged values. Alternate-seed loss histories were not found in that report. The figure is a convergence diagnostic, not an external-performance claim.

## Regeneration entry points

- `eeg_twin/Scripts/python.exe -m src.validation.run_prior_only_direction`
- `eeg_twin/Scripts/python.exe -m src.validation.compute_direction_r_cis`
- `eeg_twin/Scripts/python.exe -m src.validation.run_unconstrained_external_battery`
- `eeg_twin/Scripts/python.exe -m src.validation.run_alpha_c_sweep --eval-only`
- `eeg_twin/Scripts/python.exe -m src.validation.run_delong_oof_probes`
- `eeg_twin/Scripts/python.exe -m src.validation.run_chemberta_onehot_ablation --eval-only`
- `eeg_twin/Scripts/python.exe src/figures/make_fig6_constraint_strength.py`
- `eeg_twin/Scripts/python.exe src/figures/make_graphical_abstract.py`

See also `paper/CODE_AVAILABILITY.md`.

## Known missing artifacts

These items are not present in the locked record and are not invented here: locked-run training wall-clock; alternate-seed training curves; a public code URL and commit hash; the native EEG reference of the ds004504 files on disk; a supervised deep 2185-D classifier. Non-locked secondary Phase C artifacts now exist separately: DeLong on stored OOF scores (`models/validation/delong_oof_probe_results.json`) and a ChemBERTa-versus-padded-one-hot fold-0 ablation (`models/validation/chemberta_onehot_ablation/`). A non-locked fitted drug-conditioned linear imitation control is stored under `models/validation/linear_direction_baseline/` and is not a locked biological validation artifact. Table A.3 restates the corresponding scope boundaries.
