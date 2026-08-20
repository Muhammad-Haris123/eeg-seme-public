# Manuscript draft (Computers in Biology and Medicine)

_Assembled from `paper/sections/` for review. Figure/table assets live under `paper/figures/` and `paper/tables/`. Do not treat this file as camera-ready LaTeX._

# Pharmacodynamic Direction Transfers, Diagnostic Magnitude Does Not: Quantifying an Encoding Trade-off in a Constrained EEG-Drug CVAE Twin for Alzheimer's Disease

## Highlights

- Constraint-consistent band direction agreed 10/10 on three external EEG cohorts.
- No paired post-dose EEG; endpoint is signature agreement, not empirical PD transfer.
- CAUEEG magnitude null: Dementia vs Normal p=0.516, d=0.00 (n=727).
- OSF magnitude exploratory (HC n=12); P-ADIC AD/HC magnitude null (p=0.968).
- Latent probe AUC=0.579 vs spectral reference AUC=0.675; head OOF AUC=0.593.

---

## Graphical abstract

[Placeholder: constraint-consistent direction panel (10/10 vs training signature on three cohorts); diagnosis panel with magnitude nulls (OSF exploratory); encoding panel with AUC 0.579 vs 0.675; optional \(\alpha_c\) trade-off panel after B3.]

---

## Abstract

Chemistry-conditioned EEG digital twins for drug-response **simulation** must show that constrained, model-generated pharmacology remains consistent under external domain shift, and must separately justify any diagnostic use of twin outputs. Most EEG twin and generative models emphasize one check, not both. We trained a pharmacodynamically constrained conditional variational autoencoder (CVAE) on 2185-dimensional resting-state EEG features with fixed chemical Bidirectional Encoder Representations from Transformers (ChemBERTa) embeddings for the two study drugs Donepezil and Memantine (a conditioner, not a large-library chemical benchmark). Primary mechanistic endpoint: agreement of simulated band/connectivity **signs** with the **training constrained signature** (not paired post-dose EEG). Direction agreed at 10/10 on three external cohorts: Temple University Hospital (TUH) EEG (n = 200; effect-vector correlation r = 0.908), an Open Science Framework (OSF) AD/healthy control (HC) set (n = 92; r = 0.851), and p-adic quantum potential EEG (P-ADIC) AD/HC recordings (n = 145; r = 0.918). Latent drug-response magnitude did not diagnose reliably across four escalating external tests. TUH abnormal versus normal was null (p = 0.272, d = -0.13). OSF AD versus HC passed with caveats (p = 0.000278, d = 0.44, HC n = 12; exploratory). P-ADIC AD versus HC was null (p = 0.968, d = -0.07). CAUEEG Dementia versus Normal (n = 727) was null (p = 0.516, d = 0.00) despite calibrated scale and good PCA overlap. On CAUEEG, a logistic probe on baseline latent means reached ROC-AUC = 0.579 (permutation p = 0.010), below a theta/alpha spectral reference (ROC-AUC = 0.675), and nested heads peaked at out-of-fold ROC-AUC = 0.593. A matched unconstrained fold-0 twin failed the same direction endpoint (3/10 to 4/10 across cohorts) while its CAUEEG latent probe mean AUC was 0.539, so constraints are required for signature preservation but B1 does not show unconstrained recovery of diagnostic latent information. A prior-only signed effect matched the constrained signature at 10/10 on signs with lower effect-vector correlation (mean r = 0.636). The constrained twin therefore preserves constraint-consistent response **direction** under domain shift; diagnostic **magnitude** fails externally; encoding probes show partial clinical signal at \(\mu_{\mathrm{base}}\) below the spectral reference.

**(Word count: ~250.)**

## Keywords

pharmacodynamically constrained CVAE; EEG digital twin; drug-conditioned simulation; Alzheimer's disease; external validation; encoding information loss


---


# 1. Introduction

Digital twins and chemistry-conditioned generative models are increasingly used to simulate how a named intervention would change a patient's physiological features. For such models to be scientifically usable, two questions must be answered separately. First, does a simulated drug-response mechanism transfer outside the training cohort? Second, which twin-derived scalars, if any, carry diagnostic signal under the same external conditions? Conflating those questions invites a common failure mode: a model can look clinically useful on in-distribution diagnosis while its constrained pharmacology does not generalize, or the reverse. Conditional variational autoencoders (CVAEs) and related generative models make that risk concrete, because reconstruction quality, perturbation fidelity, and downstream classification need not move together [Kingma2014VAE, Higgins2017betaVAE, Rampasek2019DrVAE, Lotfollahi2019scGen]. What is still scarce is an experimental design that measures mechanistic transfer and diagnostic transfer on the same EEG-drug twin, then localizes information loss when they dissociate.

Alzheimer's disease (AD) drug response is a natural stress test for that design. Acetylcholinesterase inhibitors (AChEIs) such as Donepezil help only a subset of patients, with reported response rates often in the 20% to 60% range [Lu2020DonepezilGenetics], and predictors spanning genetics, pharmacokinetics, comorbidity, and EEG correlates under inconsistent response definitions [Pozzi2022AChEIPredictors, Lista2022ChEIDeterminants]. EEG slowing, band-power shifts, and connectivity changes are established AD electrophysiology markers [Jeong2004EEGreview, Dauwels2010EEGAD, Cassani2018EEGADReview, Babiloni2020EEGAD, Engels2017EEGslowing, Rossini2020NeurophysAD], and cholinergic intervention is detectable in spectral EEG [Simpraga2017mAChR, Johannsson2015AChIndex, Babiloni2006DonepezilEEG]. Those facts motivate in silico simulation of Donepezil and Memantine EEG effects. They do not license treating every twin latent as an AD biomarker. Parallel EEG digital-twin work that targets diagnosis or prognosis, including Digital Alzheimer's Disease Diagnosis (DADD) models, shows that mechanism-informed EEG twins can carry clinical signal when diagnosis is the designed endpoint [Amato2025DADD, Amato2026DADDnpj, Amato2025BNM, CorralAcero2020DigitalTwin, Bruynseels2018DigitalTwins]. The open methodological problem is different: when the twin is built for pharmacodynamically constrained drug-response simulation, how should external validation separate signed mechanism from diagnostic magnitude, and how should encoding information loss be quantified?

This paper addresses that measurement gap with a pharmacodynamically constrained CVAE digital twin. The model maps 2185-dimensional resting-state EEG features and chemical Bidirectional Encoder Representations from Transformers (ChemBERTa) drug embeddings for Donepezil and Memantine into a latent space for in silico post-drug feature simulation [Chithrananda2020ChemBERTa, Winblad2007Memantine, Rive2013Memantine]. We treat band-level drug-response direction agreement with the training signature as the primary mechanistic endpoint and test it on three independent external EEG cohorts: Temple University Hospital (TUH) clinical EEG [Obeid2016TUH], an Open Science Framework (OSF) AD/healthy control set, and p-adic quantum potential EEG (P-ADIC) AD/healthy control recordings [Shor2021PADIC]. We then test latent drug-response magnitude as a separate diagnostic endpoint across four escalating cohorts, adding Chung-Ang University Hospital EEG (CAUEEG) Normal/mild cognitive impairment (MCI)/Dementia strata [Kim2023CAUEEG]. Training features use OpenNeuro accession ds004504 [Miltiadous2023OpenNeuro]. Finally, we probe CAUEEG baseline latent means against a theta/alpha spectral marker and nested classifier heads to locate any diagnosis-mechanism dissociation at encoding rather than at readout. Domain shift across hospital EEG corpora is treated as part of the test, not as an afterthought [Gulrajani2021DomainShift, Dockes2021EEGML].

The contributions are as follows:

- We show that constrained twin drug-response direction agrees with the training signature at 10/10 band-level checks on three external cohorts (TUH, OSF, P-ADIC), with effect-magnitude correlations of r = 0.908, 0.851, and 0.918, respectively.
- We show that the same twin's latent drug-response magnitude does not provide reliable external diagnosis across four escalating tests: TUH abnormal versus normal is null (p = 0.272, d = -0.13); OSF AD versus HC passes only with HC n = 12 caveats (p = 0.000278, d = 0.44); P-ADIC AD versus HC is null (p = 0.968, d = -0.07); CAUEEG Dementia versus Normal is null (p = 0.516, d = 0.00, n = 727).
- We show that CAUEEG inputs still carry clinical spectral signal (theta/alpha probe ROC-AUC = 0.675), that baseline latent means retain a weaker above-chance probe (ROC-AUC = 0.579, permutation p = 0.010), and that nested stronger heads peak at out-of-fold ROC-AUC = 0.593, so readout complexity does not close the gap to the spectral marker.
- We therefore quantify, for this constrained CVAE architecture, a concrete measurement result: mechanistic direction transfers externally while diagnostic information is partially compressed at encoding.

Section 2 describes the twin architecture, pharmacodynamic constraints, datasets, and endpoints. Section 3 reports mechanistic direction transfer, then the escalating magnitude-diagnosis investigation, then the encoding dissection. Section 4 discusses the trade-off against prior twin and AChEI-EEG literature and states architectural implications. Section 5 concludes.

The Methods section next specifies the constrained CVAE, the external cohorts, and the three endpoint families used throughout the results.


---


# 2. Methods

The Introduction framed two separate questions: whether a constrained twin preserves a constraint-consistent, model-generated drug-response signature under external EEG domain shift, and whether a twin magnitude endpoint can diagnose externally. This section defines the architecture, constraints, training cohort, external datasets, and the three endpoint families used to answer those questions. Paired post-dose EEG is not part of this study design.

## 2.1 Overview of the digital twin architecture

Building on that dual-endpoint framing, we specify the generative model used for all experiments. The twin is a pharmacodynamically constrained conditional variational autoencoder (CVAE) in the `constrained_mlp` configuration (`src/models/config.py`, `MODEL_VARIANTS['constrained_mlp']`): multilayer perceptron encoders, no graph neural network EEG encoder, no cross-attention fusion, and constrained training loss enabled.

Input EEG is a packed 2185-dimensional feature vector for 19 standard 10-20 channels: power spectral density (PSD; 19 × 20 = 380), band powers (19 × 5 = 95; delta, theta, alpha, beta, gamma), and upper-triangle coherence and phase-locking value (PLV) connectivity across five bands (1710 features). The EEG encoder maps structured tensors (PSD, band powers, coherence, PLV) to a 128-dimensional EEG latent. The drug encoder maps a 384-dimensional chemical Bidirectional Encoder Representations from Transformers (ChemBERTa) embedding to a 64-dimensional drug latent (`MODEL_CONFIG['drug_encoder']['input_dim'] = 384`). With only two study drugs (Donepezil, Memantine), ChemBERTa supplies a fixed chemical identifier for conditioning rather than a large-library chemical generalization test. Fusion concatenates EEG latent, drug latent, and a scalar disease label (AD = 1, healthy control (HC) = 0) into a 256-dimensional fused representation. The CVAE encoder outputs a Gaussian with mean \(\mu\) and log-variance over a 128-dimensional latent; the decoder reconstructs the 2185-dimensional feature vector (`MODEL_CONFIG['cvae']['latent_dim'] = 128`).

At inference, baseline conditioning uses a zero drug vector of length 384. Donepezil and Memantine use their ChemBERTa embeddings. The quantity \(\mu_{\mathrm{base}}\) denotes the CVAE latent mean under zero-drug conditioning for a subject's EEG. Drug-conditioned means \(\mu_{\mathrm{don}}\) and \(\mu_{\mathrm{mem}}\) use the corresponding drug embeddings with the same EEG encoding path. The architecture data flow (EEG tensors and ChemBERTa drug vector into encoders, fusion with disease label, CVAE \(\mu\)/log-variance, and feature reconstruction) is illustrated in Fig. 1.

Hidden widths follow `MODEL_CONFIG`: EEG encoder hidden dims [256, 512], drug encoder [256, 128], CVAE [512, 256]. Fusion attention is disabled (`use_attention: False`).

## 2.2 Pharmacodynamic constraints

Section 2.1 defined where drugs enter the latent path. This subsection states how pharmacodynamic priors shape training, as a deliberate design choice rather than a post hoc regularizer.

Signed EEG effect priors for Donepezil and Memantine are stored in `src/drugs/pharmacological_embedding.py` (`DRUG_PHARMACOLOGY[*]['eeg_effects']`). Code comments attribute Donepezil band priors to Babiloni et al. (2006) and Jeong (2004), and Memantine priors to Winblad et al. (2007) and Rive et al. (2013) [Babiloni2006DonepezilEEG, Jeong2004EEGreview, Winblad2007Memantine, Rive2013Memantine]. The numeric priors used in the loss are:

- Donepezil: delta -0.15, theta -0.20, alpha +0.25, beta +0.10, connectivity +0.20  
- Memantine: delta -0.10, theta -0.15, alpha +0.10, beta +0.05, connectivity +0.15  

`get_eeg_effect_prior(drug_name)` returns these dictionaries for the loss. Band direction constraints operate on reconstructed versus baseline band-power blocks in the 2185-D vector (delta [380:399], theta [399:418], alpha [418:437], beta [437:456]) via `compute_band_direction_constraint` in `src/models/losses.py`. For each band, the mean reconstructed change relative to baseline is compared to the prior sign: decreases are penalized when an increase is expected, and increases are penalized when a decrease is expected, scaled by the absolute prior weight. A separate connectivity constraint on coherence features [475:1330] penalizes mean coherence decreases under drug conditions (`compute_connectivity_constraint`). Baseline (no-drug) batches incur zero constraint terms.

The total constrained objective is
\[
L = L_{\mathrm{MSE}} + \beta L_{\mathrm{KL}} + \alpha_c L_{\mathrm{band}} + \alpha_c L_{\mathrm{conn}},
\]
with \(\beta = 0.01\) (`TRAINING_CONFIG['beta_kl']`). Constraint weight \(\alpha_c = 0\) for the first `warmup_epochs = 5` epochs, then ramps as `constraint_weight * min(1, (epoch - warmup_epochs) / 5)` with `constraint_weight = 0.1` (checkpoint metadata and constrained training defaults in `run_constrained_training.py`). Constraints therefore bias reconstructed feature changes toward literature-signed pharmacology while reconstruction and KL terms remain active. This design prioritizes mechanistic consistency of simulated drug effects; any effect on diagnostic content of \(\mu\) is treated as an empirical question for Results, not assumed here.

## 2.3 Training data and procedure

With architecture and constraints fixed, we describe the derivation cohort and optimization used for the frozen twin evaluated externally.

Training features come from resting-state EEG for 66 subjects (AD n = 37, HC n = 29) stored as `data/eeg_features/{AD,HC}_*.npy` (PSD shapes `(37, 19, 20)` and `(29, 19, 20)`). The derivation cohort is OpenNeuro accession ds004504 (Miltiadous et al. AD/HC resting EEG). Feature rows follow every EEGLAB `.set` file discovered under `data/raw_eeg/{AD,HC}/` by `EEGLoader.find_eeg_files`: HC subjects `sub-037` to `sub-065` (29 files), and AD subjects `sub-001` to `sub-036` plus both `sub-001_task-eyesclosed_eeg.set` and `sub-001_task-eyesclosed_eeg_cleaned.set` (37 AD files; 36 unique BIDS subject folders). Preprocessing for training used the shared feature pipeline: band-pass 0.5-40 Hz, notch 50 Hz, ICA on, 4 s epochs at 90% overlap, per-subject z-score, then spectral and connectivity features with time-averaged epochs before Welch (`n_per_seg = 256` at 500 Hz yields 20 PSD bins), as summarized in the domain-shift diagnosis note under `models/validation/`.

Constrained training used subject-level 5-fold cross-validation (`get_subject_level_kfold`, `n_splits = 5`, `random_seed = 42` from `DATA_CONFIG`). Within each fold, the non-test subjects were split into train and validation with `test_size = 0.1` stratified by label (`run_constrained_training.py`). Optimization used Adam with learning rate \(1 \times 10^{-4}\), weight decay \(1 \times 10^{-5}\), batch size 16, and 100 epochs (`TRAINING_CONFIG`) on an NVIDIA GeForce RTX 5060 Ti (16 GB; `TRAINING_CONFIG['device']` resolved to `cuda`; recorded in `models/logs/training_history.json`). The evaluation checkpoint is `models/checkpoints_constrained/checkpoint_constrained.pt` (recorded `epoch: 99`, `constraint_weight: 0.1`, `warmup_epochs: 5`), a fold-0 alias of the 5-fold constrained CV. External twin simulations in this paper use that checkpoint with `num_samples = 5` Monte Carlo draws per condition (`simulate_post_drug_eeg`; default elsewhere is 10).

## 2.4 External validation datasets

Training established the twin. External datasets were introduced in sequence so each cohort answered a specific doubt left by the previous magnitude or label setting, matching the locked escalation in `paper/STORYLINE_LOCKED.md`. Cohort size, labels, sampling rate, montage, and study role are summarized in Table 1.

**Temple University Hospital (TUH) clinical EEG.** We used 200 successfully processed EDFs (`complete_validation_report_v3.json`, TUH processing block: 200/200). Labels for magnitude tests were header/path-derived abnormal versus normal (n = 131 abnormal, n = 69 normal in the discrimination block), not AD-specific diagnoses. Channels were mapped to 19 legacy 10-20 names, recordings cropped (`CROP_SECONDS = 320` in `process_tuh_eeg.py`), resampled to 500 Hz, notch 60 Hz (US mains), ICA off for tractability, with epoch-count spectral calibration to `TARGET_N_EPOCHS_FOR_PSD = 864`. TUH was selected first to test whether latent drug-response magnitude discriminated on independent hospital EEG after leaving the training resting-state distribution.

**Open Science Framework (OSF) Eyes_closed AD/HC.** After the TUH magnitude null under proxy labels, we used folder-labeled AD (n = 80) and Healthy/HC (n = 12) features already imported as `data/external_osf` / `data/ad_labeled_external`. Native rate was 256 Hz with 19-channel 10-20 layout, resampled to 500 Hz in the feature import; stored processed shape implied one 4 s epoch per subject before calibration to the same 864-epoch spectral reference. OSF was selected to ask whether true AD versus HC labels (still research partitions, not NIA-AA gold standard in local metadata) would restore magnitude discrimination that abnormal/normal proxies lacked.

**P-ADIC Dryad AD/HC.** After OSF's positive but HC-underpowered result, we used Dryad release recordings resolved to AD n = 49 and HC n = 96 at recording level (`complete_validation_report_v3.json`, P-ADIC discrimination block), DOI 10.5061/dryad.8gtht76pw [Shor2021PADIC]. Arrays were remapped from an assumed Nihon-order montage into training channel order, cropped to 320 s, resampled to 500 Hz when needed, notch 50 Hz, and calibrated to 864 epochs. P-ADIC was selected to test whether a larger NIA-AA-oriented AD/HC resource confirmed OSF or exposed fragility of the HC n = 12 result.

**CAUEEG (Chung-Ang University Hospital EEG).** After P-ADIC's magnitude null, we used the dementia-no-overlap benchmark pool (train+validation+test) with Normal n = 436, MCI n = 395, Dementia n = 291 (1122/1122 features OK; CAUEEG `n_by_class` in the validation report). EDFs are 19-channel average-referenced 10-20 at 200 Hz plus non-EEG channels dropped by mapping; processing used `load_tuh_edf` with notch 60 Hz (South Korea), crop 320 s, resample 500 Hz, and spectral calibration to 864 epochs (`src/tuh/process_caueeg_eeg.py`). Band-power scale versus training was ratio 1.104 (`scale_check.bandpower_ratio_caueeg_over_train`). CAUEEG was selected to test magnitude discrimination on large clinical Normal/MCI/Dementia strata with calibrated spectral scale and good PCA overlap, and to host the encoding probes.

Direction agreement for mechanism tests used TUH, OSF, and P-ADIC cohort-mean drug-minus-baseline feature effects against the training-cohort mean effects (CAUEEG was not required for the locked 10/10 direction claim).

## 2.5 Evaluation endpoints

Datasets alone do not define success. We therefore fix three endpoint families used uniformly in Results. **No paired pre/post-dose EEG** is available in the external cohorts; the direction endpoint measures agreement with the **training constrained signature**, not empirical pharmacodynamic change after drug administration.

**Primary hierarchy.** P1: constraint-consistent direction on TUH, OSF, and P-ADIC (10-block agreement and effect-vector correlation). P2: CAUEEG Dementia versus Normal magnitude on \(\bar{s}\). P3: CAUEEG latent probe ROC-AUC on \(\mu_{\mathrm{base}}\) versus a theta/alpha spectral reference. Remaining contrasts (TUH proxy labels, OSF magnitude with HC \(n=12\), P-ADIC magnitude, nested heads) are escalation or exploratory follow-ups.

**Drug-response direction agreement.** For each external cohort with saved per-subject simulations, we form the mean Donepezil-minus-baseline and Memantine-minus-baseline vectors in 2185-D feature space and compare each to the corresponding mean effect from the training-cohort constrained simulations (`analyze_cross_dataset` and cohort-equivalent helpers). Agreement is counted on five blocks: delta [380:399], theta [399:418], alpha [418:437], beta [437:456], and connectivity [475:1330] (`BAND_SLICES` in `src/tuh/tuh_validation.py`). A block agrees if \(\mathrm{sign}(\mathrm{mean}(a_{\mathrm{block}})) = \mathrm{sign}(\mathrm{mean}(b_{\mathrm{block}}))\), or both absolute means are below \(10^{-8}\). Donepezil and Memantine each contribute five checks, so perfect agreement is 10/10. We also report Pearson correlation between the full mean effect vectors (reported as `effect_magnitude_correlation` in validation JSON), with Fisher \(z\) 95% confidence intervals using feature length \(n = 2185\).

**Drug-response magnitude.** For each subject, after encoding, we compute
\[
s_{\mathrm{don}} = \|\mu_{\mathrm{don}} - \mu_{\mathrm{base}}\|_2, \quad
s_{\mathrm{mem}} = \|\mu_{\mathrm{mem}} - \mu_{\mathrm{base}}\|_2, \quad
\bar{s} = \tfrac{1}{2}(s_{\mathrm{don}} + s_{\mathrm{mem}}),
\]
as implemented in the external twin runners (`mean_drug_response`). Group differences on \(\bar{s}\) are the diagnostic magnitude tests (abnormal vs normal; AD vs HC; Dementia vs Normal).

**Encoding probes (CAUEEG only).** Baseline latents were extracted with drug vector all zeros and disease label fixed to 0 for every subject to avoid leaking diagnosis into fusion (`run_latent_probe_caueeg.py`, `PROBE_DISEASE_LABEL = 0.0`). Primary analysis used Dementia versus Normal only (n = 727; MCI excluded). A logistic regression probe used 5-fold stratified cross-validation (`StratifiedKFold`, `n_splits = 5`, `shuffle = True`, `random_state = 42`), `StandardScaler` fit on each training fold only, `LogisticRegression(class_weight='balanced', max_iter = 2000)`. A 200-label-permutation null repeated the same 5-fold probe (`N_PERM = 200`). The spectral reference used the same folds on scalar theta/alpha ratio from band powers [380:475]. Stronger heads (`run_latent_classifier_head_caueeg.py`) used nested CV: outer 5-fold stratified, inner 3-fold `GridSearchCV` on ROC-AUC for tuned L2 logistic regression, MLP, and `HistGradientBoostingClassifier`, plus a \(\mu\) concatenated with log-variance control for the best head. The best head's out-of-fold ROC-AUC was tested with 200 permutations of the same nested protocol.

**Ablations.** Matched unconstrained fold-0 checkpoint (`models/checkpoints_unconstrained/checkpoint_unconstrained.pt`, same selection rule as constrained fold_0), a prior-only signed effect baseline (`prior_only_direction.json`), and a constraint-weight (\(\alpha_c\)) fold-0 sweep test whether direction preservation and latent diagnostic AUC move with constraint strength.

## 2.6 Statistics

Endpoints in Section 2.5 map onto the following tests, with parameters taken from the run scripts.

Two-group magnitude comparisons used two-sided Mann-Whitney U tests and Cohen's d with sample variance (`ddof = 1`) as in `_cohens_d` / discriminate helpers. CAUEEG also reported a three-group Kruskal-Wallis test on Normal, MCI, and Dementia magnitude scores (`scipy.stats.kruskal`). Encoding probes reported ROC-AUC, balanced accuracy, and F1 per fold and as means with sample standard deviation across folds (`ddof = 1`), plus out-of-fold aggregates. Permutation p-values used
\[
p = \frac{1 + \#\{\mathrm{null\ AUC} \ge \mathrm{true\ AUC}\}}{N_{\mathrm{perm}} + 1}
\]
with \(N_{\mathrm{perm}} = 200\). Nested head selection optimized inner-fold ROC-AUC only; outer test folds were never used for hyperparameter choice.

No multiplicity correction (Bonferroni, FDR, or similar) was applied across the escalating external contrasts or across probe metrics. Each primary contrast was treated as a pre-specified question in the escalation sequence (proxy hospital EEG; AD/HC small HC; larger AD/HC; large clinical dementia spectrum; then encoding localization), and secondary contrasts (for example MCI pairs, disease-label ablation) are reported descriptively. External twin and probe scripts report unadjusted Mann-Whitney, Kruskal-Wallis, and permutation p-values with no `multipletests` or related correction step.

Hardware for feature extraction, constrained training, and the 1122-subject CAUEEG twin pass used the project `eeg_twin` Python 3.10 environment on an NVIDIA GeForce RTX 5060 Ti (16 GB).

These definitions and datasets are used without change in the Results that follow, beginning with constraint-consistent direction agreement on the three external cohorts that support the 10/10 signature claim.


---


# 3. Results

## 3.1 Matched unconstrained and prior-only baselines

Section 2.5 defined constraint-consistent direction agreement against the training constrained signature and reserved matched unconstrained and prior-only checks for interpreting that endpoint. We report those controls first, then the constrained external replications.

**Matched unconstrained twin.** The unconstrained checkpoint uses the same fold-0 selection rule as the constrained external twin (`checkpoints_5fold/fold_0_best.pt` aliased as `checkpoint_unconstrained.pt`; see Methods). On the same direction endpoint and constrained training signature, unconstrained agreement was 3/10 on TUH (effect-vector \(r = -0.309\)), 4/10 on OSF (\(r = -0.331\)), and 4/10 on P-ADIC (\(r = -0.327\); `unconstrained_external_battery.json`). Constrained agreement on the same three cohorts is 10/10 (Section 3.2). Magnitude contrasts for the unconstrained twin remained null or non-diagnostic on TUH abnormal versus normal (p = 0.124, d = -0.04), OSF AD versus HC (p = 0.124, d = -0.16), P-ADIC AD versus HC (p = 0.791, d = 0.03), and CAUEEG Dementia versus Normal (p = 0.088, d = -0.07). On CAUEEG Dementia versus Normal (n = 727), the unconstrained \(\mu_{\mathrm{base}}\) logistic probe reached mean ROC-AUC = 0.539, below the constrained latent probe (0.579) and below the theta/alpha spectral reference on the same subjects (mean ROC-AUC = 0.675). Table 2 summarizes constrained versus unconstrained direction on TUH, OSF, and P-ADIC.

**Prior-only baseline.** Applying literature signed band and connectivity deltas (`get_eeg_effect_prior`) as a constant 2185-D effect vector, without a CVAE, yielded 10/10 direction agreement against the constrained training signature, with mean effect-vector correlation \(r = 0.636\) (Donepezil \(r = 0.572\); Memantine \(r = 0.700\); `prior_only_direction.json`). Perfect sign agreement is therefore achievable from the prior alone. The constrained external correlations (\(r = 0.851\) to \(0.918\); Section 3.2) exceed that prior-only magnitude match, so the twin recovers additional signed structure beyond constant prior blocks.

These controls establish two facts used below. First, matched unconstrained training does not preserve the constrained signature under external domain shift. Second, prior-only signs are necessary but not sufficient to explain the high constrained external \(r\) values. Whether raising \(\alpha_c\) locks direction and how latent AUC moves is deferred to Section 3.5.

## 3.2 Constrained twin: drug-response direction agreement

Having established that unconstrained training fails the direction endpoint, we ask whether the constrained twin reproduces the training cohort's signed band and connectivity effects on independent EEG. Perfect agreement is 10/10 (five blocks × Donepezil and Memantine). Effect-vector correlation \(r\) summarizes the match between cohort-mean drug-minus-baseline feature vectors. This endpoint measures agreement with the **training constrained signature**; paired post-dose EEG is not available in these cohorts.

Across three external cohorts, constrained direction agreement was 10/10 in every case, with effect-vector correlations \(r = 0.851\) to \(r = 0.918\). Per-dataset values are summarized in Table 3 and illustrated in Fig. 2. Fisher \(z\) 95% confidence intervals for \(r\) (feature length \(n = 2185\)) are TUH 0.908 (95% CI 0.900 to 0.915), OSF 0.851 (95% CI 0.839 to 0.862), and P-ADIC 0.918 (95% CI 0.911 to 0.924) (`direction_r_fisher_ci.json`).

**TUH (first confirmation).** On Temple University Hospital clinical EEG (n = 200), Donepezil and Memantine each agreed on all five blocks (5/5 and 5/5; total 10/10), with overall effect-vector correlation \(r = 0.908\) (`layer5.cross_dataset`). A falsifying outcome would have been any band or connectivity sign flip relative to the training signature, or a near-zero correlation. Neither occurred. Relative to the matched unconstrained twin (3/10 on the same endpoint), this is the first external confirmation that the constrained mechanism survives hospital EEG outside the training resting-state distribution.

**OSF (second confirmation).** On the OSF Eyes_closed AD/HC set (n = 92), direction agreement was again 10/10 (Donepezil 5/5, Memantine 5/5), with \(r = 0.851\) (`layer5b_ad_labeled_external.cross_dataset`). This cohort changes acquisition and label structure relative to TUH. Unconstrained agreement on the same OSF features was 4/10. Agreement for the constrained twin remained perfect.

**P-ADIC (third confirmation).** On P-ADIC Dryad AD/HC recordings (n = 145), direction agreement was 10/10 (Donepezil 5/5, Memantine 5/5), with \(r = 0.918\) (`layer5d_padic_external.cross_dataset`). Unconstrained agreement was 4/10. Constrained agreement remained perfect after montage remapping and spectral calibration.

Fig. 2 shows the three constrained effect-vector correlations with the corresponding 10/10 annotations. Table 3 lists n, agreement fractions, \(r\), and replication status for each cohort. Table 2 places the matched unconstrained failures beside those constrained successes.

These results establish that the pharmacodynamically constrained twin preserves constraint-consistent drug-response *direction* across three independent external EEG resources. Band-level signed pharmacology, not merely in-sample reconstruction, generalizes under the Section 2.5 endpoint when constraints are present. Whether this mechanistic fidelity translates into diagnostic utility of drug-response *magnitude* is a separate question, addressed next.


---


# 3.3 Diagnostic endpoint: drug-response magnitude

Section 3.2 showed that constraint-consistent drug-response *direction* agrees with the training signature at 10/10 on three cohorts, while Section 3.1 showed that a matched unconstrained twin fails that endpoint. We now ask whether drug-response *magnitude* \(\bar{s}\) carries external diagnostic signal under the Section 2.5 definition. The investigation is sequential: each cohort answers a doubt left by the previous null or caveated result. Primary contrasts and secondary checks are summarized in Table 4. Effect sizes with approximate 95% confidence intervals are illustrated in Fig. 3.

**TUH: first external magnitude test (proxy labels).** On TUH clinical EEG we compared abnormal (n = 131) versus normal (n = 69) on \(\bar{s}\). The contrast was null (Mann-Whitney p = 0.272, Cohen's d = -0.13; status FAIL; `layer5.abnormal_vs_normal`). Labels are header/path-derived abnormal versus normal, not AD-specific diagnoses, and the processed cohort carried no AD-drug annotations usable as response outcomes. A credible confound for this null is therefore weak proxy labeling rather than a definitive failure of the twin under true AD/HC strata. The open question is whether AD versus HC labels restore magnitude discrimination.

**OSF: AD/HC labels with a thin control arm.** On OSF Eyes_closed AD (n = 80) versus HC (n = 12), \(\bar{s}\) was larger in AD (p = 0.000278, d = 0.44; discrimination status PASS; exploratory PASS_WITH_CAVEATS in the validation report). Separation persisted when disease conditioning was fixed for all subjects (disease = 0: p < 0.001, d = 0.64; disease = 1: p < 0.001, d = 0.34; OSF sensitivity block in `data/ad_labeled_external/validation/`), so the signal is not an artifact of feeding diagnosis into fusion alone. The result is not a clean external PASS: HC n = 12 limits power, and the absolute mean gap is small. The open question is whether a larger, clinically structured AD/HC resource confirms OSF or exposes fragility of the HC arm.

**P-ADIC: larger AD/HC, null magnitude.** On P-ADIC recordings (AD n = 49, HC n = 96), \(\bar{s}\) did not separate groups (p = 0.968, d = -0.07; status FAIL; P-ADIC discrimination in `complete_validation_report_v3.json`). Disease-label ablation remained null (disease = 0: p = 0.527, d = -0.16; disease = 1: p = 0.702, d = -0.14; both FAIL). Age imbalance (AD mean age about 70 years versus HC about 52 years in the release metadata) is a listed confound, but age would typically inflate rather than erase AD/HC differences if it drove \(\bar{s}\); here differences are absent. The open question is whether large-N hospital Normal/MCI/Dementia strata with calibrated spectral scale still show a magnitude null.

**CAUEEG: large clinical strata and a domain check.** CAUEEG was selected next because it supplies large clinical Normal (n = 436), MCI (n = 395), and Dementia (n = 291) pools (1122 features OK), with band-power scale near training (ratio 1.104) and good PCA overlap with the training cohort (overlap fraction within the training 95% radius = 0.611; Fig. 4). Primary Dementia versus Normal on \(\bar{s}\) was null (n = 291 / 436; p = 0.516, d = -0.00; status FAIL; `primary_dementia_vs_normal`). Secondary twin contrasts were likewise null: Dementia versus MCI (p = 0.877, d = 0.07), MCI versus Normal (p = 0.363, d = -0.07), AD-tagged Dementia versus Normal (n = 214 / 436; p = 0.624, d = -0.01), and Kruskal-Wallis across three groups (H = 0.90, p = 0.637). Disease-label ablation on Dementia versus Normal remained FAIL (disease = 0: p = 0.411, d = -0.04; disease = 1: p = 0.776, d = 0.00). Critically, the same CAUEEG features do carry clinical signal outside the twin endpoint: theta/alpha ratio Dementia versus Normal was strongly separated (p < 0.001, d = 0.47; `feature_space_discrimination.theta_alpha_ratio`). The magnitude null is therefore not explained by absent clinical EEG structure in the inputs, nor by a failed scale/PCA check.

Fig. 3 places the four primary magnitude contrasts on a common Cohen's d axis. Table 4 records every listed contrast with exact n, p, d, and status strings from the validation runs.

Section 3.2 established constraint-consistent direction agreement. Section 3.3 shows that the twin's magnitude endpoint does not diagnose reliably across an escalating sequence of external tests, despite PASS_WITH_CAVEATS on a small-HC OSF partition and despite clear spectral slowing in CAUEEG features. Clinical signal is present in raw features, but the diagnostic magnitude endpoint does not use it. The next section localizes residual clinical information at encoding versus readout.


---


# 3.4 Encoding analysis: where diagnostic information is lost

Section 3.3 left an explicit puzzle: CAUEEG features separate Dementia from Normal on theta/alpha slowing, yet the twin's drug-response magnitude endpoint does not. We therefore ask whether clinical signal survives CVAE encoding at all, and if so, whether a more flexible readout can recover the feature-level ceiling.

Baseline latents \(\mu_{\mathrm{base}}\) were extracted with drug vector zero and disease label fixed to 0 for every subject (Dementia versus Normal only; n = 727), so the probe cannot read diagnosis from the fusion disease bit. A linear logistic probe on \(\mu_{\mathrm{base}}\) reached mean cross-validated ROC-AUC = 0.579 (permutation p = 0.010 under 200 label shuffles of the same 5-fold protocol; `latent_probe_dementia_vs_normal`, `permutation_test_latent`). The probe therefore beats the permutation null: measurable diagnostic information survives encoding. It does not match the input reference. The same folds on scalar theta/alpha ratio reached mean ROC-AUC = 0.675 (`feature_probe_theta_alpha_dementia_vs_normal`). The latent probe sits about 0.096 below that spectral reference.

If the gap were only a weak linear readout, nested stronger heads on frozen \(\mu_{\mathrm{base}}\) should close it. They did not. Out-of-fold ROC-AUC values were 0.593 for tuned L2 logistic regression (best head; permutation p = 0.005 under 200 nested-CV shuffles), 0.543 for MLP, 0.556 for HistGradientBoosting, and 0.587 for a \(\mu\)+logvar control (`layer5e_caueeg_classifier_head_results.json`; verdict tag: ceiling reached). The best head remains near the original linear probe and well below theta/alpha. Nonlinear and more flexible readouts did not close the gap. The information loss is therefore localized to encoding under this constrained twin, not to readout complexity alone.

Fig. 5 places the spectral reference, the original latent probe, and all nested heads on a common ROC-AUC axis against the latent-probe permutation null band (5th to 95th percentiles 0.449 to 0.559) and chance (AUC = 0.5). Table 5 reports the corresponding out-of-fold (OOF) AUCs, balanced accuracies, permutation p-values where computed, and signed gaps versus the theta/alpha OOF ceiling.

Section 3.1 already showed that a matched unconstrained encoder does not raise CAUEEG latent AUC above the constrained probe (unconstrained mean AUC = 0.539 versus constrained 0.579). Encoding loss relative to the spectral reference is therefore not explained as a simple unconstrained-versus-constrained compression on this comparison. The next section asks whether sweeping constraint weight \(\alpha_c\) moves direction fidelity and latent AUC together.


---


# 3.5 Constraint-strength sweep

Sections 3.1 to 3.4 established that constraints are required for external signature preservation, that magnitude fails diagnostically, and that \(\mu_{\mathrm{base}}\) retains partial clinical signal below a spectral reference. We now vary the constraint weight \(\alpha_c\) on matched fold-0 twins (same selection rule as the paper checkpoint) and score TUH and P-ADIC direction against the constrained training signature plus CAUEEG Dementia versus Normal latent-probe AUC (`constraint_strength_sweep.json`). Results are illustrated in Fig. 6.

At \(\alpha_c = 0\) (seeded unconstrained fold-0), direction agreement was 3/10 on TUH (effect-vector \(r = -0.350\)) and 4/10 on P-ADIC (\(r = -0.364\)), with CAUEEG latent mean AUC = 0.539. At \(\alpha_c = 0.01\), agreement rose to 8/10 on both TUH and P-ADIC, but effect-vector correlations remained low (TUH \(r = 0.096\)). At \(\alpha_c = 0.05\), agreement reached 10/10 on both cohorts (TUH \(r = 0.299\); latent AUC = 0.527). At the paper weight \(\alpha_c = 0.1\), agreement remained 10/10 and effect-vector correlation peaked (TUH \(r = 0.897\); P-ADIC direction 10/10; latent AUC = 0.579). At \(\alpha_c = 0.2\) and \(0.5\), direction signs stayed at 10/10 while effect-vector \(r\) fell (TUH \(r = 0.220\) and \(0.412\)) and latent AUC was 0.514 and 0.578, respectively.

Fig. 6 therefore shows a clear \(\alpha_c\) dependence for signed direction agreement and a non-monotone effect-vector correlation that peaks at the paper weight. Latent diagnostic AUC does not fall as direction rises: the curve is non-monotone and does not support a simple direction-versus-diagnosis trade-off across \(\alpha_c\) alone. The constrained twin's external behavior is better summarized as constraint-dependent signature preservation with a preferred operating weight near \(\alpha_c = 0.1\) for effect-vector correlation, plus a separate encoding gap relative to spectral slowing that is not closed by removing constraints.

The Discussion situates these findings for simulation use, states what was and was not validated without post-dose EEG, and lists failure modes specific to this architecture and label structure.


---


# 4. Discussion

Section 3.3 established that CAUEEG clinical signal survives only partially in \(\mu_{\mathrm{base}}\) (latent probe ROC-AUC = 0.579 versus theta/alpha 0.675) and that nested heads do not close that gap (best out-of-fold ROC-AUC = 0.593). Section 3.5 showed that signed direction agreement requires nonzero \(\alpha_c\) and that effect-vector correlation peaks at \(\alpha_c = 0.1\), without a monotone drop in latent AUC as direction improves. Together with Section 3.1 (matched unconstrained direction failure; unconstrained latent AUC = 0.539), the Results spine is: constraint-consistent direction is preserved under domain shift when constraints are present; diagnostic magnitude does not; residual clinical signal at encoding sits below a spectral reference; removing constraints does not recover that spectral ceiling on this matched comparison.

Mechanistic transfer should be read against recent EEG digital twin work that targets diagnosis and prognosis rather than chemistry-conditioned response simulation. Digital Alzheimer's Disease Diagnosis (DADD) twins invert EEG into neurodegeneration-linked parameters and report strong CSF and conversion prediction relative to standard EEG metrics [Amato2025DADD, Amato2026DADDnpj]. Related personalized cortical models likewise estimate synaptic and connectivity degeneration severity from EEG for early-stage stratification [Amato2025BNM]. Those lines of work demonstrate that mechanism-informed EEG twins can carry clinical signal when the endpoint is neurodegeneration or prognosis [CorralAcero2020DigitalTwin, Bruynseels2018DigitalTwins]. Parallel EEG pharmacology and AD electrophysiology work shows that cholinergic intervention, slowing, and network disruption are detectable in spectral and connectivity EEG [Simpraga2017mAChR, Johannsson2015AChIndex, Cassani2018EEGADReview, Babiloni2020EEGAD, Meghdadi2021EEGAD, Horvath2018EEGAD, Vecchio2017ConnectivityAD, LopezSanz2019NetworkAD]. Our twin sits in a different niche: ChemBERTa-conditioned simulation of Donepezil and Memantine feature effects under explicit band and connectivity priors. The 10/10 external direction replications indicate that this pharmacodynamic constraint class can preserve signed response structure across acquisition shifts even when diagnosis is not the training objective [Obeid2016TUH, Kim2023CAUEEG, Dockes2021EEGML].

The diagnostic null and encoding gap parallel a known dissociation in generative modeling rather than standing alone. Variational autoencoders and related generative models already separate reconstruction, disentanglement, and predictive utility in other domains [Kingma2014VAE, Higgins2017betaVAE, Lotfollahi2019scGen]. In transcriptomic drug-response VAEs, reconstruction quality and classification performance already separate: better reconstruction does not automatically improve response prediction, and joint modeling of perturbation and outcome can matter more than reconstruction alone [Rampasek2019DrVAE]. What is specific here is the endpoint pair and architecture class. We separately score (i) external transfer of a literature-constrained drug-effect *direction* and (ii) external diagnosis from latent drug-response *magnitude*, then quantify residual diagnostic content in \(\mu_{\mathrm{base}}\) against a theta/alpha ceiling on the same subjects. We are not aware of a prior report that combines those three checks for a pharmacodynamically constrained EEG-drug CVAE. That is a contribution of measurement and experimental design, not a claim that no generative model has ever shown mechanism-classification tension [Gulrajani2021DomainShift, Stam2005NonlinearEEG].

Practically, a researcher building on this twin should treat it as a treatment-response simulation and hypothesis-generation tool for signed EEG pharmacology, not as a standalone external diagnostic classifier. If diagnosis is required, an auxiliary readout should be calibrated against an explicit feature-level ceiling such as theta/alpha (here ROC-AUC = 0.675 on CAUEEG Dementia versus Normal), and performance near 0.58 to 0.59 on frozen \(\mu_{\mathrm{base}}\) should be interpreted as partial survival, not as recovery of input clinical information. Magnitude \(\bar{s}\) should not be marketed as an AD biomarker from these external results.

Limitations are specific to the failure modes observed here. Encoding information loss is localized by probes and nested heads relative to a spectral reference, but the architectural mechanism (which latent dimensions or constraint terms discard slowing-related variance) is not fully mapped. CAUEEG "Dementia" is a clinical spectrum spanning AD and non-AD subtypes, so label heterogeneity remains. CAUEEG has no same-subject drug-outcome ground truth, so direction agreement is evaluated against the training constrained signature rather than against measured post-dose EEG change. Matched unconstrained and \(\alpha_c\) experiments test how direction and latent AUC move with constraint strength; B1 alone does not establish that constraints compress diagnosis relative to unconstrained training.

A concrete next experiment follows directly. Train matched twins that (a) keep the present pharmacodynamic constraints, (b) remove or anneal them after warmup, and (c) add an auxiliary diagnostic head on \(\mu_{\mathrm{base}}\) (or on a parallel unconstrained encoder branch) with an explicit penalty for falling below a held-out theta/alpha ceiling, then freeze each model and repeat the three-cohort direction protocol plus CAUEEG Dementia versus Normal magnitude, latent probe, and nested-head battery. The decisive readout is whether any alternative recovers latent probe AUC toward 0.675 without dropping direction agreement below 10/10 on TUH, OSF, and P-ADIC. That experiment would test whether the trade-off can be shifted by architecture, rather than merely restating that larger samples are desirable.

The Conclusion restates what this study establishes as fact for this twin class and what remains out of scope.


---


# 5. Conclusion

Given the Discussion's resolution that this twin preserves constraint-consistent drug-response direction under domain shift while failing as an external diagnostic magnitude tool, this study establishes four facts for a pharmacodynamically constrained EEG-drug CVAE twin. First, constrained drug-response direction agrees with the training signature at 10/10 on three independent external cohorts (TUH, OSF, and P-ADIC), with effect-vector correlations \(r = 0.908\), \(0.851\), and \(0.918\); a matched unconstrained fold-0 twin fails the same endpoint (3/10 to 4/10). Second, a prior-only signed effect also reaches 10/10 on signs but with lower mean effect-vector correlation (\(r = 0.636\)), so external constrained \(r\) values are not explained by constant priors alone. Third, drug-response magnitude does not diagnose reliably on external EEG: TUH abnormal versus normal was null (p = 0.272, d = -0.13); OSF AD versus HC was PASS_WITH_CAVEATS only (p = 0.000278, d = 0.44, HC n = 12); P-ADIC AD versus HC was null (p = 0.968, d = -0.07); CAUEEG Dementia versus Normal was null (p = 0.516, d = -0.00, n = 727). Fourth, residual clinical signal at encoding sits below a spectral reference: CAUEEG theta/alpha features reach ROC-AUC = 0.675, baseline \(\mu_{\mathrm{base}}\) retains ROC-AUC = 0.579 (permutation p = 0.010), nested heads peak at out-of-fold ROC-AUC = 0.593, and unconstrained latent AUC (0.539) does not exceed the constrained probe. Sweeping \(\alpha_c\) locks direction signs by 0.05 and peaks effect-vector correlation at 0.1 without a monotone latent-AUC trade-off. The twin therefore works as an externally consistent simulator of constrained drug-response direction, not as a standalone external diagnostic tool, and not as an empirically validated post-dose pharmacodynamic predictor in the absence of paired drug EEG.

---

## CRediT authorship contribution statement

Author contributions will be completed when the author list is finalized. No corresponding author is designated in this draft.

---

## Ethics statement

This work is a secondary analysis of de-identified EEG datasets. No new human participants were recruited and no clinical interventions were performed. Training resting-state EEG features derive from the public OpenNeuro accession ds004504 (Miltiadous et al.). External cohorts were obtained under each provider's academic data-use terms: the Temple University Hospital (TUH) EEG Corpus data use agreement; a public Open Science Framework (OSF) Eyes_closed AD/HC partition; the Dryad P-ADIC release (DOI 10.5061/dryad.8gtht76pw); and academic, non-commercial access to CAUEEG granted by the CAUEEG team (access contact: Min-jae Kim), with no redistribution of raw CAUEEG or TUH recordings. Analyses were conducted for academic research. The study does not claim prospective clinical decision support.

---

## Funding

This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

---

## Data availability

Training resting-state EEG features used in this project are from OpenNeuro accession ds004504 (Miltiadous et al. AD/HC resting EEG), processed to the feature arrays described in Methods. Temple University Hospital (TUH) clinical EEG is available to researchers under the TUH corpus data use agreement and cannot be redistributed by the authors. The OSF Eyes_closed AD/HC features used for the AD/HC external magnitude and direction checks were imported from a public OSF-hosted partition already present in the project archive. P-ADIC AD and control matrices are available from Dryad (DOI 10.5061/dryad.8gtht76pw) under the release terms of Shor et al. CAUEEG raw recordings were obtained by academic application to the CAUEEG team (access granted by Min-jae Kim; Kim et al., NeuroImage 2023) for non-commercial research use and must not be redistributed. Derived project artifacts that can be shared subject to those upstream terms include processed feature arrays generated in this repository (where redistribution is allowed), validation JSON summaries under `models/validation/`, and analysis code under `src/`. Raw CAUEEG and TUH EDFs will not be redistributed by the authors.

---

## Code availability

Analysis and figure-generation code for this manuscript live in the project repository under `src/` (training: `run_constrained_training.py`, `run_5fold_cv.py`; external batteries: `src/validation/run_unconstrained_external_battery.py`, `src/validation/run_prior_only_direction.py`, `src/validation/run_alpha_c_sweep.py`; figures/tables: `src/figures/`, `src/tables/`; Overleaf pack: `src/paper/pack_overleaf_zip.py`). The Python environment is the project virtualenv `eeg_twin` (Python 3.10); install from `requirements.txt` into that environment. Key frozen checkpoints are `models/checkpoints_constrained/checkpoint_constrained.pt` (constrained fold_0 alias) and `models/checkpoints_unconstrained/checkpoint_unconstrained.pt` (matched unconstrained fold_0). To regenerate the reported validation artifacts after activating `eeg_twin`:

```text
.\eeg_twin\Scripts\python.exe -m src.validation.run_prior_only_direction
.\eeg_twin\Scripts\python.exe -m src.validation.compute_direction_r_cis
.\eeg_twin\Scripts\python.exe -m src.validation.run_unconstrained_external_battery
.\eeg_twin\Scripts\python.exe -m src.validation.run_alpha_c_sweep --eval-only
.\eeg_twin\Scripts\python.exe src\figures\make_fig6_constraint_strength.py
.\eeg_twin\Scripts\python.exe src\paper\pack_overleaf_zip.py
```

Code is not currently in a public repository. A public mirror (URL and commit hash) will be linked at acceptance; until then, the packaged Overleaf zip `paper/overleaf_cbm_elsarticle.zip` and the local repository paths above are the reproducibility sources for the reported JSON and figures. See also `paper/CODE_AVAILABILITY.md`.

---

## Declaration of competing interests

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## Acknowledgments

We thank the Temple University Hospital (TUH) EEG Corpus team for access to clinical EEG under the TUH data use agreement; Min-jae Kim and the CAUEEG investigators for academic access to the Chung-Ang University Hospital EEG dataset; and the providers of OpenNeuro ds004504 (Miltiadous et al.), the OSF Eyes_closed AD/HC partition, and the Dryad P-ADIC release. We also thank the Alzheimer's Disease Neuroimaging Initiative (ADNI) for related clinical data infrastructure used in supporting cohort work outside the primary EEG external battery.


---
