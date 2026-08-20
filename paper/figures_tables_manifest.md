# Figures and tables manifest

Final manuscript numbering order. Every figure/table must be cited in prose
**before** it appears. Update when adding, removing, or renumbering.

| ID | One-line description | Source script / artifact | First-cite sentence (exact) | Status |
|----|----------------------|--------------------------|-----------------------------|--------|
| Fig. 1 | Constrained CVAE architecture (horizontal; GPT layout; locked dims) | Preferred: `paper/figures/fig1_architecture_chatgpt.png` (also `fig1_architecture.png`). Optional matplotlib: `src/figures/make_fig1_architecture.py` | "The architecture data flow (EEG tensors and ChemBERTa drug vector into encoders, fusion with disease label, CVAE \(\mu\)/log-variance, and feature reconstruction) is illustrated in Fig. 1." (`02_methods.md` §2.1) | generated (use ChatGPT PNG) |
| Fig. 2 | Locked fold-0 seed-42 checkpoint-specific external signature concordance (r on TUH / OSF / P-ADIC; all 10/10) | Script: `src/figures/make_fig2_direction_agreement.py` → `paper/figures/fig2_direction_agreement.{png,svg}`. Data: `models/validation/complete_validation_report_v3.json` keys `layer5.cross_dataset` (r=0.908, 10/10, n=200), `layer5b_ad_labeled_external.cross_dataset` (r=0.851, 10/10, n=92), `layer5d_padic_external.cross_dataset` (r=0.918, 10/10, n=145). Mirrors: `data/ad_labeled_external/validation/layer5b_results.json`, `models/validation/layer5d_padic_results.json`. No CI in JSON. | "Table 3 and Fig. 2 place the three locked seed-42 concordance values side by side…" (`03_results_mechanism.md` §3.2) | generated |
| Fig. 3 | Forest plot of Cohen's d for twin magnitude discrimination (TUH / OSF / P-ADIC / CAUEEG) | Script: `src/figures/make_fig3_diagnostic_null.py` → `paper/figures/fig3_diagnostic_null.{png,svg}`. Data: `complete_validation_report_v3.json` → `layer5.abnormal_vs_normal` (d=-0.13, p=0.272, n=131+69); `layer5b_ad_labeled_external.discrimination` (d=0.44, p=0.000278, n=80+12, PASS_WITH_CAVEATS); `layer5d_padic_external.discrimination` (d=-0.07, p=0.968, n=49+96); `layer5e_caueeg_external.discrimination.primary_dementia_vs_normal` (d=-0.00, p=0.516, n=291+436). Mirrors: `layer5b_results.json`, `layer5d_padic_results.json`, `layer5e_caueeg_results.json`. 95% CI computed from d,n1,n2 (Hedges–Olkin SE); not stored in JSON. | "Table 4 summarizes the primary magnitude contrasts… Fig. 3 places the corresponding effect sizes…" (`04_results_diagnosis.md` §3.3) | generated |
| Fig. 4 | PCA (PC1–PC2): training cohort vs CAUEEG; train 95% radius; overlap ≈ 0.611 | Regenerated (not a direct reuse): `src/figures/make_fig4_pca_recolor.py` → `paper/figures/fig4_pca.{png,svg}`. Same transform/data as `src/tuh/run_layer5e_caueeg.py::pca_overlap`. Overlap key: `layer5e_caueeg_external.pca.overlap_fraction_within_train_r95` = 0.611408. | "…good PCA overlap with the training cohort (overlap fraction within the training 95% radius = 0.611; Fig. 4)." (`04_results_diagnosis.md` §3.2) | generated (recolored) |
| Fig. 5 | Encoding analysis: θ/α vs twin μ_base readouts (CAUEEG Dementia vs Normal) | Script: `src/figures/make_fig5_encoding_analysis.py` → `paper/figures/fig5_encoding_analysis.{png,svg}`. Data: `layer5e_caueeg_latent_probe_results.json`, `layer5e_caueeg_classifier_head_results.json`. | "Fig. 5; Fig. 6; Table 5" (`03_results_mechanism.md` §3.4) | generated |
| Fig. 6 | Compact SEME Encoding bake-off panel (classical vs latent AUCs) | Script: `src/figures/make_fig_bakeoff_encoding.py` → `paper/figures/fig_bakeoff_encoding.{png,svg}`. Data: locked probe/head/raw2185 JSON + optional DeLong. | "Fig. 5; Fig. 6; Table 5" (`03_results_mechanism.md` §3.4) | generated |
| Fig. 7 | Constraint-weight sweep: direction agreement / r vs CAUEEG latent AUC | Script: `src/figures/make_fig6_constraint_strength.py` → `paper/figures/fig6_constraint_strength.{png,svg}`. Data: `constraint_strength_sweep.json`. | "An exploratory fold-0 α_c sweep (Fig. 7)…" (`03_results_mechanism.md` §3.5) | generated |
| Table 1 | External assessment datasets (TUH / OSF / P-ADIC / CAUEEG): n, labels, rate, montage, study role | Script: `src/tables/make_table1_datasets.py` → `paper/tables/table1_datasets.{md,tex}`. Metadata: `layer5_domain_shift_diagnosis.md`, `layer5b_ad_labeled_external_diagnosis.md`, `layer5d_padic_external_diagnosis.md`, `layer5e_caueeg_external_diagnosis.md`; n from `complete_validation_report_v3.json`. | "Cohort size, labels, sampling rate, montage, and study role are summarized in Table 1." (`02_methods.md` §2.4) | generated |
| Table 2 | Matched unconstrained vs constrained direction (TUH/OSF/P-ADIC) + prior-only note | Script: `src/tables/make_table5_ablation_direction.py` → `paper/tables/table5_ablation_direction.{md,tex}` (manuscript number Table 2). Data: `unconstrained_external_battery.json`, v3, `prior_only_direction.json`. | "Table 2 summarizes constrained versus unconstrained direction on TUH, OSF, and P-ADIC so the locked-checkpoint control contrast is visible before the external constrained replications." (`03_results_mechanism.md` §3.1) | generated |
| Table 3 | Checkpoint-specific locked fold-0 seed-42 external signature concordance (10/10 on TUH / OSF / P-ADIC) | Script: `src/tables/make_table2_direction_agreement.py` → `paper/tables/table2_direction_agreement.{md,tex}` (manuscript number Table 3). | "Table 3 and Fig. 2 place the three locked seed-42 concordance values side by side…" (`03_results_mechanism.md` §3.2) | generated |
| Table 4 | All twin-magnitude diagnostic contrasts | Script: `src/tables/make_table3_diagnostic_results.py` → `paper/tables/table3_diagnostic_results.{md,tex}` (manuscript number Table 4). | "Table 4 summarizes the primary magnitude contrasts and secondary checks so n, p, Cohen's d, and status can be compared across the escalation sequence." (`04_results_diagnosis.md` §3.3) | generated |
| Table 5 | Encoding analysis AUCs; pairs with Fig. 5 | Script: `src/tables/make_table4_encoding_results.py` → `paper/tables/table4_encoding_results.{md,tex}` (manuscript number Table 5). | "Table 5 reports the corresponding out-of-fold (OOF) AUCs…" (`05_results_encoding_analysis.md` §3.4) | generated |
| Table 6 | Fixed-fold multi-seed sensitivity of external r and signs (seeds 42/7/21/123/2024) | Hand-authored from artifact: `paper/tables/table6_multiseed_direction.tex`. Data: `models/validation/multiseed_robustness/aggregate_results.csv` (and `FINAL_MULTISEED_REPORT.md`). Seed 42 = locked historical; others = alternate train seeds under frozen `split_seed=42`. | "Table 6 reports continuous r and sign agreement for every seed×cohort cell under that fixed-fold design…" (`03_results_mechanism.md` training-seed subsection) | generated |
| Table A.1 | constrained_mlp widths and parameter count 2,648,777 | Instantiation of `UpgradedDigitalTwinModel`; `computational_requirements.json` | "Layer widths, activations, and the measured parameter count are listed in Table A.1." (`02_methods.md` §2.1) | generated |
| Table A.2 | Checkpoint size, dummy inference, package versions | `computational_requirements.json` | "Measured parameter count, checkpoint size, dummy-batch inference time, and package versions are listed in Table A.2." (`02_methods.md` §2.3) | generated |
| Fig. A.1 | Fold-0 constrained train/val loss curves | `src/figures/make_fig_s1_training_curves.py` from `constraint_curves.fold_0` | "Fold-0 train and validation total loss… are shown in Fig. A.1." (`02_methods.md` §2.3) | generated |
| Algorithm 1 | Constrained CVAE training procedure | `paper/tables/algorithm1_training.tex` from `train_phase2_upgraded.py` / `losses.py` | "The training procedure is summarized in Algorithm 1." (`02_methods.md` §2.2) | generated |
| Table A.3 | Methodological safeguards and remaining scope boundaries | `paper/tables/table_s_reviewer_defense.tex` | "Table A.3 lists methodological safeguards and remaining scope boundaries." (`S1_reproducibility.md`) | generated |

### Graphical abstract (Elsevier/CMPB front matter; not a numbered figure)

| ID | One-line description | Source script / artifact | First-cite sentence (exact) | Status |
|----|----------------------|--------------------------|-----------------------------|--------|
| Graphical abstract | Elsevier/CMPB graphical abstract: EEG features + ChemBERTa-conditioned CVAE with spectral/connectivity constraints; directional-signature concordance vs prior-only control | Author-provided figure saved as `paper/figures/graphical_abstract.png` (not regenerated from `src/figures/make_graphical_abstract.py`). Locked numbers on the figure: r = 0.908 / 0.851 / 0.918; 10/10; prior-only r = 0.636. | **Not a numbered in-text figure.** Elsevier front-matter only. | author-provided |

### Caption placement (Elsevier)

- **Figures:** caption **below** the figure.
- **Tables:** caption **above** the table (Table 1 `.tex` already places `\caption` above `\begin{tabular}`).

### Fig. 1 caption (placeholder; finalize at assembly)

**Fig. 1.** Architecture of the pharmacodynamically constrained conditional variational autoencoder (CVAE) digital twin. Resting-state EEG features (2185-D) and ChemBERTa drug embeddings are encoded and fused with a disease label, then mapped by a CVAE whose training includes pharmacodynamic band and connectivity constraints. Path (a) uses drug-conditioned decoding for drug-response direction evaluation. Path (b) uses CVAE latent means \(\mu_{\mathrm{base}}\), \(\mu_{\mathrm{don}}\), and \(\mu_{\mathrm{mem}}\) to compute drug-response magnitude \(\bar{s}\). Training cohort: N = 66 (AD 37 / HC 29).

### Fig. 2 caption notes (do not invent CI)

Caption must state that error bars / CIs are omitted because `effect_magnitude_correlation` has no SE/CI in the validation JSON. Report exact agreement fractions (10/10) and n per cohort.

### Fig. 3 caption notes

Caption must state: (1) primary contrasts and exact n/p/d from v3 JSON keys above; (2) 95% CIs for Cohen's d were computed from reported d and group sizes via Hedges–Olkin SE, not stored in artifacts; (3) OSF open marker = PASS_WITH_CAVEATS (HC n=12); (4) warm-gray styling marks the magnitude/diagnostic endpoint path.

### Fig. 4 caption notes

PCA of 2185-D features fit on training N=66; CAUEEG n=1122 projected. Report overlap_fraction_within_train_r95 = 0.611 (grade good). Dashed circle = training 95% radius. Teal CAUEEG markers emphasize the supporting domain-overlap check (magnitude null is not a failed PCA/scale check). No on-figure title.

### Fig. 5 caption notes

Central encoding figure. Amber = θ/α spectral/feature reference; teal/navy = frozen-twin μ_base readouts. Gray band = latent-probe permutation null p5–p95 (not a CI on the bar AUCs). Report gap 0.082 between θ/α (0.675) and best head (0.593). MLP/HGB fall inside or at the edge of the null band; readout sophistication does not close the feature–latent gap under tested heads. n = 727 Dementia vs Normal.

### Fig. 6 caption notes

Hero ablation figure. Report \(\alpha_c\) grid 0, 0.01, 0.05, 0.1, 0.2, 0.5 from `constraint_strength_sweep.json`. Direction agreement (mean TUH/P-ADIC %) locks by 0.05; effect-vector \(r\) peaks at 0.1; CAUEEG latent AUC is non-monotone (do not claim a causal trade-off curve). Fold-0 selection matches paper checkpoint rule. No post-dose EEG.

---

## Candidate source figures (not yet assigned manuscript numbers)

| Artifact path | Likely role |
|---------------|-------------|
| `models/validation/figures/layer5d_padic_pca.png` | P-ADIC vs train PCA overlap |
| `models/validation/figures/layer5e_caueeg_pca.png` | CAUEEG vs train PCA overlap |
| `data/ad_labeled_external/validation/fig_layer5b_pca_overlap.png` | OSF Layer 5b PCA (if used) |

Copy finalized, renumbered assets into `paper/figures/` and `paper/tables/`
only after assigning IDs and first-cite sentences above.
