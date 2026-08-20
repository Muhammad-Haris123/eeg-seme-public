# 5. Conclusion

This study shows that a literature-constrained EEG-drug CVAE can reproduce the literature-derived directional signature at a locked fold-0 seed-42 checkpoint (TUH/OSF/P-ADIC secondary sign agreement 10/10; continuous \(r = 0.908 / 0.851 / 0.918\) at that checkpoint only). Directional/sign agreement was substantially more stable than continuous fidelity across the tested fold/seed sensitivity battery: five-fold means fell to about 0.37 to 0.39, and fixed-fold multi-seed runs (train seeds 7, 21, 123, 2024) failed to reproduce the high continuous-\(r\) regime while mostly retaining 9/10 to 10/10 signs. Under the same fold freeze, matched unconstrained twins also showed unstable continuous \(r\), while constrained sign agreement exceeded unconstrained agreement in every tested seed\(\times\)cohort comparison without establishing continuous-\(r\) stabilization. A prior-only baseline already reaches 10/10 signs at mean \(r = 0.636\), and an equal-weight five-fold ensemble (\(r = 0.642 / 0.626 / 0.627\)) does not clearly exceed that prior. Latent diagnostic label information is substantially attenuated under the tested readouts relative to explicit feature references (latent ROC-AUC = 0.579 versus theta/alpha 0.675 and 2185-D logistic 0.699, a feature-space reference not a CVAE score; best head OOF AUC = 0.593), and magnitude-based diagnosis does not reliably transfer. The findings therefore support a scoped measurement framework for constrained EEG-drug simulation and hypothesis generation, not a clinically validated pharmacodynamic digital twin, not seed-stable continuous external fidelity, and not standalone external diagnosis.

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

Training resting-state EEG features used in this project are from OpenNeuro accession ds004504 (Miltiadous et al. AD/HC resting EEG), processed to the feature arrays described in Methods. Temple University Hospital (TUH) clinical EEG is available to researchers under the TUH corpus data use agreement and cannot be redistributed by the authors. The OSF Eyes_closed AD/HC features used for the AD/HC external magnitude and direction checks were imported from a public OSF-hosted partition already present in the project archive. P-ADIC AD and control matrices are available from Dryad (DOI 10.5061/dryad.8gtht76pw) under the release terms of Shor et al. CAUEEG raw recordings were obtained by academic application to the CAUEEG team (access granted by Min-jae Kim; Kim et al., NeuroImage 2023) for non-commercial research use and must not be redistributed. Derived project artifacts that can be shared subject to those upstream terms include processed feature arrays generated in this repository (where redistribution is allowed), validation JSON summaries, and analysis code. Raw CAUEEG and TUH EDFs will not be redistributed by the authors. File-level paths are listed in Supplementary Material S1.

---

## Code availability

Analysis and figure-generation code are available at https://github.com/Muhammad-Haris123/eeg-seme-public (commit `31e4249`). Locked fold-0 CVAE checkpoints (`checkpoint_constrained.pt`, `checkpoint_unconstrained.pt`) are deposited on Zenodo (DOI https://doi.org/10.5281/zenodo.22028681; version v0.1.0). Reproducibility details, environment pins, and the one-command regenerator `src/paper/regenerate_cmpb_artifacts.py` are listed in `paper/CODE_AVAILABILITY.md` and Supplementary Material S1. The Python environment is the project virtualenv `eeg_twin` (Python 3.10); install from `requirements.txt` into that environment. Raw CAUEEG and TUH recordings remain under provider terms and are not redistributed.

---

## Glossary

Terms below follow Methods and Results. They do not introduce new endpoints or claims.

- **SEME:** Signed concordance, Effect-vector fidelity, Magnitude transfer, Encoding retention (evaluation protocol).
- **CVAE:** conditional variational autoencoder.
- **EEG:** electroencephalography.
- **PD:** pharmacodynamic. In this paper the label names the literature-prior training objective, not empirical post-dose pharmacodynamic validation.
- **ROC-AUC:** area under the receiver operating characteristic curve.
- **OOF:** out-of-fold.
- **TUH:** Temple University Hospital EEG corpus.
- **OSF:** Open Science Framework Eyes_closed AD/HC set used here.
- **CAUEEG:** Chung-Ang University Hospital EEG dataset.
- **P-ADIC:** p-adic quantum potential EEG cohort (Shor et al.; Dryad).
- **ChemBERTa:** chemical Bidirectional Encoder Representations from Transformers. Here, frozen precomputed embeddings for Donepezil and Memantine, not a large-library chemistry test.
- **MSE:** mean squared error (reconstruction term).
- **KL:** Kullback-Leibler divergence (CVAE regularizer).
- **ICA:** independent component analysis (FastICA on the training path).
- **FIR:** finite impulse response filter (training band-pass 0.5-40 Hz).
- **PCA:** principal component analysis.
- **Latent representation:** CVAE latent mean under specified conditioning (baseline: zero-drug). Residual diagnostic content is scored with probes, not as an information-theoretic bottleneck.
- **Drug-response signature:** the training constrained signature (mean simulated drug-minus-baseline feature vector). External direction scoring is concordance with that signature, not post-dose pharmacodynamic validation.
- **Digital twin:** the constrained CVAE used for in silico EEG-drug signature simulation. It is not a clinically validated pharmacodynamic model.
- **Effect vector:** cohort-mean drug-minus-baseline feature vector used to compute Pearson r and block-level sign agreement.

---

## Declaration of competing interests

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## Acknowledgments

We thank the Temple University Hospital (TUH) EEG Corpus team for access to clinical EEG under the TUH data use agreement; Min-jae Kim and the CAUEEG investigators for academic access to the Chung-Ang University Hospital EEG dataset; and the providers of OpenNeuro ds004504 (Miltiadous et al.), the OSF Eyes_closed AD/HC partition, and the Dryad P-ADIC release. We also thank the Alzheimer's Disease Neuroimaging Initiative (ADNI) for related clinical data infrastructure used in supporting cohort work outside the primary EEG external battery.
