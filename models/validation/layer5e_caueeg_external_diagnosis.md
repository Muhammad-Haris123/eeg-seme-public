# Layer 5e — CAUEEG Normal / MCI / Dementia external validation

Date: 2026-08-09  
Scope: **Addition** to Layer 5 (TUH), Layer 5b (OSF), Layer 5d (P-ADIC). Does **not** replace prior layers.

---

## 0. Access / provenance

- Access granted by Min-jae Kim (CAUEEG team) after academic application  
- Raw extract: `E:\caueeg_external\extracted\caueeg-dataset\`  
- Project features: `data/caueeg_external/`  
- Results JSON: `models/validation/layer5e_caueeg_results.json`  
- Annotated into `complete_validation_report_v3.json` as `layer5e_caueeg_external` only  

**Cite:** Kim, M. J., Youn, Y. C., & Paik, J. (2023). Deep learning-based EEG analysis to classify normal, mild cognitive impairment, and dementia: algorithms and dataset. *NeuroImage*, 120054. https://doi.org/10.1016/j.neuroimage.2023.120054

Academic / non-commercial use only. Do not redistribute raw files.

---

## 1. Dataset used

| Item | Value |
|------|--------|
| Benchmark | `dementia-no-overlap.json` (train+val+test pooled for external test) |
| Classes | Normal / MCI / Dementia |
| N processed | **1122 / 1122 OK, 0 fail** |
| N by class | Normal **436**, MCI **395**, Dementia **291** |
| EEG | 19ch AVG-referenced 10–20 (+ EKG/Photic dropped), **200 Hz** native |
| Crop / epochs | 320 s → **791** epochs typical (scale ≈ 0.916 vs target 864) |
| Notch | **60 Hz** (South Korea) |
| Twin disease_label | Dementia=1; Normal/MCI=0 |

**Caveat:** “Dementia” is a **clinical spectrum** (AD and non-AD subtypes). AD-tagged symptom subset analyzed separately (n=214).

---

## 2. Loader

New modules only (`src/eeg/*` untouched):

- `src/tuh/process_caueeg_eeg.py` — EDF load via `load_tuh_edf` (strips `-AVG`) + `extract_tuh_features`
- `src/tuh/run_layer5e_caueeg.py` — twin sims, MWU/Cohen’s d, Kruskal, PCA, v3 annotation

---

## 3. Scale / domain checks

| Check | Result |
|-------|--------|
| Band-power \|mean\| ratio (CAUEEG / train) | **≈ 1.10** (acceptable; near calibration target) |
| Epoch median | 791 |
| PCA overlap within train 95% radius | **good** (~0.61) |
| Figure | `models/validation/figures/layer5e_caueeg_pca.png` |

Spectral scale is **not** the failure mode (unlike early uncalibrated TUH).

---

## 4. Primary twin result — FAIL (null)

**Endpoint:** mean latent drug-response magnitude \(0.5(\|\mu_{don}-\mu_{base}\| + \|\mu_{mem}-\mu_{base}\|)\).

### Dementia vs Normal (primary)

| | Dementia | Normal |
|--|----------|--------|
| N | 291 | 436 |
| Mean response | ≈ 0.2117 | ≈ 0.2117 |
| Mann–Whitney p | ≈ **0.52** | |
| Cohen’s d | ≈ **−0.002** | |
| Status | **FAIL** | |

### Other contrasts (all FAIL / null)

| Contrast | p | d | Status |
|----------|---|---|--------|
| Dementia vs MCI | ≈ 0.88 | ≈ 0.07 | FAIL |
| MCI vs Normal | ≈ 0.36 | ≈ −0.07 | FAIL |
| AD-tagged Dementia vs Normal | ≈ 0.62 | ≈ −0.01 | FAIL |
| Kruskal 3-group | p ≈ 0.64 | — | null |

**Interpretation:** On CAUEEG, the constrained twin’s **latent drug-shift magnitude does not separate** clinical Normal / MCI / Dementia. Means are essentially identical (~0.212). This mirrors P-ADIC Layer 5d null AD vs HC twin discrimination, despite good scale and moderate–good PCA overlap.

---

## 5. Feature-space checks (EEG content ≠ twin endpoint)

Raw/calibrated **features do separate** Dementia vs Normal on several spectral summaries, even though the twin latent drug-shift does **not**:

| Metric | MWU p (Dem vs Norm) | Cohen’s d | Note |
|--------|---------------------|-----------|------|
| theta/alpha ratio | **≈ 1.2×10⁻¹⁵** | **≈ +0.47** | Strong; expected AD-like slowing |
| mean alpha power | ≈ 8.3×10⁻⁶ | ≈ −0.25 | Lower alpha in dementia |
| flat L2 norm | ≈ 3.3×10⁻⁴ | ≈ −0.29 | |
| mean \|bandpower\| | ≈ 0.006 | ≈ −0.17 | Weak–moderate |
| mean theta | ≈ 0.21 | ≈ +0.19 | NS alone |

**Takeaway:** Domain EEG carries clinically sensible spectral structure (higher θ/α in dementia). The **null is specific to the twin drug-response latent magnitude**, not “CAUEEG features are random.”

---

## 6. Disease-label ablation

Fixed `disease_label` for all subjects and re-ran twin:

| Setting | Dem vs Norm p | d | Status |
|---------|---------------|---|--------|
| All disease=0 | ≈ 0.41 | ≈ −0.04 | FAIL |
| All disease=1 | ≈ 0.78 | ≈ 0.00 | FAIL |

Null is **not** rescued by forcing the disease conditioning bit.

---

## 7. What this does *not* claim

- Not a treatment-response validation (CAUEEG has **no** drug outcome labels)  
- Not same-subject EEG–MMSE correlation (no ADNI-style clinical scores here)  
- Not proof EEG features lack diagnostic signal (θ/α ratio clearly differs)  
- Does not unlock high-tier ART / drug-response publication claims alone  

---

## 8. Relation to prior external layers

| Layer | Resource | Twin AD/impaired vs control | Notes |
|-------|----------|----------------------------|-------|
| L5 | TUH abnormal/normal | FAIL | Proxy labels; no AD drugs |
| L5b | OSF Eyes_closed AD/HC | PASS_WITH_CAVEATS | Small HC (n=12) |
| L5d | P-ADIC AD/HC | FAIL | Null; age imbalance |
| **L5e** | **CAUEEG N/MCI/Dem** | **FAIL** | Large N; clinical labels; still null twin |

CAUEEG was the right next dataset after TUH/ADNI gaps. Large-N clinical labels + good scale still yield **null twin group separation**, while spectral features show expected slowing — an honest methods result.

---

## 9. Artifacts

- Features: `data/caueeg_external/features/*_features.npy`  
- Processing report: `data/caueeg_external/processing_report.json`  
- Twin sims: `data/caueeg_external/validation/*_sims.npz`  
- Ablations: `data/caueeg_external/validation/ablation_disease_{0,1}/`  
- Run log: `data/caueeg_external_run.log`  
- Results: `models/validation/layer5e_caueeg_results.json`  
- PCA figure: `models/validation/figures/layer5e_caueeg_pca.png`  
