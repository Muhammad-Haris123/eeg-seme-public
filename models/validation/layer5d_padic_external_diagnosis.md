# Layer 5d — P-ADIC Dryad AD vs HC external validation

Date: 2026-08-08  
Scope: **Addition** to Layer 5 (TUH) and Layer 5b (OSF). Does **not** replace either.

---

## 0. Backup / provenance

Pre-work backup: `backups/layer5d_padic_20260808_184825/`

Raw mats (after browser download; automation blocked by Dryad AWS WAF):

| File | Path | Size | SHA256 |
|------|------|------|--------|
| AD | `E:\padic_external\alz_c1_new.mat` | 3813216803 | `0c2fcee9…1576d` (match) |
| HC | `E:\padic_external\controls_c1_new.mat` | 7165225913 | `3272bb5b…f5955` (match) |
| Notes | `E:\padic_external\AUTHOR_DATASET_SHOR_BENNINGER.txt` | 4374 | present |

Project pointer: `data/padic_external/README_LOCATION.txt`  
Features / report: `data/padic_external/`  
Results JSON: `models/validation/layer5d_padic_results.json`  
Annotated into `complete_validation_report_v3.json` as `layer5d_padic_external` only.

**TUH Layer 5 and OSF Layer 5b fingerprints unchanged** (Layer 5 still FAIL p≈0.272; Layer 5b still PASS_WITH_CAVEATS p≈0.00028).

---

## 1. What was downloaded

- DOI: [10.5061/dryad.8gtht76pw](https://doi.org/10.5061/dryad.8gtht76pw)  
- Paper: Shor et al., PLoS ONE 2021 (`10.1371/journal.pone.0255529`)  
- Used: **AD + controls only** (skipped MCI / depression / schizophrenia by design)  
- Labels: AD/MCI by two senior neurologists per **NIA–AA**; controls with explicit neuro/psych exclusions (author notes)  
- Hardware: Nihon Kohden, 19-electrode 10–20, resting EO+EC, high-pass 1 Hz already applied on release

### On-disk structure (HDF5 / MATLAB v7.3)

| Group | Key | Cell layout | Resolved recordings | Age (mean, range) | sfreq in file |
|-------|-----|-------------|---------------------|-------------------|---------------|
| AD | `alz_r` | `G` (3×43) | **49** | ~70.4 (55–87) | 200 or 500 Hz |
| HC | `controls_r` | `G` (1×96) | **96** | ~52.4 (19–80) | 200 or 500 Hz |

- Arrays are `(n_times, 19)` — 19-channel axis confirmed  
- Durations typically many minutes (after crop: 320 s used for features)  
- Metadata fields: `age`, `g` (sfreq), `sex` (decode unreliable), plus AD-only `birth` / `eegdate` / `eeg_files`  
- **No embedded channel-name strings** in the mats

Recording-level N = **145** (= 49+96), matching the independent transportability study’s external n. Paper table lists AD n=40 / HC n=95 participants — AD file has multiple rows per subject column, so units are **recordings**, not unique subjects.

---

## 2. Loader approach

New modules only (shared `src/eeg/*` **not** modified):

- `src/tuh/process_padic_eeg.py` — HDF5 loader → MNE `RawArray` → reuse `extract_tuh_features`  
- `src/tuh/run_layer5d_padic.py` — twin + MWU/Cohen’s d + PCA + cross-dataset + optional disease-label ablation  
- `src/tuh/inspect_padic_mat.py` — structure probe  

Pipeline reuse:

- `EEGPreprocessor` / `EEGFeatureExtractor` / `ConnectivityAnalyzer` via TUH feature path  
- Notch **50 Hz** (Israel / EU)  
- Crop **320 s**; max epochs **864**; spectral calibration `TARGET_N_EPOCHS_FOR_PSD=864`  
- Resample 200 Hz → 500 Hz when needed  

**Processing outcome:** 145/145 OK, 0 fails. All used **791 epochs** after crop (scale ≈ 791/864 ≈ 0.916).

---

## 3. Montage mapping (explicit — not silent)

| | Order |
|--|--------|
| Assumed P-ADIC (Nihon-style, from independent Dryad loader) | Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2 |
| Training `STANDARD_10_20_CHANNELS` | Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz |

- Same **19 names** (legacy T3–T6): yes  
- Same **order**: **no** → remapped by name before features  
- Channel names embedded in `.mat`: **no** → assumption required; flagged in code + report  

If the true on-disk electrode order differs from the assumed Nihon layout, spatial features (coherence/PLV) would be wrong. Spectral PCA overlap was strong (below), which argues against a total scramble but does **not** prove the assumed order.

---

## 4. Spectral / epoch calibration check

| Check | Value |
|-------|--------|
| Epochs / recording | min=med=max=**791** |
| Target N for PSD | 864 |
| Band-power \|mean\| ratio P-ADIC / train | **~3.34×** after calibration |

Not a repeat of the TUH N=15 / ~100× bug. Residual ~3× domain gap remains (routine clinical EO/EC, filtering, age mix) — not force-fit to 1.0.

PCA vs training 95% radius:

| Feature set | Overlap |
|-------------|---------|
| Full 2185-D | **46.9%** (moderate) |
| Spectral-only (0:475) | **90.3%** (good) |

---

## 5. Layer 5d discrimination (latent mean drug-response ‖Δμ‖)

| Metric | P-ADIC 5d | TUH Layer 5 | OSF Layer 5b |
|--------|-----------|-------------|--------------|
| Contrast | AD vs HC | Abnormal vs Normal | AD vs HC |
| N | **49 / 96** | 131 / 69 | 80 / **12** |
| Mean ± SD (case) | 0.2138 ± 0.0144 | 0.1757 ± 0.0226 | 0.1669 ± 0.0023 |
| Mean ± SD (control) | 0.2147 ± 0.0143 | 0.1788 ± 0.0277 | 0.1643 ± 0.0153 |
| Mann–Whitney p | **0.968** | 0.272 | **0.00028** |
| Cohen’s d | **−0.065** | −0.13 | **0.44** |
| Direction | AD **<** HC (wrong way, tiny) | Abn < Nor | AD > HC |
| Status | **FAIL** | FAIL | PASS_WITH_CAVEATS |

Disease-label ablation (all disease_label fixed):

| Conditioning | p | d | Status |
|--------------|---|---|--------|
| Fixed 0 | 0.53 | −0.16 | FAIL |
| Fixed 1 | 0.70 | −0.14 | FAIL |

Cross-dataset drug-effect **direction** agreement vs training cohort simulations: **10/10**, r≈0.92 — twin still produces coherent signed band shifts; they simply do **not** separate P-ADIC AD from HC on latent response magnitude.

---

## 6. Comparison / interpretation

| External attempt | What it tests | Result | Honest takeaway |
|------------------|---------------|--------|-----------------|
| TUH 5 | Proxy abnormal/normal | Null | Weak label |
| OSF 5b | Folder AD/HC, HC n=12 | Positive w/ caveats | Underpowered HC; weak label docs |
| **P-ADIC 5d** | NIA–AA AD vs excluded HC, HC n=96 | **Clear null** | Best free labeled external so far for *this* endpoint — still null |

Independent literature on the **same** Dryad release + OpenNeuro derivation cohort reports modest **spectral slowing** AD/CN transport (AUC≈0.70). That does **not** imply our constrained-MLP **latent drug-response** magnitude must discriminate. Here it does not (p≈0.97).

---

## 7. Caveats (do not overclaim either way)

1. Channel order **assumed** (not in file).  
2. Eyes open/closed **mixed**; no EO/EC tags.  
3. **Age imbalance** (AD ~70 vs HC ~52) — would tend to inflate, not erase, AD–HC differences if age drove the twin; here differences are absent anyway.  
4. Units are **recordings** (AD multi-row cells), not unique subjects.  
5. Residual spectral scale ~3× vs train.  
6. OSF 5b positive result is **not** overturned, but P-ADIC is the stronger-powered labeled null for the same latent endpoint.

---

## 8. Verdict

**Layer 5d = FAIL (null).**  

P-ADIC was the right dataset to try: true external, real NIA–AA AD labels, usable 19ch 10–20, adequate HC n, epoch depth calibrated. After explicit montage remapping and epoch-aware spectral scaling, AD vs HC latent drug-response magnitude is **indistinguishable**.  

Do **not** treat this as a PASS. Do **not** replace TUH or OSF entries. Journal framing: external transport of this twin endpoint remains **unproven** on the best freely available clinical AD/HC EEG set we could run; OSF’s positive signal stays caveated by HC n=12.

### Code / artifacts

- Loader: `src/tuh/process_padic_eeg.py`  
- Runner: `src/tuh/run_layer5d_padic.py`  
- Features: `data/padic_external/features/`  
- Sims: `data/padic_external/validation/`  
- PCA fig: `models/validation/figures/layer5d_padic_pca.png`
