# Layer 5 Domain-Shift Diagnosis

Date: 2026-08-08  
Project: EEG Digital Brain Twin (Alzheimer’s)  
Scope: TUH clinical EEG validation (`Layer 5`) only

---

## 0. Backup

Not a git repository (no `.git`). Explicit file backup created before any changes:

`backups/layer5_domain_shift_20260808_162045/`

| Path | Contents |
|------|----------|
| `data/tuh_features/` | 200 `*_features.npy` + structured npz + `processing_report.json` |
| `data/tuh_mining/` | header mining CSV/JSON |
| `data/tuh_validation/` | pre-fix sims + validation JSON/figures |
| `models_validation/complete_validation_report_v3.json` | pre-fix Layer 5 report |
| `src_tuh/process_tuh_eeg.py`, `tuh_validation.py`, `mine_headers.py` | pre-edit module copies |

Confirmed: `features_npy=200`, `v3=True`, process module backed up.

---

## 1. Pipeline comparison (training vs TUH)

### Training cohort (66 subjects → Layers 1–4)

- Source: EEGLAB `.set` under `data/raw_eeg/{AD,HC}/` (OpenNeuro-style resting EEG).
- Units: BIDS lists µV; MNE loads **Volts** (~1e−4 V RMS). **Not a µV-vs-V bug.**
- Shared Phase-1 stack:
  - `src/eeg/preprocess.py` → `EEGPreprocessor` (BP 0.5–40 Hz, **notch 50 Hz**, ICA on, 4 s epochs @ 90% overlap, **per-subject z-score**)
  - `src/eeg/feature_extraction.py` → `EEGFeatureExtractor.extract_spectral_features(..., per_epoch=False)`
  - Critical behavior: with `per_epoch=False`, epochs are **averaged in time first**, then Welch PSD / band power (`n_per_seg=256` @ 500 Hz → 20 bins).
- Epoch depth on disk: AD ≈ **758** epochs/subject, HC ≈ **1000** (stack pad); cohort-weighted ≈ **864**.
- Outputs: `data/eeg_features/{AD,HC}_{psd,band_powers,coherence,plv}.npy`

### TUH clinical pipeline (200 EDFs → Layer 5)

- Loader: `src/tuh/process_tuh_eeg.py` (TUH-only)
- Channel map AR/LE → 19 legacy 10–20 names; crop; resample **500 Hz**; **notch 60 Hz** (US)
- Reuses **shared** `EEGPreprocessor` / `EEGFeatureExtractor` / `ConnectivityAnalyzer` / `flatten_features`
- First Layer 5 run used `CROP_SECONDS=60`, **`MAX_EPOCHS_FOR_FEATURES=15`**, ICA off
- Same spectral path: time-average epochs → Welch → absolute band powers

### Shared-code risk

| Module | Shared with Layers 1–4? | Touched by this fix? |
|--------|-------------------------|----------------------|
| `src/eeg/preprocess.py` | Yes | **No** |
| `src/eeg/feature_extraction.py` | Yes | **No** |
| `src/eeg/connectivity.py` | Yes | **No** |
| `api/utils/feature_processor.py` | Yes | **No** |
| `src/utils/config.py` | Yes | **No** |
| `src/tuh/process_tuh_eeg.py` | No (TUH-only) | **Yes** |
| Training features / checkpoints | Yes | **No** |

**Fix was applied only on the TUH path.** No confirmation gate was required for shared upstream edits because none were made.

---

## 2. Root cause (with evidence)

**Primary cause:** Phase-1 spectral extraction averages **z-scored** epochs in the time domain before Welch. For roughly independent zero-mean epochs, variance (hence band power) scales as **1/N**.

| Quantity | Training | TUH (first run) |
|----------|----------|-----------------|
| Epochs averaged for PSD | ~758–1000 (ref **864**) | **15** |
| Mean band power | **3.22e−5** | **3.24e−3** |
| Ratio TUH / train | — | **≈100.7×** |
| Predicted from 864/15 | — | **≈57.6×** |
| Residual after 1/N | — | **≈1.75×** (ICA on/off, clinical vs resting, etc.) |

Same-subject control (training AD re-extract): N=15 vs N=758 → ~47× band-power inflation — matches the 1/N model.

**Ruled out:** µV vs V (both Volts post-MNE; z-score removes units); relative-vs-absolute power; different PSD API.

**Why full-2185 PCA looked “stuck”:** connectivity occupies 1710/2185 dims with O(1) values; spectral blocks are O(1e−5)–O(1e−3). Unstandardized PCA/L2 is **connectivity-dominated**, so fixing spectral scale barely moves full-vector PCA (23% → 23%) while **spectral-only** PCA overlap jumps **0% → 95.5%**.

---

## 3. Fix applied (TUH-only)

In `src/tuh/process_tuh_eeg.py`:

1. Raise default crop / max epochs toward training depth (`CROP_SECONDS=320`, `MAX_EPOCHS_FOR_FEATURES=864`) for future extracts.
2. Add **epoch-count spectral calibration**: after spectral extract, multiply PSD and band powers by  
   `scale = n_epochs_used / TARGET_N_EPOCHS_FOR_PSD` with `TARGET_N_EPOCHS_FOR_PSD=864`.  
   Coherence / PLV **unchanged**.
3. Post-hoc applied the same scale to the existing 200 feature files via `recalibrate_existing_tuh_features()` (from pristine backup restore), then re-ran twin + Layer 5 analyses.

No shared Phase-1 modules modified.

---

## 4. Before / after metrics

### Feature scale

| Metric | Before | After |
|--------|--------|-------|
| TUH mean band power | 3.24e−3 | 5.63e−5 |
| TUH / train BP ratio | **100.7×** | **1.75×** |
| Spectral-only PCA % inside train 95% radius | **0.0%** | **95.5%** |
| Spectral-only mean distance | 0.072 | 0.0010 |
| Full-2185 PCA % inside train 95% radius | 23% | 23% |
| Full-2185 mean feature distance | 13.618 | 13.618 |

### Abnormal vs normal discrimination (latent ‖Δμ‖ drug response)

| Metric | Before | After |
|--------|--------|-------|
| N abnormal / normal | 131 / 69 | 131 / 69 |
| Mean ± SD abnormal | 0.1757 ± 0.0226 | 0.1757 ± 0.0226 |
| Mean ± SD normal | 0.1788 ± 0.0277 | 0.1788 ± 0.0277 |
| Mann–Whitney p | **0.2725** | **0.2725** |
| Cohen’s d | **−0.13** | **−0.13** |
| Status | FAIL | FAIL |

**Discrimination is effectively unchanged.** The twin’s latent drug-response magnitude is insensitive to this absolute spectral rescale (encoder / connectivity-dominated representation). Therefore the original FAIL is **not** explained away as a pure scale artifact.

---

## 5. Cross-dataset direction agreement (must remain intact)

Logic untouched; re-run after calibration as sanity check:

| Metric | Before | After |
|--------|--------|-------|
| Direction agreement | **10/10** | **10/10** |
| Effect magnitude r | 0.893 | **0.908** |

**Confirmed:** direction agreement preserved (slightly improved r). Fix did not break the passing sanity check.

---

## 6. Layers 1–4 unchanged

Fingerprint from immutable `complete_validation_report_v2.json` vs values retained in `v3`:

| Metric | Value | Unchanged? |
|--------|-------|------------|
| L1 constrained MSE | 0.031067 | Yes |
| L1 Pearson | 0.871692 | Yes |
| L1 AD/HC accuracy | 0.786813 | Yes |
| L2 TVB match | 9/10 | Yes |
| L2 effect-size r | 0.392847 | Yes |
| L3 / L4 status | PASS / PASS | Yes |

Shared code paths for Layers 1–4 were **not executed or modified**. Overall remains **4/5 layers PASS**.

---

## 7. Verdict (skeptical)

| Question | Answer |
|----------|--------|
| Was there a real preprocessing scale bug? | **Yes** — 1/N epoch averaging with N=15 vs ~864. |
| Did TUH-only calibration fix spectral domain shift? | **Yes** — BP ratio 100.7×→1.75×; spectral PCA overlap 0%→95.5%. |
| Did that rescue abnormal-vs-normal discrimination? | **No** — p and d unchanged. |
| Should Layer 5 be marked PASS because of the fix? | **No.** Marking PASS would credit a normalization choice for a test that did not move. |
| Residual ~1.75× spectral ratio? | Acceptable secondary domain gap (ICA off, clinical vs resting). Not forced to unit ratio with an extra affine “fit.” |
| Honest Layer 5 status | **FAIL** (discrimination criterion). Spectral scale issue **resolved**; proxy-label hypothesis (TUH abnormal ≠ AD enrichment) remains the credible explanation for the null. |

**Optional nuance:** If Layer 5 were redefined around cross-dataset pharmacodynamic consistency alone, that sub-check is strong (10/10, r≈0.91). Under the stated abnormal-vs-normal gate, Layer 5 stays **FAIL**.

Artifacts:
- `data/tuh_validation/layer5_calibration_before_after.json`
- `models/validation/complete_validation_report_v3.json` (annotated)
- `src/tuh/process_tuh_eeg.py` (future extracts auto-calibrate)
- `src/tuh/run_layer5_calibration.py`
