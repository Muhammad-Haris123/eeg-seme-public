# Layer 5b — AD-labeled external validation

Date: 2026-08-08  
Scope: **Addition** to Layer 5 (TUH). Does **not** replace TUH results unless you explicitly decide to supersede.

---

## 0. Backup

Not a git repo. Timestamped backup before this work:

`backups/layer5b_ad_labeled_20260808_165411/`

Confirmed contents:
- `data/tuh_features/` (200 feature files)
- `data/tuh_mining/`, `data/tuh_validation/`
- `models_validation/complete_validation_report_v3.json`
- `src_tuh/*.py` (6 modules)

TUH Layer 5 p-value fingerprint unchanged after this run (`tuh_layer5_untouched: true`).  
**Shared `src/eeg/*` modules were not modified.**

---

## 1. Candidate dataset research

| Dataset | N (groups) | Format | Rate / montage | Label quality | Freely usable here? | Verdict |
|---------|------------|--------|----------------|---------------|---------------------|---------|
| **Miltiadous OpenNeuro ds004504** | AD 36, FTD 23, HC 29 | BIDS `.set` | 500 Hz, 19ch 10–20 | Clinician-confirmed AD/FTD/CN + MMSE (AHEPA neurology) | Yes, but… | **Cannot use as external** — this **is** the training cohort (project: AD 37 / HC 29 from same OpenNeuro-style `.set` tree). Circular. |
| **CAUEEG** (Chung-Ang) | Dementia / MCI / Normal (large clinical set) | EDF + JSON annotations | Clinical EEG | Hospital diagnosis labels | **No** — ethics form + committee approval required; not downloadable in-session | Deferred |
| **CAUEEG test-only subset** | Real test splits + fake train | Restricted Google Drive | Same | Same | Partial / gated | Not used |
| **BrainLAT syn51549340** | AD 35, HC 31 (local) | EEGLAB `.set` | 512→500 Hz, **128ch A1–A19** | Clinical AD/HC groups | Yes (already on disk) | **Not primary** — **no A1–A19 → 10–20 mapping**; prior external classification ~chance. Serious montage risk. |
| **OSF Eyes_closed** (`testing_data/AD` vs `Healthy`) | AD 80, HC 12 | Per-channel `.txt` | 256→500 Hz, **19ch 10–20** | Folder-level AD vs Healthy partitions | Yes (already imported as `data/external_osf`) | **Chosen primary** for Layer 5b |
| TUH abnormal/normal (Layer 5) | 131 / 69 used | EDF | 250/… → 500 Hz, 10–20 | Generic EEG abnormality — **not** AD | Yes | Prior null (proxy label) |

### Skeptical notes on labels

- **OSF:** Local loader maps folders `AD` / `Healthy` → AD/HC. That is stronger than TUH “abnormal,” but the on-disk metadata does **not** document DSM/NIA-AA criteria, MMSE, or specialist confirmation the way Miltiadous’s data descriptor does. Treat as **research AD/HC partitions**, not gold-standard clinical adjudication.
- **HC n=12** is underpowered and unstable for effect-size bragging; a few outliers can move Mann–Whitney.
- **BrainLAT** has better N balance but broken spatial correspondence to the trained 10–20 encoder — a positive result there would be hard to trust; a null would be uninterpretable.

---

## 2. Dataset prepared

Path: `data/ad_labeled_external/`  
Source copy: OSF features from `data/external_osf/features/` + `README.md` provenance.

Processed EEG shapes on disk: `(N, 1, 19, 2000)` → **1× 4 s epoch** per subject after Phase-1 import.

---

## 3. Pipeline adaptation

- Reused TUH spectral calibration helpers from `src/tuh/process_tuh_eeg.py` (`TARGET_N_EPOCHS_FOR_PSD=864`, `calibrate_flat_features_for_epoch_count`).
- New runner only: `src/tuh/run_layer5b_ad_labeled.py` (no edits to `src/eeg/preprocess.py` / `feature_extraction.py`).
- Calibration: scale PSD + band powers by `1/864`; coherence/PLV unchanged.
- Scale check: BP ratio vs train **3323× → 3.85×** after 1/N (residual domain gap remains; not force-fit to 1.0).

---

## 4. Layer 5b results (OSF AD vs HC)

### Discrimination (latent mean drug-response ‖Δμ‖)

| Metric | OSF Layer 5b | TUH Layer 5 (proxy) |
|--------|--------------|---------------------|
| N | AD 80 / HC 12 | Abn 131 / Nor 69 |
| Mean ± SD (case) | 0.1669 ± 0.0023 | 0.1757 ± 0.0226 |
| Mean ± SD (control) | 0.1643 ± 0.0153 | 0.1788 ± 0.0277 |
| Mann–Whitney p | **0.00028** | 0.272 |
| Cohen’s d | **0.44** | −0.13 |
| Direction | AD > HC | Abn < Nor (wrong way) |
| Status | **PASS (with caveats)** | FAIL |

### Disease-label confound check

Twin also receives `disease_label`. Re-ran with **fixed** disease for all subjects:

| Conditioning | p | d | AD > HC? |
|--------------|---|---|----------|
| True labels | 2.8e−4 | 0.44 | Yes |
| All disease=0 | 5.1e−5 | 0.64 | Yes |
| All disease=1 | 6.8e−5 | 0.34 | Yes |

→ Separation is **not** an artifact of feeding AD/HC into the disease channel alone; EEG features drive it.

### Feature overlap vs training

| Check | Value |
|-------|-------|
| Full-2185 PCA in train 95% radius | **8.7% (poor)** — connectivity-dominated domain shift |
| Spectral-only overlap after calibration | **100%** |
| Cross-dataset drug-effect directions | **10/10**, r ≈ **0.85** |

---

## 5. Comparison vs TUH Layer 5

| Question | Answer |
|----------|--------|
| Did real AD/HC labels beat the abnormal/normal proxy? | **Yes** — null (p=0.27) → significant AD>HC (p≈3e−4). |
| Is this enough to declare Layer 5 overall PASS and drop TUH? | **Not automatically.** TUH remains a valid negative control on weak proxies. OSF is a **supplement (5b)**. |
| Should TUH be superseded? | **Your call.** Recommendation: keep Layer 5 = TUH FAIL (proxy), add Layer 5b = OSF PASS_WITH_CAVEATS. |

---

## 6. Layers 1–4 / TUH untouched

- Shared EEG modules: **not modified**
- TUH features / mining / validation: **not overwritten** (backup retained)
- v2 Layer 1–4 fingerprints unchanged; v3 gains `layer5b_ad_labeled_external` without flipping `layer5.status`

---

## 7. Honest verdict

**PASS with caveats — not a clean slam dunk.**

What holds:
- With folder-level AD vs Healthy labels and correct 10–20 montage, the twin’s latent drug-response magnitude is **larger in AD than HC**, significant even when disease conditioning is fixed.
- Cross-dataset pharmacodynamic directions remain coherent (10/10).

What does **not** get dressed up:
- **HC n=12** is fragile; absolute mean gap is tiny (~0.0025) with oddly low AD variance — significance is real under Mann–Whitney but **effect magnitude is small**.
- OSF label documentation locally is weaker than Miltiadous’s published clinical descriptor.
- Full feature-space overlap with training is still **poor** (connectivity domain shift).
- This does **not** prove clinical utility or that “abnormal EEG” should have worked; it suggests the **proxy label**, not only the twin, was the Layer 5 problem.

If this were still a null (it is not), the interpretation would be: drug-response latents may need behavioral/EEG change targets, not diagnosis alone. Here diagnosis helps, but the signal is mild and the control arm is thin — next step should be a larger clinically adjudicated HC/AD set (e.g. CAUEEG after ethics approval) before claiming robust external AD validation.

### Artifacts

- `data/ad_labeled_external/` (features + calibrated + README)
- `data/ad_labeled_external/validation/layer5b_results.json`
- `data/ad_labeled_external/validation/fig_layer5b_*.png`
- `src/tuh/run_layer5b_ad_labeled.py`
- `models/validation/complete_validation_report_v3.json` → added `layer5b_ad_labeled_external`
