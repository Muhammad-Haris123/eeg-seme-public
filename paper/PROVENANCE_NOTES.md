# Provenance notes (documentation only)

Date: 2026-08-12

This file records artifact facts that must not be silently “fixed” by regenerating data or changing locked preprocessing defaults. Statements below distinguish stored-file facts from inference.

---

## TUH crop_seconds: stored processing report vs current script default

**Stored-file facts**

- The stored TUH processing report associated with the locked TUH features (`data/tuh_features/processing_report.json`) records `crop_seconds=60.0` and `apply_ica=false`.
- Per-recording `load_meta.cropped_to_s` in that report is `60.0` (example: `aaaaabdo_s003_t000.edf`).
- The current repository script `src/tuh/process_tuh_eeg.py` specifies `CROP_SECONDS = 320.0`.

**Inference / implication (not a claim about unobserved reprocessing)**

- Therefore the current script default does not document the preprocessing configuration recorded for the stored locked TUH features.
- The locked TUH results use the stored features and must not be regenerated under the current default without an explicit reproducibility decision.
- The manuscript Methods statement of 320 s requires correction or clarification in a future revision pass.

**This session did not**

- reprocess TUH,
- regenerate TUH features,
- change `CROP_SECONDS`,
- or edit manuscript Methods text for this discrepancy.
