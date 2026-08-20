# Table 1. External assessment datasets

**Caption (place ABOVE the table, Elsevier):** External EEG datasets used for escalating external assessment of the constrained twin. Sampling rates are native acquisition followed by the common 500 Hz analysis rate. Roles follow the Results Part B narrative sequence.

| Dataset | N (total) | N by group | Labels | Sampling rate | Montage | Role in study |
|---------|-----------|------------|--------|---------------|---------|---------------|
| TUH | 200 | Abn 131 / Nor 69 | Abnormal vs normal (proxy) | Mixed EDF → 500 Hz | 19ch 10–20 | First external test; proxy labels |
| OSF | 92 | AD 80 / HC 12 | Folder AD vs Healthy | 256 → 500 Hz | 19ch 10–20 | AD/HC labels; small HC |
| P-ADIC | 145 | AD 49 / HC 96 | NIA-AA AD vs HC | 200/500 → 500 Hz | 19ch 10–20 (Nihon remap) | Larger AD/HC; confirms null |
| CAUEEG | 1122 | Nor 436 / MCI 395 / Dem 291 | Clinical Normal/MCI/Dementia | 200 → 500 Hz | 19ch AVG 10–20 | Large-N clinical; encoding analysis |

## Source notes

- TUH: `models/validation/layer5_domain_shift_diagnosis.md`; `complete_validation_report_v3.json` → `layer5`
- OSF: `models/validation/layer5b_ad_labeled_external_diagnosis.md`; `layer5b_ad_labeled_external`
- P-ADIC: `models/validation/layer5d_padic_external_diagnosis.md`; `layer5d_padic_external`
- CAUEEG: `models/validation/layer5e_caueeg_external_diagnosis.md`; `layer5e_caueeg_external`
