# Code availability (CMPB / SEME manuscript)

Author and institution fields are omitted from this draft until submission metadata are finalized.

## Protocol product

**SEME** = Signed concordance, Effect-vector fidelity, Magnitude transfer, Encoding retention  
Identity lock: `paper/SEME_PROTOCOL.md`  
Cover letter draft: `paper/CMPB_COVER_LETTER.md`

## Environment

- Python 3.10 virtualenv: `eeg_twin/` (measured: Python 3.10.0)
- Measured package versions: PyTorch 2.10.0+cu128, NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.7.2, MNE 1.11.0, Transformers 5.0.0 (`models/validation/computational_requirements.json`)
- Dependency file: `requirements.txt`
- License (code): `LICENSE` (MIT; dataset terms remain upstream)
- Citation file: `CITATION.cff`

```powershell
.\eeg_twin\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
.\eeg_twin\Scripts\python.exe -c "import sys; print(sys.executable)"
```

## One-command regenerate (Path A)

```powershell
.\eeg_twin\Scripts\python.exe src\paper\regenerate_cmpb_artifacts.py
.\eeg_twin\Scripts\python.exe src\paper\regenerate_cmpb_artifacts.py --pack-zip
```

This regenerates DeLong outputs, selected tables/figures from frozen JSON, and optionally the Overleaf zip. It does **not** overwrite locked CVAE checkpoints or reprocess TUH.

## Frozen checkpoints (locked; do not overwrite)

Locked `.pt` weights are **not** stored in this GitHub mirror (size and clinical-data hygiene). Download from Zenodo:

| Role | Filename on Zenodo |
|------|--------------------|
| Constrained external twin | `checkpoint_constrained.pt` |
| Matched unconstrained twin | `checkpoint_unconstrained.pt` |
| Selection rule | `CHECKPOINT_SELECTION.md` |

**Zenodo (v0.1.0):** https://doi.org/10.5281/zenodo.22028681  
**Concept DOI (all versions):** https://doi.org/10.5281/zenodo.22028680  
Citation: Haris, M. (2026). SEME locked EEG–drug CVAE checkpoints (constrained and unconstrained) (Version v0.1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.22028681

## Non-locked Phase C secondary artifacts

| Role | Path |
|------|------|
| OOF probe scores | `models/validation/oof_probe_scores_phase1.npz` |
| DeLong | `models/validation/delong_oof_probe_results.json` |
| ChemBERTa vs one-hot | `models/validation/chemberta_onehot_ablation/` |

## Public repository

- **Code:** https://github.com/Muhammad-Haris123/eeg-seme-public
- **Commit:** `31e424933ef84adbb957d78b70b964721aa5a264` (short: `31e4249`)
- **Weights:** https://doi.org/10.5281/zenodo.22028681
- Do not redistribute raw TUH/CAUEEG beyond provider terms.
