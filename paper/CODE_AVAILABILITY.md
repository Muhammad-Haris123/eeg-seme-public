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

| Role | Path |
|------|------|
| Constrained external twin | `models/checkpoints_constrained/checkpoint_constrained.pt` |
| Matched unconstrained twin | `models/checkpoints_unconstrained/checkpoint_unconstrained.pt` |
| Selection rule | `models/checkpoints_unconstrained/CHECKPOINT_SELECTION.md` |

## Non-locked Phase C secondary artifacts

| Role | Path |
|------|------|
| OOF probe scores | `models/validation/oof_probe_scores_phase1.npz` |
| DeLong | `models/validation/delong_oof_probe_results.json` |
| ChemBERTa vs one-hot | `models/validation/chemberta_onehot_ablation/` |

## Public release checklist (at acceptance)

1. Push a tagged commit; paste URL + hash into this file, `CITATION.cff`, and the Data/Code availability statements.
2. Confirm regenerator succeeds on a clean machine with `eeg_twin` + frozen validation JSON.
3. Do not redistribute raw TUH/CAUEEG beyond provider terms.

## Public release status

Code is prepared for public release packaging (`LICENSE`, `CITATION.cff`, regenerator). **Public URL and commit hash: to be inserted at acceptance.** Until then, the local repository and Overleaf package are the reproducibility record.


## Public repository

https://github.com/Muhammad-Haris123/eeg-seme-public


## Public repository

- URL: https://github.com/Muhammad-Haris123/eeg-seme-public
- Commit (initial push): `41cd6fff0f5962a5380f883cdfbdf233f05dfdb1`
- Zenodo DOI for `.pt` checkpoints: TBD
