# SEME v0.1.0

First public release for the CMPB manuscript package.

## Contents
- SEME evaluation protocol sources under `paper/`
- Frozen validation JSON under `models/validation/`
- One-command scorecard: `src/seme/demo_scorecard.py`
- Artifact regenerator: `src/paper/regenerate_cmpb_artifacts.py`

## Not included
- Raw clinical EEG (obtain from providers)
- Locked `.pt` checkpoints (Zenodo)

## Locked weights
https://doi.org/10.5281/zenodo.22028681

## Quick demo
```powershell
python -m venv eeg_twin
.\eeg_twin\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
.\eeg_twin\Scripts\python.exe src\seme\demo_scorecard.py
```
