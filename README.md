# SEME — Decoupled Evaluation Protocol for Literature-Constrained EEG–Drug Generators

Public code and frozen validation artifacts for the CMPB manuscript.

## What this repo is
- **SEME** protocol (Signed concordance, Effect-vector fidelity, Magnitude transfer, Encoding retention)
- Paper sources under `paper/`
- Regenerable tables/figures from frozen JSON under `models/validation/`

## What this repo is NOT
- Raw EEG (TUH/CAUEEG/OpenNeuro/OSF/P-ADIC) — obtain from providers
- Locked `.pt` checkpoints — https://doi.org/10.5281/zenodo.22028681
- Training virtualenv (`eeg_twin`) — create locally from `requirements.txt`

## Quick start
```powershell
python -m venv eeg_twin
.\eeg_twin\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
.\eeg_twin\Scripts\python.exe src\paper\regenerate_cmpb_artifacts.py --skip-figures
```

See `paper/CODE_AVAILABILITY.md` and `paper/SEME_PROTOCOL.md`.
