"""Measure architecture size, checkpoint size, and dummy-batch inference time.

Does not retrain. Does not change scientific validation JSON.
Saves: models/validation/computational_requirements.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.models.config import MODEL_VARIANTS
from src.models.train_phase2_upgraded import UpgradedDigitalTwinModel

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "models" / "checkpoints_constrained" / "checkpoint_constrained.pt"
OUT = ROOT / "models" / "validation" / "computational_requirements.json"


def _pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def main() -> None:
    cfg = MODEL_VARIANTS["constrained_mlp"]
    model = UpgradedDigitalTwinModel(config=cfg)
    n_params = int(sum(p.numel() for p in model.parameters()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if CKPT.exists():
        state = torch.load(CKPT, map_location=device, weights_only=False)
        sd = state.get("model_state_dict", state)
        model.load_state_dict(sd, strict=False)

    bsz = 16
    psd = torch.randn(bsz, 19, 20, device=device)
    bp = torch.randn(bsz, 19, 5, device=device)
    coh = torch.randn(bsz, 5, 19, 19, device=device)
    plv = torch.randn(bsz, 5, 19, 19, device=device)
    drug = torch.randn(bsz, 384, device=device)
    disease = torch.zeros(bsz, device=device)
    drug_id = torch.zeros(bsz, dtype=torch.long, device=device)

    def _fwd():
        with torch.no_grad():
            return model(psd, bp, coh, plv, drug, disease, return_latent=False, drug_id=drug_id)

    for _ in range(5):
        _fwd()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    n_rep = 50
    t0 = time.perf_counter()
    for _ in range(n_rep):
        _fwd()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_batch = 1000.0 * elapsed / n_rep
    ms_per_subject = ms_per_batch / bsz
    subjects_per_s = 1000.0 / ms_per_subject

    peak_mb = None
    if device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Dummy-batch inference on the locked constrained_mlp architecture. Not a clinical latency claim. Training wall-clock for the locked fold-0 run is not stored in constrained_evaluation_report.json.",
        "variant": "constrained_mlp",
        "checkpoint": str(CKPT) if CKPT.exists() else None,
        "checkpoint_bytes": int(CKPT.stat().st_size) if CKPT.exists() else None,
        "parameter_count": n_params,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory)
        if device.type == "cuda"
        else None,
        "dummy_batch_size": bsz,
        "warmup_forwards": 5,
        "timed_forwards": n_rep,
        "ms_per_batch": ms_per_batch,
        "ms_per_subject": ms_per_subject,
        "subjects_per_second": subjects_per_s,
        "peak_gpu_memory_allocated_mib": peak_mb,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "sklearn": _pkg_version("sklearn"),
        "mne": _pkg_version("mne"),
        "transformers": _pkg_version("transformers"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
