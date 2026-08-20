"""
Load P-ADIC Dryad MATLAB v7.3 (HDF5) recordings into the shared EEG pipeline.

Montage note (IMPORTANT):
  Dryad mats do NOT embed channel names. We use the Nihon-style 19-channel order
  documented by independent P-ADIC users (eeg-slowing-transportability) and remap
  by name into STANDARD_10_20_CHANNELS. This is an explicit assumption — not silent.

Does not modify shared src/eeg/* modules.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np

from src.tuh.process_tuh_eeg import (
    CROP_SECONDS,
    MAX_EPOCHS_FOR_FEATURES,
    TARGET_N_EPOCHS_FOR_PSD,
    TARGET_SFREQ,
    extract_tuh_features,
)
from src.utils.config import STANDARD_10_20_CHANNELS

# Assumed on-disk channel axis order for P-ADIC (time, 19) arrays.
# Source: noronhareuben1/eeg-slowing-transportability transportability/eeg_features.py
# Matches classic Nihon Kohden 10–20 layout with legacy T3/T4/T5/T6 names.
PADIC_CHANNELS: Tuple[str, ...] = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
)

# Israel / EU mains
PADIC_NOTCH_HZ = 50.0

GROUP_FILE = {
    "AD": ("alz_c1_new.mat", "alz_r"),
    "HC": ("controls_c1_new.mat", "controls_r"),
}


@dataclass
class PadicRecording:
    group: str
    subject_col: int
    recording_row: int
    recording_id: str
    sfreq: float
    age: Optional[float]
    n_samples: int
    duration_s: float
    shape: Tuple[int, ...]


def montage_report() -> Dict:
    train = list(STANDARD_10_20_CHANNELS)
    padic = list(PADIC_CHANNELS)
    same_names = set(train) == set(padic)
    order_match = train == padic
    remap = [padic.index(ch) for ch in train] if same_names else []
    return {
        "padic_assumed_order": padic,
        "train_order": train,
        "same_channel_names": same_names,
        "order_identical": order_match,
        "remap_padic_index_to_train": remap,
        "embedded_channel_names_in_mat": False,
        "assumption_source": (
            "Independent open loader for this Dryad release "
            "(eeg-slowing-transportability CHANNELS tuple); "
            "author notes confirm 19-electrode 10–20 but do not list order."
        ),
        "mismatch_flag": (not order_match) and same_names,
    }


def _decode_scalar(handle: h5py.File, ref) -> Optional[object]:
    if ref is None:
        return None
    try:
        obj = handle[ref]
    except Exception:
        return None
    arr = np.asarray(obj)
    if arr.dtype == object:
        for child in arr.flat:
            v = _decode_scalar(handle, child)
            if v is not None and v != "":
                return v
        return None
    if arr.dtype.kind in "SUa":
        return "".join(
            x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in arr.flat
        ).strip()
    if arr.dtype.kind in "ui" and arr.size > 1:
        chars = "".join(chr(int(v)) for v in arr.flat if 32 <= int(v) <= 126)
        return chars.strip() or None
    if arr.size == 0:
        return None
    v = arr.reshape(-1)[0]
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def _resolve_recording(handle: h5py.File, ref):
    if ref is None:
        return None
    try:
        obj = handle[ref]
    except Exception:
        return None
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is not None and dtype is not None and getattr(dtype, "kind", "O") != "O":
        if len(shape) == 2 and 19 in shape and max(shape) > 1000:
            return obj
    arr = np.asarray(obj)
    if arr.dtype == object:
        for child in arr.flat:
            ds = _resolve_recording(handle, child)
            if ds is not None:
                return ds
    return None


def iter_group_recordings(mat_path: Path, group_name: str, label: str) -> Iterator[Tuple[PadicRecording, object, h5py.File]]:
    """
    Yield (meta, h5_dataset, open_file). Caller must not close file until done with dataset.
    Prefer using load_recording_array() which manages the file.
    """
    raise NotImplementedError


def list_recordings(mat_path: Path, group_key: str, label: str) -> List[PadicRecording]:
    gname = GROUP_FILE[label][1] if label in GROUP_FILE else group_key
    out: List[PadicRecording] = []
    with h5py.File(mat_path, "r") as handle:
        if gname not in handle:
            raise KeyError(f"{gname} not in {mat_path}")
        g = handle[gname]
        refs = np.asarray(g["G"])
        for idx in np.ndindex(refs.shape):
            ds = _resolve_recording(handle, refs[idx])
            if ds is None:
                continue
            row, col = int(idx[0]), int(idx[1]) if len(idx) > 1 else 0
            if len(idx) == 1:
                row, col = 0, int(idx[0])
            sfreq = _decode_scalar(handle, np.asarray(g["g"])[idx]) if "g" in g else 500.0
            try:
                sfreq_f = float(sfreq) if sfreq is not None else 500.0
            except Exception:
                sfreq_f = 500.0
            if not np.isfinite(sfreq_f) or sfreq_f <= 0:
                sfreq_f = 500.0
            age_v = _decode_scalar(handle, np.asarray(g["age"])[idx]) if "age" in g else None
            try:
                age_f = float(np.asarray(age_v).reshape(-1)[0]) if age_v is not None else None
                if age_f is not None and not np.isfinite(age_f):
                    age_f = None
            except Exception:
                age_f = None
            n_samples = int(max(ds.shape))
            out.append(
                PadicRecording(
                    group=label,
                    subject_col=col,
                    recording_row=row,
                    recording_id=f"{label}_{col:03d}_r{row}",
                    sfreq=sfreq_f,
                    age=age_f,
                    n_samples=n_samples,
                    duration_s=float(n_samples / sfreq_f),
                    shape=tuple(int(x) for x in ds.shape),
                )
            )
    return out


def load_recording_array(mat_path: Path, label: str, recording_id: str) -> Tuple[np.ndarray, PadicRecording]:
    """
    Return data as (19, n_times) in *assumed PADIC channel order*, plus meta.
    """
    fname, gname = GROUP_FILE[label]
    assert mat_path.name == fname or mat_path.exists()
    with h5py.File(mat_path, "r") as handle:
        g = handle[gname]
        refs = np.asarray(g["G"])
        for idx in np.ndindex(refs.shape):
            row = int(idx[0]) if len(idx) > 1 else 0
            col = int(idx[1]) if len(idx) > 1 else int(idx[0])
            rid = f"{label}_{col:03d}_r{row}"
            if rid != recording_id:
                continue
            ds = _resolve_recording(handle, refs[idx])
            if ds is None:
                raise RuntimeError(f"no dataset for {recording_id}")
            sfreq = _decode_scalar(handle, np.asarray(g["g"])[idx]) if "g" in g else 500.0
            sfreq_f = float(sfreq) if sfreq is not None else 500.0
            age_v = _decode_scalar(handle, np.asarray(g["age"])[idx]) if "age" in g else None
            try:
                age_f = float(np.asarray(age_v).reshape(-1)[0]) if age_v is not None else None
            except Exception:
                age_f = None
            arr = np.asarray(ds, dtype=np.float64)
            if arr.shape[1] == 19:
                data = arr.T  # (19, T)
            elif arr.shape[0] == 19:
                data = arr
            else:
                raise ValueError(f"cannot find 19-ch axis: {arr.shape}")
            meta = PadicRecording(
                group=label,
                subject_col=col,
                recording_row=row,
                recording_id=rid,
                sfreq=sfreq_f,
                age=age_f,
                n_samples=int(data.shape[1]),
                duration_s=float(data.shape[1] / sfreq_f),
                shape=tuple(arr.shape),
            )
            return data, meta
    raise KeyError(recording_id)


def remap_to_train_order(data_padic: np.ndarray) -> np.ndarray:
    """data_padic: (19, T) in PADIC_CHANNELS order → (19, T) in STANDARD_10_20_CHANNELS."""
    if data_padic.shape[0] != 19:
        raise ValueError(data_padic.shape)
    report = montage_report()
    if not report["same_channel_names"]:
        raise RuntimeError("Channel name set mismatch — refuse silent alignment")
    idx = report["remap_padic_index_to_train"]
    return data_padic[idx, :]


def recording_to_raw(
    data_padic: np.ndarray,
    sfreq: float,
    crop_seconds: float = CROP_SECONDS,
    target_sfreq: float = TARGET_SFREQ,
):
    """Build MNE Raw in training channel order; crop + resample."""
    import mne

    mne.set_log_level("ERROR")
    data = remap_to_train_order(data_padic)
    # finite check
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    info = mne.create_info(
        ch_names=list(STANDARD_10_20_CHANNELS),
        sfreq=float(sfreq),
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    meta = {
        "original_sfreq": float(sfreq),
        "original_duration_s": float(raw.times[-1]) if len(raw.times) else 0.0,
        "montage_remap_applied": True,
        "montage_report": montage_report(),
        "notch_freq": PADIC_NOTCH_HZ,
    }
    if crop_seconds and meta["original_duration_s"] > crop_seconds:
        raw.crop(tmin=0.0, tmax=crop_seconds)
        meta["cropped_to_s"] = float(crop_seconds)
    else:
        meta["cropped_to_s"] = meta["original_duration_s"]
    if meta["cropped_to_s"] < 60.0:
        raise ValueError(f"Recording too short: {meta['cropped_to_s']:.1f}s")
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-3:
        raw.resample(target_sfreq, npad="auto", verbose=False)
        meta["resampled_to"] = float(target_sfreq)
    else:
        meta["resampled_to"] = float(raw.info["sfreq"])
    return raw, meta


def process_padic_cohort(
    padic_dir: Path,
    out_dir: Path,
    groups: Tuple[str, ...] = ("AD", "HC"),
    crop_seconds: float = CROP_SECONDS,
) -> Dict:
    """Extract calibrated 2185-D features for AD + HC recordings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    mont = montage_report()
    if mont["mismatch_flag"]:
        print("[montage] ORDER MISMATCH vs training — remapping by channel name")
        print("  PADIC:", mont["padic_assumed_order"])
        print("  TRAIN:", mont["train_order"])
        print("  remap:", mont["remap_padic_index_to_train"])
    else:
        print("[montage] order already matches training")

    results = []
    epoch_counts = {"AD": [], "HC": []}

    for label in groups:
        fname, _ = GROUP_FILE[label]
        mat_path = padic_dir / fname
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        recs = list_recordings(mat_path, GROUP_FILE[label][1], label)
        print(f"[{label}] {len(recs)} recordings in {mat_path.name}")
        for i, rec in enumerate(recs):
            rec_out = asdict(rec)
            try:
                data, meta_r = load_recording_array(mat_path, label, rec.recording_id)
                raw, load_meta = recording_to_raw(
                    data, meta_r.sfreq, crop_seconds=crop_seconds
                )
                flat, structured, feat_meta = extract_tuh_features(
                    raw,
                    apply_ica=False,
                    max_epochs=MAX_EPOCHS_FOR_FEATURES,
                    notch_freq=PADIC_NOTCH_HZ,
                    apply_epoch_power_calibration=True,
                    target_n_epochs=TARGET_N_EPOCHS_FOR_PSD,
                )
                n_ep = int(feat_meta["preprocess"].get("n_epochs_used") or feat_meta.get("epoch_power_calibration", {}).get("n_epochs_used") or 0)
                # preprocess_raw stores n_epochs in prep_meta via extract_tuh_features
                n_ep = int(feat_meta.get("epoch_power_calibration", {}).get("n_epochs_used", n_ep))
                epoch_counts[label].append(n_ep)

                feat_path = feat_dir / f"{rec.recording_id}_features.npy"
                struct_path = feat_dir / f"{rec.recording_id}_structured.npz"
                np.save(feat_path, flat.astype(np.float32))
                np.savez_compressed(
                    struct_path,
                    psd=structured["psd"],
                    band_powers=structured["band_powers"],
                    coherence=structured["coherence"],
                    plv=structured["plv"],
                )
                rec_out.update(
                    {
                        "ok": True,
                        "feature_path": str(feat_path),
                        "structured_path": str(struct_path),
                        "n_epochs_used": n_ep,
                        "load_meta": load_meta,
                        "epoch_power_calibration": feat_meta.get("epoch_power_calibration"),
                        "disease_label": 1 if label == "AD" else 0,
                    }
                )
                print(
                    f"  [{i+1}/{len(recs)}] {rec.recording_id} "
                    f"sfreq={meta_r.sfreq:.0f} dur={meta_r.duration_s:.0f}s "
                    f"epochs={n_ep} scale={feat_meta.get('epoch_power_calibration',{}).get('scale')}"
                )
            except Exception as e:
                rec_out.update({"ok": False, "error": str(e), "traceback": traceback.format_exc()})
                print(f"  [{i+1}/{len(recs)}] FAIL {rec.recording_id}: {e}")
            results.append(rec_out)

    report = {
        "dataset": "P-ADIC",
        "doi": "10.5061/dryad.8gtht76pw",
        "padic_dir": str(padic_dir),
        "out_dir": str(out_dir),
        "montage": mont,
        "notch_hz": PADIC_NOTCH_HZ,
        "crop_seconds": crop_seconds,
        "target_n_epochs_for_psd": TARGET_N_EPOCHS_FOR_PSD,
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": sum(1 for r in results if not r.get("ok")),
        "epoch_counts": {
            g: {
                "n": len(v),
                "min": int(min(v)) if v else None,
                "median": float(np.median(v)) if v else None,
                "max": int(max(v)) if v else None,
                "mean": float(np.mean(v)) if v else None,
            }
            for g, v in epoch_counts.items()
        },
        "results": results,
    }
    report_path = out_dir / "processing_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[done] {report['n_ok']} ok / {report['n_fail']} fail -> {report_path}")
    return report


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--padic-dir", type=Path, default=Path(r"E:\padic_external"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "padic_external",
    )
    args = ap.parse_args()
    # pointer README
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "README_LOCATION.txt").write_text(
        f"Raw mats live at: {args.padic_dir}\nFeatures written under this folder.\n",
        encoding="utf-8",
    )
    process_padic_cohort(args.padic_dir, args.out_dir)


if __name__ == "__main__":
    main()
