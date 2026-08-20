"""
Load TUH EDF files, map channels, preprocess, and extract 2185-D features.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.eeg.feature_extraction import EEGFeatureExtractor
from src.eeg.preprocess import EEGPreprocessor
from src.tuh.mine_headers import infer_label_from_path, parse_filename_ids
from src.utils.config import FILTER_CONFIG, STANDARD_10_20_CHANNELS
from api.utils.feature_processor import flatten_features

# TUH / 10-10 → legacy 10-20 names used by the training pipeline
CHANNEL_ALIASES = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "F3": "F3",
    "F4": "F4",
    "C3": "C3",
    "C4": "C4",
    "P3": "P3",
    "P4": "P4",
    "O1": "O1",
    "O2": "O2",
    "F7": "F7",
    "F8": "F8",
    "T3": "T3",
    "T4": "T4",
    "T5": "T5",
    "T6": "T6",
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    # modern 10-10 ↔ legacy
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

TUH_TO_STANDARD = CHANNEL_ALIASES  # public alias expected by callers

TARGET_SFREQ = 500.0  # training PSD layout (n_per_seg=256 → 20 bins)
MIN_MAPPED_CHANNELS = 15
# Match training epoch depth: 4s windows @ 90% overlap → ~2.5 epochs/s.
# AD train ≈758 epochs, HC ≈1000; cohort-weighted ≈864.
CROP_SECONDS = 320.0
MAX_EPOCHS_FOR_FEATURES = 864
# Spectral features use time-average of z-scored epochs before Welch; power ∝ 1/N.
# Scale PSD/band_powers to the training reference epoch count.
TARGET_N_EPOCHS_FOR_PSD = 864
APPLY_EPOCH_POWER_CALIBRATION = True


def _normalize_ch_token(name: str) -> str:
    n = name.upper().strip()
    n = n.replace("EEG", " ")
    n = n.replace("-REF", " ").replace("-LE", " ").replace("-AVG", " ")
    n = n.replace("REF", " ").replace("LE", " ")
    n = re.sub(r"[^A-Z0-9]", "", n)
    return n


def map_tuh_channels(raw) -> Tuple[object, List[str], Dict]:
    """
    Pick and rename channels to STANDARD_10_20_CHANNELS order.

    Accepts both 01_tcp_ar (EEG FP1-REF) and 02_tcp_le (EEG FP1-LE),
    plus T7/T8/P7/P8 remaps to T3/T4/T5/T6.
    """
    standard = list(STANDARD_10_20_CHANNELS)
    wanted = {c: i for i, c in enumerate(standard)}

    chosen_raw = {}  # standard_name -> original channel name
    for ch in raw.ch_names:
        token = _normalize_ch_token(ch)
        if token in CHANNEL_ALIASES:
            std = CHANNEL_ALIASES[token]
            # Prefer first match; keep AR/LE whichever appears first
            if std not in chosen_raw:
                chosen_raw[std] = ch

    mapped = [c for c in standard if c in chosen_raw]
    if len(mapped) < MIN_MAPPED_CHANNELS:
        raise ValueError(
            f"Only {len(mapped)} mappable channels (<{MIN_MAPPED_CHANNELS}). "
            f"Found={[ _normalize_ch_token(c) for c in raw.ch_names[:25] ]}"
        )

    picks = [chosen_raw[c] for c in mapped]
    raw_p = raw.copy().pick(picks)
    rename = {chosen_raw[c]: c for c in mapped}
    raw_p.rename_channels(rename)

    # If fewer than 19, we cannot fill missing channels with zeros safely for connectivity.
    if len(mapped) < 19:
        raise ValueError(f"Need all 19 standard channels for 2185-D features, got {len(mapped)}")

    # Enforce exact order
    raw_p.reorder_channels(standard)
    info = {
        "n_mapped": len(mapped),
        "mapping": {std: chosen_raw[std] for std in mapped},
        "channels": standard,
    }
    return raw_p, standard, info


def load_tuh_edf(
    edf_path: str,
    preload: bool = True,
    crop_seconds: float = CROP_SECONDS,
    target_sfreq: float = TARGET_SFREQ,
    notch_freq: float = 60.0,
) -> Tuple[object, Dict]:
    """
    Load a TUH EDF, map channels, crop, resample to training rate, set notch for US mains.
    """
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=False)
    meta = {
        "edf_path": edf_path,
        "original_sfreq": float(raw.info["sfreq"]),
        "original_n_channels": len(raw.ch_names),
        "original_duration_s": float(raw.times[-1]) if len(raw.times) else 0.0,
    }

    raw, ch_names, ch_info = map_tuh_channels(raw)
    meta["channel_info"] = ch_info

    # Crop long clinical recordings for tractable ICA/features
    if crop_seconds and meta["original_duration_s"] > crop_seconds:
        raw.crop(tmin=0.0, tmax=crop_seconds)
        meta["cropped_to_s"] = crop_seconds
    else:
        meta["cropped_to_s"] = meta["original_duration_s"]

    if meta["cropped_to_s"] < 60.0:
        raise ValueError(f"Recording too short after crop: {meta['cropped_to_s']:.1f}s")

    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-3:
        raw.resample(target_sfreq, npad="auto", verbose=False)
        meta["resampled_to"] = target_sfreq
    else:
        meta["resampled_to"] = float(raw.info["sfreq"])

    # Attach notch preference for downstream preprocessor (US / Philadelphia = 60 Hz)
    meta["notch_freq"] = notch_freq
    return raw, meta


def _make_tuh_preprocessor(notch_freq: float = 60.0) -> EEGPreprocessor:
    filt = FILTER_CONFIG.copy()
    filt["notch_freq"] = notch_freq
    return EEGPreprocessor(filter_config=filt)


def unflatten_features(flat: np.ndarray) -> Dict[str, np.ndarray]:
    """Inverse of flatten_features / prepare_target_features ordering."""
    flat = np.asarray(flat, dtype=np.float64).reshape(-1)
    if flat.size != 2185:
        raise ValueError(f"Expected 2185 features, got {flat.size}")
    n_ch, n_psd, n_bands = 19, 20, 5
    triu = np.triu_indices(n_ch, k=1)
    i = 0
    psd = flat[i : i + n_ch * n_psd].reshape(n_ch, n_psd)
    i += n_ch * n_psd
    band_powers = flat[i : i + n_ch * n_bands].reshape(n_ch, n_bands)
    i += n_ch * n_bands
    coherence = np.zeros((n_bands, n_ch, n_ch), dtype=np.float64)
    plv = np.zeros((n_bands, n_ch, n_ch), dtype=np.float64)
    for b in range(n_bands):
        coh_vals = flat[i : i + 171]
        i += 171
        plv_vals = flat[i : i + 171]
        i += 171
        coherence[b][triu] = coh_vals
        coherence[b][(triu[1], triu[0])] = coh_vals
        plv[b][triu] = plv_vals
        plv[b][(triu[1], triu[0])] = plv_vals
        np.fill_diagonal(coherence[b], 1.0)
        np.fill_diagonal(plv[b], 1.0)
    return {
        "psd": psd.astype(np.float32),
        "band_powers": band_powers.astype(np.float32),
        "coherence": coherence.astype(np.float32),
        "plv": plv.astype(np.float32),
    }


def calibrate_spectral_power_for_epoch_count(
    psd: np.ndarray,
    band_powers: np.ndarray,
    n_epochs_used: int,
    target_n_epochs: int = TARGET_N_EPOCHS_FOR_PSD,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Correct absolute spectral scale after time-averaging z-scored epochs.

    Phase-1 extract_spectral_features(per_epoch=False) averages epochs in time
    before Welch. For approximately independent zero-mean epochs, variance
    (and thus band power) scales as 1/N. Training used ~758–1000 epochs; TUH
    batch runs that use fewer epochs must rescale PSD/band_powers by N/N_ref.
    Coherence/PLV are unchanged (already normalized measures).
    """
    n = max(int(n_epochs_used), 1)
    target = max(int(target_n_epochs), 1)
    scale = float(n) / float(target)
    return psd * scale, band_powers * scale, scale


def calibrate_flat_features_for_epoch_count(
    flat: np.ndarray,
    n_epochs_used: int,
    target_n_epochs: int = TARGET_N_EPOCHS_FOR_PSD,
) -> Tuple[np.ndarray, float]:
    """Scale PSD [0:380] and band_powers [380:475] blocks of a 2185-D vector."""
    flat = np.asarray(flat, dtype=np.float64).copy()
    n = max(int(n_epochs_used), 1)
    target = max(int(target_n_epochs), 1)
    scale = float(n) / float(target)
    flat[0:380] *= scale
    flat[380:475] *= scale
    return flat.astype(np.float32), scale


def extract_tuh_features(
    raw,
    apply_ica: bool = False,
    max_epochs: int = MAX_EPOCHS_FOR_FEATURES,
    notch_freq: float = 60.0,
    apply_epoch_power_calibration: bool = APPLY_EPOCH_POWER_CALIBRATION,
    target_n_epochs: int = TARGET_N_EPOCHS_FOR_PSD,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Preprocess a mapped Raw and extract a (2185,) feature vector.

    Reuses EEGFeatureExtractor (spectral) and ConnectivityAnalyzer (coherence/PLV).
    Epochs are averaged before connectivity so batch processing stays tractable
    while still calling the same connectivity primitives as Phase 1.

    Spectral absolute scale is calibrated to TARGET_N_EPOCHS_FOR_PSD so TUH
    features match the training cohort's epoch-averaging depth (TUH-only;
    does not modify shared preprocess/feature_extraction modules).
    """
    from src.eeg.connectivity import ConnectivityAnalyzer
    from src.utils.config import FREQUENCY_BANDS

    preprocessor = _make_tuh_preprocessor(notch_freq=notch_freq)
    data, prep_meta = preprocessor.preprocess_raw(raw, apply_ica=apply_ica, ica_exclude=None)
    # data: (epochs, channels, time)
    if data.shape[0] > max_epochs:
        # Uniform temporal subsample
        idx = np.linspace(0, data.shape[0] - 1, max_epochs).astype(int)
        data = data[idx]
        prep_meta["n_epochs_used"] = int(max_epochs)
    else:
        prep_meta["n_epochs_used"] = int(data.shape[0])

    if data.shape[1] != 19:
        raise ValueError(f"Expected 19 channels after preprocess, got {data.shape[1]}")

    sfreq = float(raw.info["sfreq"])

    extractor = EEGFeatureExtractor()
    batch = data[np.newaxis, ...]  # (1, epochs, ch, time)
    spectral = extractor.extract_spectral_features(batch, sfreq=sfreq, per_epoch=False)
    psd = spectral["psd"][0]
    band_powers = spectral["band_powers"][0]

    epoch_scale = 1.0
    if apply_epoch_power_calibration:
        psd, band_powers, epoch_scale = calibrate_spectral_power_for_epoch_count(
            psd, band_powers, prep_meta["n_epochs_used"], target_n_epochs=target_n_epochs
        )

    # Average epochs → (channels, time), then reuse ConnectivityAnalyzer multiband APIs
    avg_data = np.mean(data, axis=0)
    analyzer = ConnectivityAnalyzer()
    coh_dict = analyzer.compute_multiband_connectivity(avg_data, sfreq, method="coherence")
    plv_dict = analyzer.compute_multiband_connectivity(avg_data, sfreq, method="plv")
    band_order = list(FREQUENCY_BANDS.keys())
    coherence = np.stack([coh_dict[b] for b in band_order], axis=0)
    plv = np.stack([plv_dict[b] for b in band_order], axis=0)

    if psd.shape != (19, 20):
        raise ValueError(f"PSD shape {psd.shape} != (19, 20); check sampling rate")
    if band_powers.shape != (19, 5):
        raise ValueError(f"Band powers shape {band_powers.shape} != (19, 5)")
    if coherence.shape != (5, 19, 19) or plv.shape != (5, 19, 19):
        raise ValueError(f"Connectivity shapes unexpected: {coherence.shape}, {plv.shape}")

    flat = flatten_features(psd, band_powers, coherence, plv).astype(np.float32)
    if flat.shape != (2185,):
        raise ValueError(f"Flat feature dim {flat.shape} != (2185,)")

    structured = {
        "psd": psd.astype(np.float32),
        "band_powers": band_powers.astype(np.float32),
        "coherence": coherence.astype(np.float32),
        "plv": plv.astype(np.float32),
    }
    meta = {
        "preprocess": prep_meta,
        "sfreq": sfreq,
        "feature_dim": 2185,
        "apply_ica": apply_ica,
        "connectivity_mode": "epoch_average_then_multiband",
        "epoch_power_calibration": {
            "applied": bool(apply_epoch_power_calibration),
            "n_epochs_used": int(prep_meta["n_epochs_used"]),
            "target_n_epochs": int(target_n_epochs),
            "scale": float(epoch_scale),
        },
    }
    return flat, structured, meta


def recalibrate_existing_tuh_features(
    feature_dir: str,
    processing_report_path: Optional[str] = None,
    target_n_epochs: int = TARGET_N_EPOCHS_FOR_PSD,
    dry_run: bool = False,
) -> Dict:
    """
    Post-hoc apply epoch-count spectral calibration to already-extracted TUH features.

    Only scales PSD and band_power blocks; coherence/PLV unchanged.
    Writes calibrated arrays in-place (backup first) and updates processing_report.
    """
    feature_dir = str(feature_dir)
    if processing_report_path is None:
        processing_report_path = os.path.join(feature_dir, "processing_report.json")
    with open(processing_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    calibrated = []
    for rec in report.get("results", []):
        if not rec.get("ok"):
            continue
        n_epochs = int(rec.get("n_epochs_used") or MAX_EPOCHS_FOR_FEATURES)
        # Avoid double-calibration if already applied at target depth
        prev = rec.get("epoch_power_calibration") or {}
        if prev.get("applied") and abs(float(prev.get("target_n_epochs", 0)) - target_n_epochs) < 1e-6:
            # If already calibrated to this target, skip
            if abs(float(prev.get("scale", 1.0)) - (n_epochs / float(target_n_epochs))) < 1e-9:
                calibrated.append({**rec, "skipped": "already_calibrated"})
                continue
            # If calibrated to a different scheme, restore would need backup; scale relative
            # Here we assume features on disk are UNCALIBRATED originals from the first run.
            pass

        feat_path = rec["feature_path"]
        struct_path = rec.get("structured_path")
        flat = np.load(feat_path)
        flat_c, scale = calibrate_flat_features_for_epoch_count(
            flat, n_epochs, target_n_epochs=target_n_epochs
        )
        if not dry_run:
            np.save(feat_path, flat_c)
            if struct_path and os.path.exists(struct_path):
                z = np.load(struct_path)
                psd = z["psd"] * scale
                bp = z["band_powers"] * scale
                np.savez_compressed(
                    struct_path,
                    psd=psd.astype(np.float32),
                    band_powers=bp.astype(np.float32),
                    coherence=z["coherence"],
                    plv=z["plv"],
                )
        rec["epoch_power_calibration"] = {
            "applied": True,
            "n_epochs_used": n_epochs,
            "target_n_epochs": int(target_n_epochs),
            "scale": float(scale),
            "method": "posthoc_1_over_N_spectral",
        }
        calibrated.append(
            {
                "patient_id": rec.get("patient_id"),
                "session_id": rec.get("session_id"),
                "n_epochs_used": n_epochs,
                "scale": float(scale),
                "feature_path": feat_path,
            }
        )

    report["epoch_power_calibration"] = {
        "applied": True,
        "target_n_epochs": int(target_n_epochs),
        "method": "posthoc_1_over_N_spectral",
        "n_calibrated": len(calibrated),
    }
    if not dry_run:
        with open(processing_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
    return report["epoch_power_calibration"]


def process_single_edf(
    edf_path: str,
    output_dir: str,
    apply_ica: bool = False,
) -> Dict:
    """Load → features → save .npy (+ structured .npz)."""
    filename = os.path.basename(edf_path)
    patient_id, session_id = parse_filename_ids(filename)
    label = infer_label_from_path(edf_path)
    out_stem = f"{patient_id}_{session_id}"
    feat_path = os.path.join(output_dir, f"{out_stem}_features.npy")
    struct_path = os.path.join(output_dir, f"{out_stem}_structured.npz")

    raw, load_meta = load_tuh_edf(edf_path)
    flat, structured, feat_meta = extract_tuh_features(
        raw, apply_ica=apply_ica, notch_freq=load_meta.get("notch_freq", 60.0)
    )
    os.makedirs(output_dir, exist_ok=True)
    np.save(feat_path, flat)
    np.savez_compressed(struct_path, **structured)

    return {
        "ok": True,
        "filename": filename,
        "patient_id": patient_id,
        "session_id": session_id,
        "label": label,
        "feature_path": feat_path,
        "structured_path": struct_path,
        "feature_dim": int(flat.shape[0]),
        "load_meta": {k: v for k, v in load_meta.items() if k != "channel_info"},
        "n_epochs_used": feat_meta.get("preprocess", {}).get("n_epochs_used"),
        "epoch_power_calibration": feat_meta.get("epoch_power_calibration"),
    }


def process_tuh_directory(
    edfs_dir: str,
    output_dir: str,
    max_files: int = 200,
    apply_ica: bool = False,
) -> Dict:
    """Process all EDFs under edfs_dir (recursive)."""
    from tqdm import tqdm

    edf_paths = []
    for root, _, files in os.walk(edfs_dir):
        for f in files:
            if f.lower().endswith(".edf") and not f.startswith("."):
                edf_paths.append(os.path.join(root, f))
    edf_paths = sorted(edf_paths)[:max_files]

    os.makedirs(output_dir, exist_ok=True)
    results = []
    n_ok = 0
    for path in tqdm(edf_paths, desc="Extracting TUH features"):
        try:
            rec = process_single_edf(path, output_dir, apply_ica=apply_ica)
            results.append(rec)
            n_ok += 1
        except Exception as exc:
            filename = os.path.basename(path)
            patient_id, session_id = parse_filename_ids(filename)
            results.append(
                {
                    "ok": False,
                    "filename": filename,
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "label": infer_label_from_path(path),
                    "edf_path": path,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )

    report = {
        "edfs_dir": edfs_dir,
        "output_dir": output_dir,
        "n_attempted": len(edf_paths),
        "n_success": n_ok,
        "n_failed": len(edf_paths) - n_ok,
        "success_rate": float(n_ok / len(edf_paths)) if edf_paths else 0.0,
        "apply_ica": apply_ica,
        "target_sfreq": TARGET_SFREQ,
        "notch_freq_hz": 60.0,
        "crop_seconds": CROP_SECONDS,
        "results": results,
    }
    report_path = os.path.join(output_dir, "processing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report
