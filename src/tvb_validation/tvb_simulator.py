"""
TVB (The Virtual Brain) validation simulator for Alzheimer's digital twin.

Generates synthetic EEG under known pharmacological parameters so CVAE
drug-effect directions can be compared against a computational ground truth.

Falls back to a literature-parameterized scipy synthesizer when TVB is
unavailable (common on some Windows installs).

Author: Research Team
Date: 2026
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
from scipy import signal
from tqdm import tqdm

# NumPy 2.x removed np.trapz; feature_extraction.py still calls it.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from src.eeg.connectivity import ConnectivityAnalyzer
from src.eeg.feature_extraction import EEGFeatureExtractor
from src.models.config import DATA_ROOT, PROJECT_ROOT
from src.utils.config import FREQUENCY_BANDS, PSD_CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# TVB availability
# ---------------------------------------------------------------------------
TVB_AVAILABLE = False
_tvb_lab = None
try:
    from tvb.simulator import lab as _tvb_lab  # noqa: F401

    TVB_AVAILABLE = True
except ImportError:
    TVB_AVAILABLE = False

# Canonical 10-20 channel order (matches user / API convention)
CHANNEL_NAMES_10_20: List[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]

N_CHANNELS = 19
N_BANDS = 5
PSD_DIM = 20
EEG_FLAT_SIZE = 2185
SFREQ_DEFAULT = 256.0

# Band index ranges on the 2185-D vector.
# Documentation sometimes lists consecutive channel blocks (delta=[380:399], …);
# the training flatten is (channels, bands) C-order via prepare_target_features,
# so correct band means use reshape(19, 5)[:, band_idx]. Connectivity comparison
# uses the documented coherence window [475:1330].
DELTA_SLICE = slice(380, 399)
THETA_SLICE = slice(399, 418)
ALPHA_SLICE = slice(418, 437)
BETA_SLICE = slice(437, 456)
CONNECTIVITY_SLICE = slice(475, 1330)
BAND_POWER_BLOCK = slice(380, 475)
BAND_INDEX = {"delta": 0, "theta": 1, "alpha": 2, "beta": 3, "gamma": 4}


def band_block_mean(features: np.ndarray, band: str) -> float:
    """Mean over channels for one band from flattened (n, 2185) features."""
    bp = features[:, BAND_POWER_BLOCK].reshape(features.shape[0], N_CHANNELS, N_BANDS)
    return float(np.nanmean(bp[:, :, BAND_INDEX[band]]))


def connectivity_block_mean(features: np.ndarray) -> float:
    return float(np.nanmean(features[:, CONNECTIVITY_SLICE]))

# Wilson-Cowan baseline parameters (standard TVB / Wilson-Cowan literature values)
WC_PARAMS = {
    "c_ee": 12.0,  # excitatory-excitatory coupling
    "c_ei": 4.0,  # excitatory-inhibitory
    "c_ie": 13.0,  # inhibitory-excitatory
    "c_ii": 11.0,  # inhibitory-inhibitory
    "tau_e": 10.0,  # excitatory time constant (ms)
    "tau_i": 10.0,  # inhibitory time constant (ms)
}

# Global coupling: Deco et al. (2013) J Neurosci — resting-state FC fitting
HEALTHY_COUPLING = 0.042

DONEPEZIL_DOSE = {
    # Babiloni et al. (2006) Clin Neurophysiol — AChE-I EEG effects scale with dose
    "low": {"ach_boost": 0.08, "ach_speedup": 0.05, "connectivity_boost": 0.10},
    "standard": {"ach_boost": 0.15, "ach_speedup": 0.10, "connectivity_boost": 0.20},
    "high": {"ach_boost": 0.22, "ach_speedup": 0.15, "connectivity_boost": 0.30},
}

MEMANTINE_DOSE = {
    # Winblad et al. (2007) Dement Geriatr Cogn Disord; Rive et al. (2013)
    "low": {"nmda_block": 0.10, "neuroprotect": 0.05},
    "standard": {"nmda_block": 0.20, "neuroprotect": 0.12},
}

# Spectral profiles (μV amplitudes) — Jeong (2004) Clin Neurophysiol review
FALLBACK_PROFILES = {
    "healthy_control": {"delta": 15.0, "theta": 10.0, "alpha": 25.0, "beta": 8.0},
    "alzheimer_mild": {"delta": 17.0, "theta": 12.0, "alpha": 18.0, "beta": 7.0},
    "alzheimer_moderate": {"delta": 20.0, "theta": 15.0, "alpha": 12.0, "beta": 6.0},
    "alzheimer_severe": {"delta": 24.0, "theta": 18.0, "alpha": 8.0, "beta": 5.0},
}

GROUND_TRUTH = {
    "donepezil_expected": {
        "alpha_direction": "increase",
        "alpha_effect_size": 0.25,
        "theta_direction": "decrease",
        "theta_effect_size": 0.20,
        "delta_direction": "decrease",
        "delta_effect_size": 0.15,
        "beta_direction": "increase",
        "beta_effect_size": 0.10,
        "connectivity_direction": "increase",
        "connectivity_effect_size": 0.20,
        "source": "Babiloni_et_al_2006",
    },
    "memantine_expected": {
        "alpha_direction": "increase",
        "alpha_effect_size": 0.10,
        "theta_direction": "decrease",
        "theta_effect_size": 0.15,
        "delta_direction": "decrease",
        "delta_effect_size": 0.10,
        "beta_direction": "increase",
        "beta_effect_size": 0.05,
        "connectivity_direction": "increase",
        "connectivity_effect_size": 0.15,
        "source": "Winblad_et_al_2007",
    },
}

CONDITIONS = [
    "healthy_control",
    "alzheimer_mild",
    "alzheimer_moderate",
    "alzheimer_donepezil",
    "alzheimer_memantine",
    "alzheimer_combination",
]


def _file_prefix() -> str:
    return "tvb_" if TVB_AVAILABLE else "synthetic_fallback_"


def _region_mask(labels: List[str], keywords: Tuple[str, ...]) -> np.ndarray:
    mask = np.zeros(len(labels), dtype=bool)
    for i, lab in enumerate(labels):
        low = str(lab).lower()
        if any(k.lower() in low for k in keywords):
            mask[i] = True
    return mask


def _load_tvb_connectivity():
    """Load TVB 76-region default connectivity (explicit tvb_data path)."""
    import tvb_data
    from tvb.simulator.lab import connectivity

    zip_path = (
        Path(tvb_data.__file__).resolve().parent / "connectivity" / "connectivity_76.zip"
    )
    conn = connectivity.Connectivity.from_file(str(zip_path))
    conn.configure()
    return conn


def build_alzheimer_network(severity: str = "moderate"):
    """
    Build a whole-brain Wilson-Cowan network mimicking AD pathology.

    Uses TVB's default 76-region connectivity with AD-specific modifications.

    Returns
    -------
    (simulator, connectivity) or raises ImportError if TVB unavailable.
    """
    if not TVB_AVAILABLE:
        raise ImportError("TVB is not available; use generate_synthetic_eeg_fallback().")

    from tvb.simulator.lab import (
        coupling,
        integrators,
        models,
        monitors,
        noise,
        simulator,
    )

    conn = _load_tvb_connectivity()
    labels = [str(x) for x in conn.region_labels]
    weights = np.array(conn.weights, dtype=np.float64, copy=True)

    # Normalize weights (TVB convention before coupling scale)
    w_max = weights.max()
    if w_max > 0:
        weights = weights / w_max

    coupling_strength = HEALTHY_COUPLING  # Deco et al. (2013) J Neurosci
    noise_sigma = 0.008

    hippo = _region_mask(labels, ("HC", "PHC", "hippocamp"))
    dmn = _region_mask(labels, ("CCP", "PCS", "PCM", "PCI", "CCR"))  # PCC / medial parietal
    frontal = _region_mask(labels, ("PFC", "FEF", "PMC", "M1"))

    # Stefanovski et al. (2019) Front. Comput. Neurosci. — AD coupling reduction
    if severity == "healthy" or severity is None:
        pass
    elif severity == "mild":
        coupling_strength *= 0.85
        noise_sigma = 0.01
    elif severity == "moderate":
        coupling_strength *= 0.70
        # Zimmermann et al. (2018) NeuroImage — hippocampal disconnection in AD
        weights[np.ix_(hippo, hippo)] *= 0.5
        weights[hippo, :] *= 0.5
        weights[:, hippo] *= 0.5
        # Maestú et al. (2015) NeuroImage — DMN disruption in AD
        weights[np.ix_(dmn, dmn)] *= 0.6
    elif severity == "severe":
        coupling_strength *= 0.55
        weights[np.ix_(hippo, hippo)] *= 0.3
        weights[hippo, :] *= 0.3
        weights[:, hippo] *= 0.3
        weights[np.ix_(dmn, dmn)] *= 0.4
        weights[np.ix_(frontal, frontal)] *= 0.75
        noise_sigma = 0.012
    else:
        raise ValueError(f"Unknown severity: {severity}")

    conn.weights = weights
    conn.configure()

    # Store masks for drug-effect helpers
    conn._ad_hippo_mask = hippo  # type: ignore[attr-defined]
    conn._ad_dmn_mask = dmn  # type: ignore[attr-defined]
    conn._ad_severity = severity  # type: ignore[attr-defined]

    wc = models.WilsonCowan(
        c_ee=np.array([WC_PARAMS["c_ee"]]),
        c_ei=np.array([WC_PARAMS["c_ei"]]),
        c_ie=np.array([WC_PARAMS["c_ie"]]),
        c_ii=np.array([WC_PARAMS["c_ii"]]),
        tau_e=np.array([WC_PARAMS["tau_e"]]),
        tau_i=np.array([WC_PARAMS["tau_i"]]),
    )

    coup = coupling.Linear(a=np.array([coupling_strength]))
    # dt=0.5 ms: stable for WC and ~5× faster than dt=0.1
    # TVB WC expects nsig shaped for configured state vars; scalar array broadcasts
    # (array([s, s]) becomes invalid (2,1,1) after configure on some TVB versions).
    integrator = integrators.HeunStochastic(
        dt=0.5,
        noise=noise.Additive(nsig=np.array([noise_sigma])),
    )
    # EEG-rate monitor at 256 Hz (TemporalAverage; full EEG monitor needs surface mesh)
    mon = monitors.TemporalAverage(period=1000.0 / SFREQ_DEFAULT)

    sim = simulator.Simulator(
        model=wc,
        connectivity=conn,
        coupling=coup,
        integrator=integrator,
        monitors=(mon,),
        simulation_length=1000.0,  # overwritten in run_tvb_simulation
    )
    sim.configure()
    return sim, conn


def apply_donepezil_effect(sim, dose_level: str = "standard"):
    """
    Modify simulator for Donepezil (AChE inhibitor) pharmacodynamics.

    Mechanism: ↑ACh → ↑alpha, ↓theta, improved posterior/hippocampal connectivity.
    Sources: Babiloni et al. (2006) Clin Neurophysiol; Jeong (2004) review.
    """
    if dose_level not in DONEPEZIL_DOSE:
        raise ValueError(f"Unknown donepezil dose_level: {dose_level}")
    p = DONEPEZIL_DOSE[dose_level]

    sim.model.c_ee = np.array([float(sim.model.c_ee[0]) * (1.0 + p["ach_boost"])])
    sim.model.tau_e = np.array([float(sim.model.tau_e[0]) * (1.0 - p["ach_speedup"])])

    conn = sim.connectivity
    hippo = getattr(conn, "_ad_hippo_mask", None)
    if hippo is not None and hippo.any():
        boost = 1.0 + p["connectivity_boost"]
        w = np.array(conn.weights, dtype=np.float64, copy=True)
        w[hippo, :] *= boost
        w[:, hippo] *= boost
        conn.weights = w
        conn.configure()

    # Mild DMN reconnection (posterior connectivity improvement)
    dmn = getattr(conn, "_ad_dmn_mask", None)
    if dmn is not None and dmn.any():
        w = np.array(conn.weights, dtype=np.float64, copy=True)
        w[np.ix_(dmn, dmn)] *= 1.0 + 0.5 * p["connectivity_boost"]
        conn.weights = w
        conn.configure()

    sim.configure()
    return sim


def apply_memantine_effect(sim, dose_level: str = "standard"):
    """
    Modify simulator for Memantine (NMDA antagonist) pharmacodynamics.

    Mechanism: ↓pathological glutamate excitotoxicity → ↓delta/theta;
    weaker alpha recovery than Donepezil.
    Sources: Winblad et al. (2007); Rive et al. (2013).
    """
    if dose_level not in MEMANTINE_DOSE:
        raise ValueError(f"Unknown memantine dose_level: {dose_level}")
    p = MEMANTINE_DOSE[dose_level]
    nmda = p["nmda_block"]

    sim.model.c_ee = np.array([float(sim.model.c_ee[0]) * (1.0 - nmda * 0.5)])
    sim.model.c_ie = np.array([float(sim.model.c_ie[0]) * (1.0 + nmda * 0.3)])

    conn = sim.connectivity
    hippo = getattr(conn, "_ad_hippo_mask", None)
    if hippo is not None and hippo.any():
        boost = 1.0 + p["neuroprotect"]
        w = np.array(conn.weights, dtype=np.float64, copy=True)
        w[hippo, :] *= boost
        w[:, hippo] *= boost
        conn.weights = w
        conn.configure()

    sim.configure()
    return sim


def _regions_to_10_20(region_ts: np.ndarray) -> np.ndarray:
    """
    Project N-region TVB time series to 19 approximate 10-20 channels.

    region_ts: (n_time, n_regions)
    returns: (19, n_time)
    """
    n_time, n_regions = region_ts.shape
    if n_regions == N_CHANNELS:
        return region_ts.T.astype(np.float64)

    idx = np.linspace(0, n_regions - 1, N_CHANNELS).astype(int)
    # Deduplicate while preserving order
    seen = set()
    unique_idx = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            unique_idx.append(i)
    while len(unique_idx) < N_CHANNELS:
        for i in range(n_regions):
            if i not in seen:
                seen.add(i)
                unique_idx.append(i)
            if len(unique_idx) >= N_CHANNELS:
                break
    eeg = region_ts[:, unique_idx[:N_CHANNELS]].T
    return eeg.astype(np.float64)


def run_tvb_simulation(
    sim,
    duration_ms: float = 60000,
    n_subjects: int = 20,
    noise_seed: int = 42,
) -> np.ndarray:
    """
    Run TVB simulation variants; return EEG (n_subjects, 19, n_timepoints).
    """
    if not TVB_AVAILABLE:
        raise ImportError("TVB is not available.")

    n_time = int(round(duration_ms * SFREQ_DEFAULT / 1000.0))
    out = np.zeros((n_subjects, N_CHANNELS, n_time), dtype=np.float64)

    for s_idx in tqdm(range(n_subjects), desc="TVB subjects"):
        seed = int(noise_seed + s_idx)
        try:
            sim.integrator.noise.random_stream.seed(seed)
        except Exception:
            np.random.seed(seed)

        sim.simulation_length = float(duration_ms)
        sim.configure()

        (tavg_time, tavg_data), = sim.run()
        # tavg_data: (time, state_var, regions, mode) — use excitatory activity (sv 0)
        region_ts = np.asarray(tavg_data)[:, 0, :, 0]  # (time, regions)
        eeg = _regions_to_10_20(region_ts)

        # Resample / trim to exact length
        if eeg.shape[1] != n_time:
            if eeg.shape[1] > n_time:
                eeg = eeg[:, :n_time]
            else:
                pad = np.zeros((N_CHANNELS, n_time - eeg.shape[1]))
                eeg = np.concatenate([eeg, pad], axis=1)

        # Scale to µV-like amplitudes for feature pipeline stability
        eeg = (eeg - eeg.mean(axis=1, keepdims=True))
        std = eeg.std(axis=1, keepdims=True) + 1e-12
        eeg = eeg / std * 15.0  # ~15 µV RMS

        out[s_idx] = eeg

    return out


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f (pink) noise via spectral shaping."""
    white = rng.normal(size=n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n)
    pink = pink / (pink.std() + 1e-12)
    return pink


def _electrode_covariance(n_channels: int = 19) -> np.ndarray:
    """Spatial covariance from approximate 10-20 electrode distances."""
    # Rough 2D scalp coordinates (normalized)
    coords = {
        "Fp1": (-0.3, 0.9), "Fp2": (0.3, 0.9),
        "F7": (-0.7, 0.5), "F3": (-0.4, 0.55), "Fz": (0.0, 0.6), "F4": (0.4, 0.55), "F8": (0.7, 0.5),
        "T3": (-0.9, 0.0), "C3": (-0.45, 0.0), "Cz": (0.0, 0.0), "C4": (0.45, 0.0), "T4": (0.9, 0.0),
        "T5": (-0.7, -0.5), "P3": (-0.4, -0.55), "Pz": (0.0, -0.6), "P4": (0.4, -0.55), "T6": (0.7, -0.5),
        "O1": (-0.3, -0.9), "O2": (0.3, -0.9),
    }
    names = CHANNEL_NAMES_10_20
    pos = np.array([coords[n] for n in names], dtype=np.float64)
    dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=-1))
    cov = np.exp(-dist / 0.45)
    cov = cov + 1e-3 * np.eye(n_channels)
    # Ensure PD
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    return (eigvecs * eigvals) @ eigvecs.T


def generate_synthetic_eeg_fallback(
    condition: str,
    n_subjects: int = 20,
    duration_s: float = 60.0,
    sfreq: float = SFREQ_DEFAULT,
    noise_seed: int = 42,
) -> np.ndarray:
    """
    Generate realistic synthetic EEG WITHOUT TVB (scipy only).

    Uses known spectral characteristics of AD vs healthy EEG (Jeong 2004)
    with drug-specific amplitude modifiers (Babiloni 2006; Winblad 2007).

    Returns
    -------
    eeg_array : (n_subjects, 19, duration_s * sfreq)
    """
    n_time = int(duration_s * sfreq)
    t = np.arange(n_time) / sfreq
    cov = _electrode_covariance(N_CHANNELS)
    chol = np.linalg.cholesky(cov)

    # Base profile
    if condition in ("healthy_control",):
        base = dict(FALLBACK_PROFILES["healthy_control"])
        drugs = []
    elif condition == "alzheimer_mild":
        base = dict(FALLBACK_PROFILES["alzheimer_mild"])
        drugs = []
    elif condition in ("alzheimer_moderate", "alzheimer_severe"):
        key = condition if condition in FALLBACK_PROFILES else "alzheimer_moderate"
        base = dict(FALLBACK_PROFILES[key])
        drugs = []
    elif condition == "alzheimer_donepezil":
        base = dict(FALLBACK_PROFILES["alzheimer_moderate"])
        drugs = ["donepezil"]
    elif condition == "alzheimer_memantine":
        base = dict(FALLBACK_PROFILES["alzheimer_moderate"])
        drugs = ["memantine"]
    elif condition == "alzheimer_combination":
        base = dict(FALLBACK_PROFILES["alzheimer_moderate"])
        drugs = ["donepezil", "memantine"]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Drug modifications — Babiloni et al. (2006); Winblad et al. (2007)
    for drug in drugs:
        if drug == "donepezil":
            base["alpha"] *= 1.25
            base["theta"] *= 0.80
            base["delta"] *= 0.85
        elif drug == "memantine":
            base["delta"] *= 0.90
            base["theta"] *= 0.85
            base["alpha"] *= 1.10

    band_freqs = {"delta": 2.0, "theta": 6.0, "alpha": 10.0, "beta": 20.0}
    out = np.zeros((n_subjects, N_CHANNELS, n_time), dtype=np.float64)

    for s_idx in tqdm(range(n_subjects), desc=f"Fallback EEG [{condition}]"):
        rng = np.random.default_rng(noise_seed + s_idx)
        amps = {
            b: base[b] * (1.0 + 0.1 * rng.normal())
            for b in ("delta", "theta", "alpha", "beta")
        }
        # Independent oscillators per channel, then mix spatially
        raw = np.zeros((N_CHANNELS, n_time), dtype=np.float64)
        for ch in range(N_CHANNELS):
            sig = np.zeros(n_time)
            for band, f0 in band_freqs.items():
                phase = rng.uniform(0, 2 * np.pi)
                # Small per-channel frequency jitter
                f = f0 * (1.0 + 0.02 * rng.normal())
                sig += amps[band] * np.sin(2 * np.pi * f * t + phase)
            sig += 3.0 * _pink_noise(n_time, rng)
            raw[ch] = sig

        # Spatial correlation (nearby 10-20 electrodes correlated)
        mixed = chol @ raw
        out[s_idx] = mixed

    return out


def _ensure_psd_20(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Interpolate / truncate PSD to exactly PSD_DIM frequency bins."""
    target_freqs = np.linspace(max(freqs.min(), 0.5), min(freqs.max(), 40.0), PSD_DIM)
    out = np.zeros((psd.shape[0], PSD_DIM), dtype=np.float64)
    for ch in range(psd.shape[0]):
        out[ch] = np.interp(target_freqs, freqs, psd[ch])
    return out


def _flatten_features(
    psd: np.ndarray,
    band_powers: np.ndarray,
    coherence: np.ndarray,
    plv: np.ndarray,
) -> np.ndarray:
    """
    Flatten to 2185-D matching prepare_target_features / model_wrapper.

    psd: (19, 20), band_powers: (19, 5), coherence/plv: (5, 19, 19)
    """
    psd_flat = psd.reshape(-1)
    bp_flat = band_powers.reshape(-1)
    parts = [psd_flat, bp_flat]
    tri = np.triu_indices(N_CHANNELS, k=1)
    for b in range(N_BANDS):
        parts.append(coherence[b][tri].astype(np.float64))
        parts.append(plv[b][tri].astype(np.float64))
    vec = np.concatenate(parts).astype(np.float64)
    if vec.size != EEG_FLAT_SIZE:
        raise ValueError(f"Expected {EEG_FLAT_SIZE} features, got {vec.size}")
    return vec


def extract_tvb_eeg_features(
    eeg_array: np.ndarray,
    sfreq: float = SFREQ_DEFAULT,
) -> np.ndarray:
    """
    Extract features in the SAME 2185-dim format as the real pipeline.

    Reuses EEGFeatureExtractor + ConnectivityAnalyzer — does not reimplement
    PSD / band power / coherence / PLV.

    Parameters
    ----------
    eeg_array : (n_subjects, 19, n_timepoints)

    Returns
    -------
    features_array : (n_subjects, 2185)
    """
    if eeg_array.ndim != 3 or eeg_array.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected (n, 19, T), got {eeg_array.shape}")

    n_subjects = eeg_array.shape[0]
    # At 256 Hz, n_per_seg=128 → ~2 Hz resolution → ~20 bins in [0.5, 40]
    # (training data used 500 Hz + n_per_seg=256 → same ~20 bins)
    psd_cfg = PSD_CONFIG.copy()
    if abs(sfreq - 256.0) < 1.0:
        psd_cfg["n_per_seg"] = 128
    extractor = EEGFeatureExtractor(psd_config=psd_cfg)
    analyzer = ConnectivityAnalyzer()
    band_names = list(FREQUENCY_BANDS.keys())

    features = np.zeros((n_subjects, EEG_FLAT_SIZE), dtype=np.float64)
    info = mne.create_info(ch_names=CHANNEL_NAMES_10_20, sfreq=sfreq, ch_types="eeg")

    for i in tqdm(range(n_subjects), desc="Feature extraction"):
        data = np.asarray(eeg_array[i], dtype=np.float64)

        # Create MNE RawArray (canonical channel names) then pull data back
        raw = mne.io.RawArray(data, info, verbose=False)
        data = raw.get_data()

        freqs, psd = extractor.compute_psd(data, sfreq)
        psd = _ensure_psd_20(freqs, psd)

        band_dict = extractor.compute_all_band_powers(data, sfreq)
        band_powers = np.stack([band_dict[b] for b in band_names], axis=1)  # (19, 5)

        coh_dict = analyzer.compute_multiband_connectivity(data, sfreq, method="coherence")
        plv_dict = analyzer.compute_multiband_connectivity(data, sfreq, method="plv")
        coherence = np.stack([coh_dict[b] for b in band_names], axis=0)
        plv = np.stack([plv_dict[b] for b in band_names], axis=0)

        features[i] = _flatten_features(psd, band_powers, coherence, plv)

    return features


def _band_means_from_features(features: np.ndarray) -> Dict[str, float]:
    """Mean band / connectivity summaries (reshape-correct band extraction)."""
    return {
        "delta": band_block_mean(features, "delta"),
        "theta": band_block_mean(features, "theta"),
        "alpha": band_block_mean(features, "alpha"),
        "beta": band_block_mean(features, "beta"),
        "connectivity": connectivity_block_mean(features),
    }


def _build_condition_simulator(condition: str):
    """Build TVB simulator for a named validation condition."""
    if condition == "healthy_control":
        sim, conn = build_alzheimer_network(severity="healthy")
        return sim, conn, {"severity": "healthy", "drugs": []}

    if condition == "alzheimer_mild":
        sim, conn = build_alzheimer_network(severity="mild")
        return sim, conn, {"severity": "mild", "drugs": []}

    if condition == "alzheimer_moderate":
        sim, conn = build_alzheimer_network(severity="moderate")
        return sim, conn, {"severity": "moderate", "drugs": []}

    if condition == "alzheimer_donepezil":
        sim, conn = build_alzheimer_network(severity="moderate")
        apply_donepezil_effect(sim, dose_level="standard")
        return sim, conn, {"severity": "moderate", "drugs": ["donepezil_standard"]}

    if condition == "alzheimer_memantine":
        sim, conn = build_alzheimer_network(severity="moderate")
        apply_memantine_effect(sim, dose_level="standard")
        return sim, conn, {"severity": "moderate", "drugs": ["memantine_standard"]}

    if condition == "alzheimer_combination":
        sim, conn = build_alzheimer_network(severity="moderate")
        apply_donepezil_effect(sim, dose_level="standard")
        apply_memantine_effect(sim, dose_level="standard")
        return sim, conn, {"severity": "moderate", "drugs": ["donepezil_standard", "memantine_standard"]}

    raise ValueError(f"Unknown condition: {condition}")


def generate_full_tvb_validation_dataset(
    output_dir: str | Path = "data/tvb_validation",
    n_subjects_per_condition: int = 20,
    duration_ms: float = 60000,
    use_fallback: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Generate the complete TVB (or fallback) validation dataset for 6 conditions.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    force_fallback = use_fallback if use_fallback is not None else (not TVB_AVAILABLE)
    prefix = "synthetic_fallback_" if force_fallback else "tvb_"

    results: Dict[str, Any] = {"prefix": prefix, "conditions": {}, "tvb_available": TVB_AVAILABLE}

    for condition in CONDITIONS:
        feat_path = output_dir / f"{prefix}{condition}_features.npy"
        meta_path = output_dir / f"{prefix}{condition}_metadata.json"

        if feat_path.exists():
            print(f"[cache] Loading {feat_path.name}")
            feats = np.load(feat_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        else:
            print(f"\n=== Generating: {condition} (n={n_subjects_per_condition}) ===")
            meta = {
                "condition": condition,
                "n_subjects": n_subjects_per_condition,
                "duration_ms": duration_ms,
                "sfreq": SFREQ_DEFAULT,
                "method": "synthetic_fallback" if force_fallback else "tvb_wilson_cowan",
                "feature_dim": EEG_FLAT_SIZE,
            }

            try:
                if force_fallback:
                    eeg = generate_synthetic_eeg_fallback(
                        condition,
                        n_subjects=n_subjects_per_condition,
                        duration_s=duration_ms / 1000.0,
                        sfreq=SFREQ_DEFAULT,
                    )
                else:
                    sim, conn, build_meta = _build_condition_simulator(condition)
                    meta.update(build_meta)
                    eeg = run_tvb_simulation(
                        sim,
                        duration_ms=duration_ms,
                        n_subjects=n_subjects_per_condition,
                        noise_seed=42,
                    )
            except Exception as exc:
                print(f"[WARN] TVB path failed for {condition}: {exc}")
                print("       Falling back to synthetic EEG generator.")
                force_fallback = True
                prefix = "synthetic_fallback_"
                feat_path = output_dir / f"{prefix}{condition}_features.npy"
                meta_path = output_dir / f"{prefix}{condition}_metadata.json"
                meta["method"] = "synthetic_fallback"
                meta["tvb_error"] = str(exc)
                if feat_path.exists():
                    feats = np.load(feat_path)
                    results["conditions"][condition] = {
                        "features_path": str(feat_path),
                        "band_means": _band_means_from_features(feats),
                        "n": int(feats.shape[0]),
                        "cached": True,
                    }
                    continue
                eeg = generate_synthetic_eeg_fallback(
                    condition,
                    n_subjects=n_subjects_per_condition,
                    duration_s=duration_ms / 1000.0,
                    sfreq=SFREQ_DEFAULT,
                )

            feats = extract_tvb_eeg_features(eeg, sfreq=SFREQ_DEFAULT)
            np.save(feat_path, feats)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"Saved {feat_path.name} shape={feats.shape}")

        results["conditions"][condition] = {
            "features_path": str(feat_path),
            "band_means": _band_means_from_features(feats),
            "n": int(feats.shape[0]),
            "cached": feat_path.exists(),
        }

    results["prefix"] = prefix

    # Ground-truth reference JSON
    gt_path = output_dir / "tvb_ground_truth_summary.json"
    gt_path.write_text(json.dumps(GROUND_TRUTH, indent=2), encoding="utf-8")

    # Summary table (ASCII separators — Windows cp1252 consoles cannot print box-drawing)
    print("\nTVB VALIDATION DATASET")
    print("-" * 74)
    print(f"{'Condition':<24} {'N':>4}  {'Alpha_mean':>11}  {'Theta_mean':>11}  {'Conn_mean':>11}")
    for cond in (
        "healthy_control",
        "alzheimer_moderate",
        "alzheimer_donepezil",
        "alzheimer_memantine",
    ):
        info = results["conditions"].get(cond, {})
        bm = info.get("band_means", {})
        print(
            f"{cond:<24} {info.get('n', 0):>4}  "
            f"{bm.get('alpha', float('nan')):>11.3f}  "
            f"{bm.get('theta', float('nan')):>11.3f}  "
            f"{bm.get('connectivity', float('nan')):>11.3f}"
        )
    print("-" * 74)

    ad = results["conditions"].get("alzheimer_moderate", {}).get("band_means", {})
    don = results["conditions"].get("alzheimer_donepezil", {}).get("band_means", {})
    if ad and don:
        a_dir = "UP" if don["alpha"] > ad["alpha"] else "DOWN"
        t_dir = "UP" if don["theta"] > ad["theta"] else "DOWN"
        print("Expected Donepezil: alpha UP, theta DOWN")
        print(f"Observed Donepezil: alpha {a_dir}, theta {t_dir}")
    print("-" * 74)

    summary_path = output_dir / f"{prefix}dataset_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


if __name__ == "__main__":
    generate_full_tvb_validation_dataset()
