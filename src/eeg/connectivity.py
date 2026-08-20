"""
EEG Connectivity Analysis Module.

This module computes connectivity measures for EEG data:
- Coherence (magnitude-squared coherence)
- Phase Locking Value (PLV)

These features are biologically interpretable and suitable for explainability.

Author: Research Team
Date: 2026
"""

import warnings
from typing import Dict, Optional, Tuple

import mne
import numpy as np
from scipy import signal

from ..utils.config import CONNECTIVITY_CONFIG, FREQUENCY_BANDS

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ConnectivityAnalyzer:
    """
    Analyzer for EEG connectivity measures.
    
    Computes:
    - Coherence: Linear correlation in frequency domain
    - PLV: Phase synchronization measure
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize connectivity analyzer.
        
        Args:
            config: Connectivity configuration (defaults to CONNECTIVITY_CONFIG)
        """
        self.config = config or CONNECTIVITY_CONFIG.copy()
        self.frequency_bands = FREQUENCY_BANDS.copy()
    
    def compute_coherence(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        sfreq: float,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude-squared coherence between two signals.
        
        Uses manual implementation to ensure correctness:
        C_xy(f) = |P_xy(f)|^2 / (P_xx(f) * P_yy(f))
        
        Args:
            data1: First signal (time series)
            data2: Second signal (time series)
            sfreq: Sampling frequency (Hz)
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz)
        
        Returns:
            Tuple of (frequencies, coherence)
        """
        if fmin is None:
            fmin = self.config['fmin']
        if fmax is None:
            fmax = self.config['fmax']
        
        n_fft = self.config.get('n_fft', None)
        if n_fft is None:
            # CRITICAL FIX: Use smaller segments to ensure multiple segments for Welch method
            # This prevents degenerate coherence = 1.0
            # Standard practice: Use 1/4 to 1/2 of signal length, cap at 512
            signal_length = len(data1)
            n_fft = min(signal_length // 4, 512)  # Cap at 512 for efficiency
        
        # Ensure minimum segment length (at least 128 samples for meaningful FFT)
        # and maximum (not larger than signal)
        if n_fft < 128:
            n_fft = min(len(data1), 128)
        elif n_fft > len(data1):
            n_fft = len(data1)
        
        # CRITICAL: Ensure we have at least 2 segments for Welch method
        # If n_fft >= signal_length, we'd only have 1 segment → coherence = 1.0 (degenerate)
        if n_fft >= len(data1):
            n_fft = len(data1) // 2
            if n_fft < 64:
                n_fft = 64
        
        n_overlap = self.config.get('n_overlap', None)
        if n_overlap is None:
            n_overlap = n_fft // 2
        
        # Manual coherence computation using Welch's method
        # Compute cross-spectral density and power spectral densities
        freqs, Pxy = signal.csd(
            data1, data2,
            fs=sfreq,
            nperseg=n_fft,
            noverlap=n_overlap,
            window='hann'
        )
        
        _, Pxx = signal.welch(
            data1,
            fs=sfreq,
            nperseg=n_fft,
            noverlap=n_overlap,
            window='hann'
        )
        
        _, Pyy = signal.welch(
            data2,
            fs=sfreq,
            nperseg=n_fft,
            noverlap=n_overlap,
            window='hann'
        )
        
        # Compute magnitude-squared coherence
        # C_xy(f) = |P_xy(f)|^2 / (P_xx(f) * P_yy(f))
        coh = np.abs(Pxy) ** 2 / (Pxx * Pyy)
        
        # Handle numerical issues (division by zero, etc.)
        coh = np.nan_to_num(coh, nan=0.0, posinf=1.0, neginf=0.0)
        coh = np.clip(coh, 0.0, 1.0)
        
        # Filter frequency range
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs_filtered = freqs[freq_mask]
        coh_filtered = coh[freq_mask]
        
        return freqs_filtered, coh_filtered
    
    def compute_plv(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float
    ) -> float:
        """
        Compute Phase Locking Value (PLV) between two signals.
        
        PLV measures phase synchronization independent of amplitude.
        
        Args:
            data1: First signal (time series)
            data2: Second signal (time series)
            sfreq: Sampling frequency (Hz)
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz)
        
        Returns:
            PLV value (0-1, where 1 = perfect phase locking)
        """
        # Band-pass filter
        sos = signal.butter(4, [fmin, fmax], btype='band', fs=sfreq, output='sos')
        data1_filtered = signal.sosfiltfilt(sos, data1)
        data2_filtered = signal.sosfiltfilt(sos, data2)
        
        # Compute analytic signal (Hilbert transform)
        analytic1 = signal.hilbert(data1_filtered)
        analytic2 = signal.hilbert(data2_filtered)
        
        # Extract phases
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # Compute phase difference
        phase_diff = phase1 - phase2
        
        # PLV = |mean(exp(i * phase_diff))|
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return plv
    
    def compute_band_coherence(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        sfreq: float,
        band: str
    ) -> float:
        """
        Compute coherence for a specific frequency band.
        
        Args:
            data1: First signal
            data2: Second signal
            sfreq: Sampling frequency
            band: Frequency band name ('delta', 'theta', 'alpha', 'beta', 'gamma')
        
        Returns:
            Mean coherence in the specified band
        """
        if band not in self.frequency_bands:
            raise ValueError(f"Unknown band: {band}. Must be one of {list(self.frequency_bands.keys())}")
        
        fmin, fmax = self.frequency_bands[band]
        
        # Ensure signals have sufficient length and variance
        if len(data1) < 100 or len(data2) < 100:
            # Too short for reliable coherence
            return 0.0
        
        if np.std(data1) < 1e-10 or np.std(data2) < 1e-10:
            # Zero variance - return 0 (not 1.0)
            return 0.0
        
        # Check if signals are identical (would give coherence = 1.0)
        if np.allclose(data1, data2, atol=1e-10):
            # Only return 1.0 if signals are truly identical
            return 1.0
        
        freqs, coh = self.compute_coherence(data1, data2, sfreq, fmin=fmin, fmax=fmax)
        
        # Filter out any NaN or Inf values
        coh = coh[np.isfinite(coh)]
        
        if len(coh) == 0:
            return 0.0
        
        # Return mean coherence in band
        mean_coh = np.mean(coh)
        
        # Sanity check: coherence should be in [0, 1]
        mean_coh = np.clip(mean_coh, 0.0, 1.0)
        
        return mean_coh
    
    def compute_connectivity_matrix(
        self,
        data: np.ndarray,
        sfreq: float,
        method: str = 'coherence',
        band: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute connectivity matrix for all channel pairs.
        
        Args:
            data: EEG data (channels, time) or (epochs, channels, time)
            sfreq: Sampling frequency
            method: Connectivity method ('coherence' or 'plv')
            band: Frequency band (if None, uses full range)
        
        Returns:
            Connectivity matrix (channels, channels)
        """
        # Handle epoch dimension - SCIENTIFICALLY BEST: Per-epoch coherence then average
        # This is the gold standard approach used in high-quality EEG research
        # We sample epochs to make it feasible while maintaining scientific validity
        if data.ndim == 3:
            n_epochs, n_channels, n_time = data.shape
            
            # SCIENTIFICALLY VALID APPROACH: Sample epochs, compute per-epoch, then average
            # This captures epoch-to-epoch variability (important for robustness)
            # Sampling strategy: Use 50-100 epochs (or all if fewer)
            # This balances accuracy with computational feasibility
            
            target_epochs = 100  # Scientifically sufficient for robust estimates
            if n_epochs <= target_epochs:
                # Use all epochs if we have few enough
                epoch_indices = range(n_epochs)
            else:
                # Sample uniformly (not randomly) to ensure temporal coverage
                # This is more representative than random sampling
                step = n_epochs / target_epochs
                epoch_indices = [int(i * step) for i in range(target_epochs)]
                # Ensure we don't exceed bounds
                epoch_indices = [min(idx, n_epochs - 1) for idx in epoch_indices]
            
            epoch_matrices = []
            
            for epoch_idx in epoch_indices:
                epoch_data = data[epoch_idx, :, :]
                epoch_matrix = self._compute_connectivity_matrix_single(
                    epoch_data, sfreq, method, band
                )
                epoch_matrices.append(epoch_matrix)
            
            # Average coherence matrices across epochs
            # This is the scientifically correct way to aggregate connectivity
            connectivity_matrix = np.mean(epoch_matrices, axis=0)
        else:
            connectivity_matrix = self._compute_connectivity_matrix_single(
                data, sfreq, method, band
            )
        
        return connectivity_matrix
    
    def _compute_connectivity_matrix_single(
        self,
        data: np.ndarray,
        sfreq: float,
        method: str = 'coherence',
        band: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute connectivity matrix for single epoch or averaged data.
        
        Args:
            data: EEG data (channels, time)
            sfreq: Sampling frequency
            method: Connectivity method ('coherence' or 'plv')
            band: Frequency band (if None, uses full range)
        
        Returns:
            Connectivity matrix (channels, channels)
        """
        n_channels = data.shape[0]
        connectivity_matrix = np.zeros((n_channels, n_channels))
        
        if band is not None:
            fmin, fmax = self.frequency_bands[band]
        else:
            fmin = self.config['fmin']
            fmax = self.config['fmax']
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    # Diagonal: self-connection (coherence = 1.0 by definition)
                    connectivity_matrix[i, j] = 1.0
                else:
                    if method == 'coherence':
                        if band is not None:
                            connectivity_matrix[i, j] = self.compute_band_coherence(
                                data[i], data[j], sfreq, band
                            )
                        else:
                            _, coh = self.compute_coherence(
                                data[i], data[j], sfreq, fmin=fmin, fmax=fmax
                            )
                            # Filter NaN/Inf and clip to [0, 1]
                            coh = coh[np.isfinite(coh)]
                            if len(coh) > 0:
                                connectivity_matrix[i, j] = np.clip(np.mean(coh), 0.0, 1.0)
                            else:
                                connectivity_matrix[i, j] = 0.0
                    
                    elif method == 'plv':
                        connectivity_matrix[i, j] = self.compute_plv(
                            data[i], data[j], sfreq, fmin=fmin, fmax=fmax
                        )
                    
                    else:
                        raise ValueError(f"Unknown method: {method}")
        
        return connectivity_matrix
    
    def compute_multiband_connectivity(
        self,
        data: np.ndarray,
        sfreq: float,
        method: str = 'coherence'
    ) -> Dict[str, np.ndarray]:
        """
        Compute connectivity matrices for all frequency bands.
        
        Args:
            data: EEG data (channels, time) or (epochs, channels, time)
            sfreq: Sampling frequency
            method: Connectivity method ('coherence' or 'plv')
        
        Returns:
            Dictionary mapping band names to connectivity matrices
        """
        connectivity_dict = {}
        
        for band_name in self.frequency_bands.keys():
            connectivity_dict[band_name] = self.compute_connectivity_matrix(
                data, sfreq, method=method, band=band_name
            )
        
        return connectivity_dict


def compute_connectivity_features(
    preprocessed_data: np.ndarray,
    sfreq: float,
    methods: Optional[list] = None,
    per_epoch: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute connectivity features for preprocessed EEG data.
    
    Args:
        preprocessed_data: EEG data (subjects, epochs, channels, time)
        sfreq: Sampling frequency
        methods: List of methods ['coherence', 'plv'] (defaults to config)
        per_epoch: If True, compute per epoch; if False, average across epochs
    
    Returns:
        Dictionary with connectivity features
    """
    if methods is None:
        methods = CONNECTIVITY_CONFIG['methods']
    
    analyzer = ConnectivityAnalyzer()
    
    n_subjects, n_epochs, n_channels, n_time = preprocessed_data.shape
    
    features = {}
    
    for method in methods:
        print(f"  Computing {method} connectivity...")
        if per_epoch:
            # Shape: (subjects, epochs, bands, channels, channels)
            method_features = []
            
            for subj_idx in range(n_subjects):
                if subj_idx % 5 == 0:
                    print(f"    Subject {subj_idx+1}/{n_subjects}...")
                subject_features = []
                for epoch_idx in range(n_epochs):
                    epoch_data = preprocessed_data[subj_idx, epoch_idx, :, :]
                    band_connectivity = analyzer.compute_multiband_connectivity(
                        epoch_data, sfreq, method=method
                    )
                    # Stack bands: (bands, channels, channels)
                    band_stack = np.stack([
                        band_connectivity[band] 
                        for band in FREQUENCY_BANDS.keys()
                    ], axis=0)
                    subject_features.append(band_stack)
                
                method_features.append(np.stack(subject_features, axis=0))
            
            features[method] = np.stack(method_features, axis=0)
        
        else:
            # CRITICAL FIX: Compute connectivity per epoch, then average
            # Shape: (subjects, bands, channels, channels)
            method_features = []
            
            for subj_idx in range(n_subjects):
                if subj_idx % 5 == 0:
                    print(f"    Processing subject {subj_idx+1}/{n_subjects}...")
                
                # Pass all epochs to compute_connectivity_matrix
                # It will compute per-epoch and average internally
                subject_epochs = preprocessed_data[subj_idx, :, :, :]  # (epochs, channels, time)
                band_connectivity = analyzer.compute_multiband_connectivity(
                    subject_epochs, sfreq, method=method
                )
                # Stack bands: (bands, channels, channels)
                band_stack = np.stack([
                    band_connectivity[band] 
                    for band in FREQUENCY_BANDS.keys()
                ], axis=0)
                method_features.append(band_stack)
            
            features[method] = np.stack(method_features, axis=0)
    
    return features



