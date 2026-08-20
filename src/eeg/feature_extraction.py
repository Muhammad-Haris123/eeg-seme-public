"""
EEG Feature Extraction Module.

This module extracts biologically meaningful features from preprocessed EEG:
- Spectral features (Power Spectral Density, band powers)
- Connectivity features (Coherence, PLV)

All features are designed to be interpretable for explainability.

Author: Research Team
Date: 2026
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import torch
from scipy import signal

from ..utils.config import (
    FREQUENCY_BANDS,
    PSD_CONFIG,
    EEG_FEATURES_DIR
)
from .connectivity import compute_connectivity_features, ConnectivityAnalyzer

warnings.filterwarnings('ignore', category=RuntimeWarning)


class EEGFeatureExtractor:
    """
    Feature extractor for EEG data.
    
    Extracts:
    - Spectral features (PSD, band powers)
    - Connectivity features (coherence, PLV)
    """
    
    def __init__(
        self,
        frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        psd_config: Optional[Dict] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            frequency_bands: Frequency band definitions
            psd_config: PSD computation configuration
        """
        self.frequency_bands = frequency_bands or FREQUENCY_BANDS.copy()
        self.psd_config = psd_config or PSD_CONFIG.copy()
    
    def compute_psd(
        self,
        data: np.ndarray,
        sfreq: float,
        method: str = 'welch'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density.
        
        Args:
            data: EEG data (channels, time) or (epochs, channels, time)
            sfreq: Sampling frequency
            method: PSD method ('welch' or 'multitaper')
        
        Returns:
            Tuple of (frequencies, PSD) where PSD shape is (channels, frequencies)
        """
        # Handle epoch dimension
        if data.ndim == 3:
            # Average across epochs
            data = np.mean(data, axis=0)
        
        n_channels = data.shape[0]
        
        # Compute PSD for each channel
        psd_list = []
        freqs_list = []
        
        for ch_idx in range(n_channels):
            if method == 'welch':
                freqs, psd = signal.welch(
                    data[ch_idx],
                    fs=sfreq,
                    nperseg=self.psd_config.get('n_per_seg', None),
                    noverlap=self.psd_config.get('n_overlap', None),
                    nfft=self.psd_config.get('n_fft', None),
                    window=self.psd_config.get('window', 'hann')
                )
            else:
                raise ValueError(f"Unsupported PSD method: {method}")
            
            # Filter frequency range
            fmin = self.psd_config.get('fmin', 0.5)
            fmax = self.psd_config.get('fmax', 40.0)
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            
            freqs_list.append(freqs[freq_mask])
            psd_list.append(psd[freq_mask])
        
        # Use first channel's frequencies (should be same for all)
        freqs = freqs_list[0]
        psd = np.array(psd_list)  # (channels, frequencies)
        
        return freqs, psd
    
    def compute_band_power(
        self,
        data: np.ndarray,
        sfreq: float,
        band: str
    ) -> np.ndarray:
        """
        Compute power in a specific frequency band.
        
        Args:
            data: EEG data (channels, time) or (epochs, channels, time)
            sfreq: Sampling frequency
            band: Frequency band name
        
        Returns:
            Band power per channel (channels,) or (epochs, channels)
        """
        if band not in self.frequency_bands:
            raise ValueError(f"Unknown band: {band}")
        
        fmin, fmax = self.frequency_bands[band]
        
        # Compute PSD
        freqs, psd = self.compute_psd(data, sfreq)
        
        # Find frequency indices in band
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        
        # Integrate power in band (trapezoidal rule)
        band_power = np.trapz(psd[:, freq_mask], freqs[freq_mask], axis=1)
        
        return band_power
    
    def compute_all_band_powers(
        self,
        data: np.ndarray,
        sfreq: float
    ) -> Dict[str, np.ndarray]:
        """
        Compute power for all frequency bands.
        
        Args:
            data: EEG data (channels, time) or (epochs, channels, time)
            sfreq: Sampling frequency
        
        Returns:
            Dictionary mapping band names to power arrays
        """
        band_powers = {}
        
        for band_name in self.frequency_bands.keys():
            band_powers[band_name] = self.compute_band_power(data, sfreq, band_name)
        
        return band_powers
    
    def extract_spectral_features(
        self,
        preprocessed_data: np.ndarray,
        sfreq: float,
        per_epoch: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract spectral features from preprocessed EEG.
        
        Args:
            preprocessed_data: EEG data (subjects, epochs, channels, time)
            sfreq: Sampling frequency
            per_epoch: If True, compute per epoch; if False, average across epochs
        
        Returns:
            Dictionary with spectral features:
            - 'psd': Power spectral density
            - 'band_powers': Band power features
        """
        n_subjects, n_epochs, n_channels, n_time = preprocessed_data.shape
        
        features = {}
        
        if per_epoch:
            # Compute per epoch
            # Shape: (subjects, epochs, channels, frequencies)
            psd_list = []
            band_powers_list = []
            
            for subj_idx in range(n_subjects):
                subject_psd = []
                subject_bands = []
                
                for epoch_idx in range(n_epochs):
                    epoch_data = preprocessed_data[subj_idx, epoch_idx, :, :]
                    freqs, psd = self.compute_psd(epoch_data, sfreq)
                    band_powers = self.compute_all_band_powers(epoch_data, sfreq)
                    
                    subject_psd.append(psd)
                    # Stack band powers: (channels, bands)
                    band_stack = np.stack([
                        band_powers[band] 
                        for band in self.frequency_bands.keys()
                    ], axis=1)
                    subject_bands.append(band_stack)
                
                psd_list.append(np.stack(subject_psd, axis=0))
                band_powers_list.append(np.stack(subject_bands, axis=0))
            
            features['psd'] = np.stack(psd_list, axis=0)
            features['band_powers'] = np.stack(band_powers_list, axis=0)
            features['frequencies'] = freqs
        
        else:
            # Average across epochs first
            psd_list = []
            band_powers_list = []
            
            for subj_idx in range(n_subjects):
                # Average across epochs
                subject_data = np.mean(preprocessed_data[subj_idx, :, :, :], axis=0)
                freqs, psd = self.compute_psd(subject_data, sfreq)
                band_powers = self.compute_all_band_powers(subject_data, sfreq)
                
                psd_list.append(psd)
                # Stack band powers: (channels, bands)
                band_stack = np.stack([
                    band_powers[band] 
                    for band in self.frequency_bands.keys()
                ], axis=1)
                band_powers_list.append(band_stack)
            
            features['psd'] = np.stack(psd_list, axis=0)
            features['band_powers'] = np.stack(band_powers_list, axis=0)
            features['frequencies'] = freqs
        
        return features
    
    def extract_all_features(
        self,
        preprocessed_data: np.ndarray,
        sfreq: float,
        include_connectivity: bool = True,
        connectivity_methods: Optional[List[str]] = None,
        per_epoch: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract all features (spectral + connectivity).
        
        Args:
            preprocessed_data: EEG data (subjects, epochs, channels, time)
            sfreq: Sampling frequency
            include_connectivity: Whether to compute connectivity features
            connectivity_methods: List of connectivity methods
            per_epoch: If True, compute per epoch
        
        Returns:
            Dictionary with all features
        """
        all_features = {}
        
        # Spectral features
        print("Extracting spectral features...")
        spectral_features = self.extract_spectral_features(
            preprocessed_data, sfreq, per_epoch=per_epoch
        )
        all_features.update(spectral_features)
        
        # Connectivity features
        if include_connectivity:
            print("Extracting connectivity features...")
            connectivity_features = compute_connectivity_features(
                preprocessed_data,
                sfreq,
                methods=connectivity_methods,
                per_epoch=per_epoch
            )
            all_features.update(connectivity_features)
        
        return all_features
    
    def save_features(
        self,
        features: Dict[str, np.ndarray],
        group: str,
        output_dir: Optional[Path] = None
    ):
        """
        Save extracted features to disk.
        
        Args:
            features: Dictionary of features
            group: Group identifier
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = EEG_FEATURES_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each feature type
        for feature_name, feature_data in features.items():
            if isinstance(feature_data, np.ndarray):
                feature_path = output_dir / f"{group}_{feature_name}.npy"
                np.save(feature_path, feature_data)
                print(f"Saved {feature_name}: {feature_path} (shape: {feature_data.shape})")
        
        # Save metadata
        import json
        metadata = {
            'group': group,
            'feature_names': list(features.keys()),
            'shapes': {name: list(data.shape) if isinstance(data, np.ndarray) else str(data)
                      for name, data in features.items()}
        }
        metadata_path = output_dir / f"{group}_features_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def extract_features_for_all_groups(
    preprocessed_data: Dict[str, Tuple[np.ndarray, List[Dict]]],
    include_connectivity: bool = True,
    save_output: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract features for all groups.
    
    Args:
        preprocessed_data: Dictionary mapping group to (data, metadata)
        include_connectivity: Whether to compute connectivity
        save_output: Whether to save features
    
    Returns:
        Dictionary mapping group to features dictionary
    """
    extractor = EEGFeatureExtractor()
    
    all_features = {}
    
    for group, (data, metadata_list) in preprocessed_data.items():
        print(f"\n{'='*60}")
        print(f"Extracting features for {group} group...")
        print(f"{'='*60}")
        
        # Get sampling frequency from metadata
        sfreq = metadata_list[0]['sfreq']
        
        # Extract features
        features = extractor.extract_all_features(
            data,
            sfreq,
            include_connectivity=include_connectivity,
            per_epoch=False  # Average across epochs for efficiency
        )
        
        all_features[group] = features
        
        if save_output:
            extractor.save_features(features, group)
    
    return all_features

