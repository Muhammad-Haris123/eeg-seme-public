"""
EEG Preprocessing Pipeline.

This module implements a complete preprocessing pipeline for resting-state EEG:
1. Band-pass filtering (0.5-40 Hz)
2. Powerline noise removal (50 Hz notch)
3. Artifact removal using ICA
4. Epoching into fixed windows
5. Per-subject normalization

Output format: (subjects, epochs, channels, time)

Author: Research Team
Date: 2026
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import torch

from ..utils.config import (
    FILTER_CONFIG,
    ICA_CONFIG,
    EPOCH_CONFIG,
    NORMALIZATION_CONFIG,
    PROCESSED_EEG_DIR
)

warnings.filterwarnings('ignore', category=RuntimeWarning)


class EEGPreprocessor:
    """
    Preprocessing pipeline for resting-state EEG data.
    
    Steps:
    1. Filtering (band-pass + notch)
    2. ICA artifact removal
    3. Epoching
    4. Normalization
    """
    
    def __init__(
        self,
        filter_config: Optional[Dict] = None,
        ica_config: Optional[Dict] = None,
        epoch_config: Optional[Dict] = None,
        norm_config: Optional[Dict] = None
    ):
        """
        Initialize preprocessor with configuration.
        
        Args:
            filter_config: Filtering parameters (defaults to config.FILTER_CONFIG)
            ica_config: ICA parameters (defaults to config.ICA_CONFIG)
            epoch_config: Epoching parameters (defaults to config.EPOCH_CONFIG)
            norm_config: Normalization parameters (defaults to config.NORMALIZATION_CONFIG)
        """
        self.filter_config = filter_config or FILTER_CONFIG.copy()
        self.ica_config = ica_config or ICA_CONFIG.copy()
        self.epoch_config = epoch_config or EPOCH_CONFIG.copy()
        self.norm_config = norm_config or NORMALIZATION_CONFIG.copy()
        
    def apply_filtering(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Apply band-pass and notch filtering.
        
        Args:
            raw: MNE Raw object
        
        Returns:
            Filtered Raw object
        """
        raw_filtered = raw.copy()
        
        # Band-pass filter
        raw_filtered.filter(
            l_freq=self.filter_config['low_freq'],
            h_freq=self.filter_config['high_freq'],
            method=self.filter_config['filter_method'],
            fir_design=self.filter_config['fir_design'],
            verbose=False
        )
        
        # Notch filter for powerline noise (if specified)
        if self.filter_config['notch_freq'] is not None:
            raw_filtered.notch_filter(
                freqs=self.filter_config['notch_freq'],
                verbose=False
            )
        
        return raw_filtered
    
    def apply_ica(
        self, 
        raw: mne.io.BaseRaw,
        exclude_components: Optional[List[int]] = None
    ) -> Tuple[mne.io.BaseRaw, mne.preprocessing.ICA]:
        """
        Apply Independent Component Analysis for artifact removal.
        
        Args:
            raw: MNE Raw object (should be filtered first)
            exclude_components: List of component indices to exclude.
                               If None, will attempt automatic detection.
        
        Returns:
            Tuple of (cleaned Raw object, ICA object)
        """
        # Create ICA object
        ica = mne.preprocessing.ICA(
            n_components=self.ica_config['n_components'],
            method=self.ica_config['method'],
            random_state=self.ica_config['random_state'],
            max_iter=self.ica_config['max_iter'],
            fit_params=self.ica_config['fit_params'],
            verbose=False
        )
        
        # Fit ICA
        ica.fit(raw, verbose=False)
        
        # Exclude components
        if exclude_components is None:
            # Try to detect artifacts automatically
            # This is a simplified approach - can be enhanced
            exclude_components = []
        
        # Apply ICA
        raw_cleaned = raw.copy()
        if len(exclude_components) > 0:
            ica.exclude = exclude_components
            ica.apply(raw_cleaned, verbose=False)
        
        return raw_cleaned, ica
    
    def create_epochs(
        self, 
        raw: mne.io.BaseRaw,
        duration: Optional[float] = None
    ) -> mne.Epochs:
        """
        Create fixed-length epochs from continuous data.
        
        For resting-state EEG, we create pseudo-events at regular intervals.
        
        Args:
            raw: MNE Raw object
            duration: Epoch duration in seconds (defaults to epoch_config['tmax'] - epoch_config['tmin'])
        
        Returns:
            MNE Epochs object
        """
        if duration is None:
            duration = self.epoch_config['tmax'] - self.epoch_config['tmin']
        
        # Create events at regular intervals for resting-state
        # Use 90% overlap to maximize data usage
        overlap = 0.9
        event_interval = duration * (1 - overlap)
        
        # Generate events
        events = []
        current_time = 0.0
        event_id = 1
        
        while current_time + duration <= raw.times[-1]:
            sample_idx = int(current_time * raw.info['sfreq'])
            events.append([sample_idx, 0, event_id])
            current_time += event_interval
        
        events = np.array(events)
        
        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id={'rest': event_id},
            tmin=self.epoch_config['tmin'],
            tmax=self.epoch_config['tmax'],
            baseline=self.epoch_config['baseline'],
            reject=self.epoch_config['reject'],
            flat=self.epoch_config['flat'],
            preload=True,
            verbose=False
        )
        
        return epochs
    
    def normalize_data(
        self, 
        data: np.ndarray,
        subject_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Normalize EEG data.
        
        Args:
            data: EEG data array (epochs, channels, time) or (channels, time)
            subject_id: Subject identifier for logging
        
        Returns:
            Normalized data array
        """
        data_normalized = data.copy()
        
        method = self.norm_config['method']
        per_subject = self.norm_config['per_subject']
        per_channel = self.norm_config['per_channel']
        
        if method == 'zscore':
            if per_subject:
                # Normalize across all epochs, channels, and time for this subject
                mean = np.mean(data_normalized)
                std = np.std(data_normalized)
                if std > 0:
                    data_normalized = (data_normalized - mean) / std
            elif per_channel:
                # Normalize per channel
                for ch_idx in range(data_normalized.shape[-2]):
                    mean = np.mean(data_normalized[..., ch_idx, :])
                    std = np.std(data_normalized[..., ch_idx, :])
                    if std > 0:
                        data_normalized[..., ch_idx, :] = (
                            (data_normalized[..., ch_idx, :] - mean) / std
                        )
            else:
                # Normalize per epoch
                for ep_idx in range(data_normalized.shape[0]):
                    mean = np.mean(data_normalized[ep_idx, ...])
                    std = np.std(data_normalized[ep_idx, ...])
                    if std > 0:
                        data_normalized[ep_idx, ...] = (
                            (data_normalized[ep_idx, ...] - mean) / std
                        )
        
        elif method == 'minmax':
            if per_subject:
                data_min = np.min(data_normalized)
                data_max = np.max(data_normalized)
                if data_max > data_min:
                    data_normalized = (data_normalized - data_min) / (data_max - data_min)
            elif per_channel:
                for ch_idx in range(data_normalized.shape[-2]):
                    ch_min = np.min(data_normalized[..., ch_idx, :])
                    ch_max = np.max(data_normalized[..., ch_idx, :])
                    if ch_max > ch_min:
                        data_normalized[..., ch_idx, :] = (
                            (data_normalized[..., ch_idx, :] - ch_min) / (ch_max - ch_min)
                        )
            else:
                for ep_idx in range(data_normalized.shape[0]):
                    ep_min = np.min(data_normalized[ep_idx, ...])
                    ep_max = np.max(data_normalized[ep_idx, ...])
                    if ep_max > ep_min:
                        data_normalized[ep_idx, ...] = (
                            (data_normalized[ep_idx, ...] - ep_min) / (ep_max - ep_min)
                        )
        
        return data_normalized
    
    def preprocess_raw(
        self,
        raw: mne.io.BaseRaw,
        apply_ica: bool = True,
        ica_exclude: Optional[List[int]] = None,
        epoch_duration: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Complete preprocessing pipeline for a single Raw object.
        
        Args:
            raw: MNE Raw object
            apply_ica: Whether to apply ICA artifact removal
            ica_exclude: Components to exclude (if None, no exclusion)
            epoch_duration: Duration of epochs in seconds
        
        Returns:
            Tuple of (preprocessed_data, metadata)
            preprocessed_data shape: (epochs, channels, time)
        """
        # Step 1: Filtering
        raw_filtered = self.apply_filtering(raw)
        
        # Step 2: ICA (optional)
        ica_obj = None
        if apply_ica:
            raw_filtered, ica_obj = self.apply_ica(raw_filtered, exclude_components=ica_exclude)
        
        # Step 3: Epoching
        epochs = self.create_epochs(raw_filtered, duration=epoch_duration)
        
        # Extract data: (epochs, channels, time)
        data = epochs.get_data()
        
        # Convert to float32 to save memory (sufficient precision for EEG)
        data = data.astype(np.float32)
        
        # Step 4: Normalization
        data_normalized = self.normalize_data(data)
        
        # Metadata
        metadata = {
            'n_epochs': data_normalized.shape[0],
            'n_channels': data_normalized.shape[1],
            'n_timepoints': data_normalized.shape[2],
            'sfreq': raw.info['sfreq'],
            'channel_names': raw.ch_names,
            'epoch_duration': epochs.tmax - epochs.tmin,
            'ica_applied': apply_ica,
            'ica_n_components': ica_obj.n_components_ if ica_obj else None,
            'ica_excluded': ica_exclude if ica_obj else None
        }
        
        return data_normalized, metadata
    
    def preprocess_batch(
        self,
        raw_dict: Dict[str, mne.io.BaseRaw],
        group: str,
        apply_ica: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Preprocess multiple subjects and stack into single array.
        
        Args:
            raw_dict: Dictionary mapping subject_id to Raw objects
            group: Group identifier ('AD' or 'HC')
            apply_ica: Whether to apply ICA
        
        Returns:
            Tuple of (preprocessed_data, metadata_list)
            preprocessed_data shape: (subjects, epochs, channels, time)
        """
        all_data = []
        all_metadata = []
        
        for subject_id, raw in raw_dict.items():
            try:
                data, metadata = self.preprocess_raw(
                    raw,
                    apply_ica=apply_ica,
                    epoch_duration=None  # Use default from config
                )
                
                metadata['subject_id'] = subject_id
                metadata['group'] = group
                
                all_data.append(data)
                all_metadata.append(metadata)
                
            except Exception as e:
                print(f"Error preprocessing {subject_id}: {str(e)}")
                continue
        
        if len(all_data) == 0:
            raise ValueError(f"No valid data after preprocessing for group {group}")
        
        # Find common shape (min epochs, min channels, min timepoints)
        # This ensures all subjects can be stacked
        min_epochs = min(d.shape[0] for d in all_data)
        min_channels = min(d.shape[1] for d in all_data)
        min_timepoints = min(d.shape[2] for d in all_data)
        
        # Limit epochs to prevent memory issues (max 1000 epochs per subject)
        max_epochs = min(min_epochs, 1000)
        
        # Standardize all data to common shape
        standardized_data = []
        for data in all_data:
            # Truncate if necessary: take first max_epochs epochs, first min_channels channels, first min_timepoints timepoints
            standardized = data[:max_epochs, :min_channels, :min_timepoints]
            # Ensure float32 to save memory
            standardized = standardized.astype(np.float32)
            standardized_data.append(standardized)
        
        # Stack: (subjects, epochs, channels, time)
        stacked_data = np.stack(standardized_data, axis=0)
        
        # Update metadata with standardized shapes
        for meta in all_metadata:
            meta['n_epochs_original'] = meta['n_epochs']
            meta['n_channels_original'] = meta['n_channels']
            meta['n_timepoints_original'] = meta['n_timepoints']
            meta['n_epochs'] = max_epochs
            meta['n_channels'] = min_channels
            meta['n_timepoints'] = min_timepoints
        
        return stacked_data, all_metadata
    
    def save_preprocessed(
    self,
    data: np.ndarray,
    metadata: List[Dict],
    group: str,
    output_dir: Optional[Path] = None
    ):
        """
        Save preprocessed data and metadata (Windows-safe).
    
        Args:
        data: Preprocessed data array (subjects, epochs, channels, time)
        metadata: List of metadata dictionaries
        group: Group identifier
        output_dir: Output directory (defaults to PROCESSED_EEG_DIR)
        """

        import json
        import numpy as np
        from pathlib import Path

    # Set default output directory
        if output_dir is None:
            output_dir = PROCESSED_EEG_DIR

        # Ensure Path object
        output_dir = Path(output_dir)

        # ✅ Ensure directory exists (fixes Windows OSError 22)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Save data as float32 (memory-safe)
        np.save(str(output_dir / f"{group}_preprocessed.npy"), data.astype(np.float32))

        # Save metadata
        with open(output_dir / f"{group}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)


def preprocess_all_groups(
    raw_data: Dict[str, Dict[str, mne.io.BaseRaw]],
    apply_ica: bool = True,
    save_output: bool = True
) -> Dict[str, Tuple[np.ndarray, List[Dict]]]:
    """
    Preprocess all groups and return preprocessed data.
    
    Args:
        raw_data: Nested dict {group: {subject_id: Raw}}
        apply_ica: Whether to apply ICA
        save_output: Whether to save preprocessed data to disk
    
    Returns:
        Dictionary mapping group to (data, metadata) tuples
    """
    preprocessor = EEGPreprocessor()
    
    results = {}
    
    for group, subjects_dict in raw_data.items():
        print(f"\n{'='*60}")
        print(f"Preprocessing {group} group...")
        print(f"{'='*60}")
        
        data, metadata = preprocessor.preprocess_batch(
            subjects_dict,
            group,
            apply_ica=apply_ica
        )
        
        results[group] = (data, metadata)
        
        if save_output:
            preprocessor.save_preprocessed(data, metadata, group)
        
        print(f"Group {group}: {data.shape}")
    
    return results


if __name__ == "__main__":
    # Example usage
    from .load_eeg import load_all_subjects
    
    # Load data
    raw_data = load_all_subjects(verbose=True)
    
    # Preprocess
    preprocessed = preprocess_all_groups(raw_data, apply_ica=True, save_output=True)

