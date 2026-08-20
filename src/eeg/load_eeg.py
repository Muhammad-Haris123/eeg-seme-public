"""
EEG Data Loading Module.

This module handles loading EEG data from various formats (.edf, .set, .fif)
compatible with OpenNeuro and Figshare datasets.

ASSUMPTIONS:
- All EEG data is RESTING-STATE recordings
- Data uses the international 10-20 electrode system
- Data is organized by group (AD/HC) in separate folders
- No automatic dataset downloading - data must be manually placed

Author: Research Team
Date: 2026
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np

from ..utils.config import (
    RAW_EEG_DIR,
    SUPPORTED_EEG_FORMATS,
    EEG_GROUPS,
    get_group_path,
    STANDARD_10_20_CHANNELS
)

# Suppress MNE warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class EEGLoader:
    """
    Loader for EEG data from multiple formats.
    
    Supports:
    - .edf (European Data Format)
    - .set/.fdt (EEGLAB format)
    - .fif (MNE-Python format)
    - .bdf (Biosemi format)
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize EEG loader.
        
        Args:
            data_dir: Path to raw EEG data directory. If None, uses config default.
        """
        self.data_dir = data_dir if data_dir else RAW_EEG_DIR
        self.supported_formats = SUPPORTED_EEG_FORMATS
        
    def find_eeg_files(self, group: str) -> List[Path]:
        """
        Find all EEG files for a given group.
        
        Args:
            group: Group identifier ('AD' or 'HC')
        
        Returns:
            List of paths to EEG files
        
        Raises:
            ValueError: If group is invalid or directory doesn't exist
        """
        if group not in EEG_GROUPS:
            raise ValueError(f"Unknown group: {group}. Must be one of {list(EEG_GROUPS.keys())}")
        
        group_path = get_group_path(group)
        
        if not group_path.exists():
            raise FileNotFoundError(f"Group directory not found: {group_path}")
        
        eeg_files = []
        for ext in self.supported_formats:
            # Search in root directory
            eeg_files.extend(list(group_path.glob(f"*{ext}")))
            eeg_files.extend(list(group_path.glob(f"*{ext.upper()}")))
            # Search recursively in subdirectories (e.g., sub-XXX/eeg/*.set)
            eeg_files.extend(list(group_path.glob(f"**/*{ext}")))
            eeg_files.extend(list(group_path.glob(f"**/*{ext.upper()}")))
        
        # Remove duplicates and sort
        eeg_files = sorted(list(set(eeg_files)))
        
        return eeg_files
    
    def detect_file_format(self, file_path: Path) -> str:
        """
        Detect EEG file format from extension.
        
        Args:
            file_path: Path to EEG file
        
        Returns:
            File format string ('edf', 'set', 'fif', 'bdf')
        
        Raises:
            ValueError: If format is not supported
        """
        ext = file_path.suffix.lower()
        
        if ext == '.edf' or ext == '.bdf':
            return 'edf'
        elif ext == '.set':
            return 'set'
        elif ext == '.fif':
            return 'fif'
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def load_raw_eeg(self, file_path: Path, preload: bool = True) -> mne.io.BaseRaw:
        """
        Load raw EEG data from file.
        
        Args:
            file_path: Path to EEG file
            preload: Whether to preload data into memory
        
        Returns:
            MNE Raw object
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"EEG file not found: {file_path}")
        
        file_format = self.detect_file_format(file_path)
        
        try:
            if file_format == 'edf':
                raw = mne.io.read_raw_edf(
                    file_path,
                    preload=preload,
                    verbose=False
                )
            elif file_format == 'set':
                # EEGLAB .set files may have associated .fdt files
                raw = mne.io.read_raw_eeglab(
                    file_path,
                    preload=preload,
                    verbose=False
                )
            elif file_format == 'fif':
                raw = mne.io.read_raw_fif(
                    file_path,
                    preload=preload,
                    verbose=False
                )
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            # Set channel types if not already set
            if raw.ch_names:
                # Try to infer channel types
                ch_types = []
                for ch_name in raw.ch_names:
                    ch_name_lower = ch_name.lower()
                    if 'eeg' in ch_name_lower or any(
                        std_ch.lower() in ch_name_lower 
                        for std_ch in STANDARD_10_20_CHANNELS
                    ):
                        ch_types.append('eeg')
                    elif 'eog' in ch_name_lower or 'eye' in ch_name_lower:
                        ch_types.append('eog')
                    elif 'ecg' in ch_name_lower or 'heart' in ch_name_lower:
                        ch_types.append('ecg')
                    elif 'emg' in ch_name_lower:
                        ch_types.append('emg')
                    else:
                        ch_types.append('eeg')  # Default to EEG
                
                # Only set if we have the right number
                if len(ch_types) == len(raw.ch_names):
                    raw.set_channel_types(dict(zip(raw.ch_names, ch_types)))
            
            return raw
            
        except Exception as e:
            raise RuntimeError(f"Error loading EEG file {file_path}: {str(e)}")
    
    def load_group_data(
        self, 
        group: str, 
        max_files: Optional[int] = None,
        verbose: bool = False
    ) -> Dict[str, mne.io.BaseRaw]:
        """
        Load all EEG files for a group.
        
        Args:
            group: Group identifier ('AD' or 'HC')
            max_files: Maximum number of files to load (None = all)
            verbose: Whether to print progress
        
        Returns:
            Dictionary mapping file names to Raw objects
        """
        eeg_files = self.find_eeg_files(group)
        
        if max_files is not None:
            eeg_files = eeg_files[:max_files]
        
        if len(eeg_files) == 0:
            if verbose:
                print(f"Warning: No EEG files found for group {group}")
            return {}
        
        raw_data = {}
        for file_path in eeg_files:
            if verbose:
                print(f"Loading {file_path.name}...")
            
            try:
                raw = self.load_raw_eeg(file_path, preload=True)
                raw_data[file_path.stem] = raw
            except Exception as e:
                if verbose:
                    print(f"Error loading {file_path.name}: {str(e)}")
                continue
        
        return raw_data
    
    def get_data_info(self, raw):
        """
        Extract metadata information from Raw EEG object safely.
    Handles missing subject_info without crashing.
        """

    # SAFE subject info handling
        subject_info = raw.info.get('subject_info')
        subject_id = 'unknown'

        if isinstance(subject_info, dict):
            subject_id = subject_info.get('his_id', 'unknown')

        info = {
        'n_channels': len(raw.ch_names),
        'n_samples': raw.n_times,
        'sfreq': raw.info['sfreq'],
        'duration': raw.n_times / raw.info['sfreq'],
        'channel_names': raw.ch_names,
        'channel_types': raw.get_channel_types(),
        'subject_id': subject_id,
        'recording_date': str(raw.info.get('meas_date', 'unknown'))
    }

        return info

    
    def validate_eeg_data(
        self, 
        raw: mne.io.BaseRaw,
        min_channels: int = 10,
        min_duration: float = 60.0,
        min_sfreq: float = 250.0
    ) -> Tuple[bool, List[str]]:
        """
        Validate EEG data against minimum requirements.
        
        Args:
            raw: MNE Raw object
            min_channels: Minimum number of channels required
            min_duration: Minimum recording duration in seconds
            min_sfreq: Minimum sampling frequency in Hz
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings_list = []
        
        # Check number of channels
        n_channels = len(raw.ch_names)
        if n_channels < min_channels:
            warnings_list.append(
                f"Insufficient channels: {n_channels} < {min_channels}"
            )
        
        # Check duration
        duration = raw.n_times / raw.info['sfreq']
        if duration < min_duration:
            warnings_list.append(
                f"Short recording: {duration:.1f}s < {min_duration}s"
            )
        
        # Check sampling frequency
        sfreq = raw.info['sfreq']
        if sfreq < min_sfreq:
            warnings_list.append(
                f"Low sampling rate: {sfreq:.1f} Hz < {min_sfreq} Hz"
            )
        
        # Check for EEG channels
        eeg_channels = [ch for ch in raw.ch_names if 'eeg' in raw.get_channel_types()]
        if len(eeg_channels) == 0:
            warnings_list.append("No EEG channels detected")
        
        is_valid = len(warnings_list) == 0
        
        return is_valid, warnings_list


def load_all_subjects(
    data_dir: Optional[Path] = None,
    max_per_group: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, mne.io.BaseRaw]]:
    """
    Convenience function to load all subjects from both groups.
    
    Args:
        data_dir: Path to raw EEG data directory
        max_per_group: Maximum files per group to load
        verbose: Whether to print progress
    
    Returns:
        Nested dictionary: {group: {subject_id: Raw}}
    """
    loader = EEGLoader(data_dir)
    
    all_data = {}
    for group in EEG_GROUPS.keys():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Loading {EEG_GROUPS[group]} ({group}) data...")
            print(f"{'='*60}")
        
        group_data = loader.load_group_data(group, max_files=max_per_group, verbose=verbose)
        all_data[group] = group_data
        
        if verbose:
            print(f"Loaded {len(group_data)} subjects")
    
    return all_data


if __name__ == "__main__":
    # Example usage
    loader = EEGLoader()
    
    # Find files
    ad_files = loader.find_eeg_files('AD')
    hc_files = loader.find_eeg_files('HC')
    
    print(f"Found {len(ad_files)} AD files")
    print(f"Found {len(hc_files)} HC files")
    
    # Load all data
    all_data = load_all_subjects(verbose=True)

