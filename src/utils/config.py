"""
Configuration module for EEG-Guided Digital Brain Twin Framework.

This module centralizes all configuration parameters for the data pipeline.
All paths and parameters are configurable and documented.

Author: Research Team
Date: 2026
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# PROJECT ROOT AND PATHS
# ============================================================================

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_ROOT = PROJECT_ROOT / "data"
RAW_EEG_DIR = DATA_ROOT / "raw_eeg"
PROCESSED_EEG_DIR = DATA_ROOT / "processed_eeg"
EEG_FEATURES_DIR = DATA_ROOT / "eeg_features"
DRUG_EMBEDDINGS_DIR = DATA_ROOT / "drug_embeddings"

# Source directories
SRC_ROOT = PROJECT_ROOT / "src"
EEG_MODULE_DIR = SRC_ROOT / "eeg"
DRUGS_MODULE_DIR = SRC_ROOT / "drugs"

# ============================================================================
# EEG PROCESSING CONFIGURATION
# ============================================================================

# Supported EEG file formats
SUPPORTED_EEG_FORMATS = ['.edf', '.set', '.fif', '.bdf']

# Group labels
EEG_GROUPS = {
    'AD': 'Alzheimer\'s Disease',
    'HC': 'Healthy Controls'
}

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Filtering parameters
FILTER_CONFIG = {
    'low_freq': 0.5,      # High-pass filter cutoff (Hz)
    'high_freq': 40.0,    # Low-pass filter cutoff (Hz)
    'notch_freq': 50.0,   # Powerline noise frequency (Hz) - set to None if not needed
    'filter_method': 'fir',  # 'fir' or 'iir'
    'fir_design': 'firwin'   # FIR filter design method
}

# ICA parameters
ICA_CONFIG = {
    'n_components': None,  # None = use all channels, or specify number
    'method': 'fastica',   # 'fastica' or 'infomax'
    'random_state': 42,    # For reproducibility
    'max_iter': 1000,      # Maximum iterations
    'fit_params': {}       # Additional parameters for ICA fitting
}

# Epoching parameters
EPOCH_CONFIG = {
    'tmin': 0.0,           # Start time relative to event (seconds)
    'tmax': 4.0,           # End time relative to event (seconds)
    'baseline': None,      # Baseline correction: None or (start, end) tuple
    'reject': None,        # Rejection thresholds: None or dict with channel types
    'flat': None           # Flat channel thresholds: None or dict
}

# Normalization parameters
NORMALIZATION_CONFIG = {
    'method': 'zscore',    # 'zscore' or 'minmax'
    'per_subject': True,   # Normalize per subject (recommended)
    'per_channel': False   # Normalize per channel (usually False)
}

# ============================================================================
# FEATURE EXTRACTION PARAMETERS
# ============================================================================

# Frequency bands (Hz) - Standard EEG bands
FREQUENCY_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 40.0)  # Upper limit based on filter cutoff
}

# Power Spectral Density parameters
# n_per_seg=256 at 500 Hz yields ~20 frequency bins in [0.5, 40] Hz (encoder expects psd_dim=20).
# Internal (ds004504) and BrainLAT pipelines use this so feature dims match.
PSD_CONFIG = {
    'method': 'welch',     # 'welch' or 'multitaper'
    'n_fft': None,         # None = use data length, or specify
    'n_overlap': None,     # None = 50% overlap, or specify
    'n_per_seg': 256,      # Welch segment length: 500/256 Hz resolution -> 20 bins in [0.5, 40] Hz
    'fmin': 0.5,           # Minimum frequency (Hz)
    'fmax': 40.0,          # Maximum frequency (Hz)
    'window': 'hann'       # Window function
}

# Connectivity parameters
CONNECTIVITY_CONFIG = {
    'methods': ['coherence', 'plv'],  # Connectivity measures to compute
    'fmin': 0.5,           # Minimum frequency (Hz)
    'fmax': 40.0,          # Maximum frequency (Hz)
    'sfreq': None,         # Sampling frequency (will be inferred from data)
    'n_fft': None,         # FFT length for connectivity
    'n_overlap': None      # Overlap for connectivity computation
}

# ============================================================================
# DRUG CONFIGURATION
# ============================================================================

# Target drugs for Alzheimer's Disease
TARGET_DRUGS = {
    'donepezil': {
        'name': 'Donepezil',
        'drugbank_id': 'DB00843',
        'chembl_id': 'CHEMBL90',
        'smiles': None  # Will be fetched
    },
    'memantine': {
        'name': 'Memantine',
        'drugbank_id': 'DB01018',
        'chembl_id': 'CHEMBL405',
        'smiles': None  # Will be fetched
    }
}

# ChemBERTa model configuration
CHEMBERTA_CONFIG = {
    'model_name': 'DeepChem/ChemBERTa-77M-MLM',  # HuggingFace model
    'max_length': 512,     # Maximum SMILES sequence length
    'batch_size': 1,       # Batch size for embedding generation
    'device': 'cpu'        # 'cpu' or 'cuda' - will auto-detect GPU if available
}

# ============================================================================
# DATA VALIDATION PARAMETERS
# ============================================================================

# Expected channel names for 10-20 system (standard subset)
STANDARD_10_20_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

# Minimum requirements for valid EEG data
MIN_DATA_REQUIREMENTS = {
    'min_channels': 10,    # Minimum number of channels
    'min_duration': 60.0,  # Minimum recording duration (seconds)
    'min_sampling_rate': 250.0  # Minimum sampling rate (Hz)
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# File naming conventions
OUTPUT_CONFIG = {
    'preprocessed_suffix': '_preprocessed',
    'features_suffix': '_features',
    'metadata_suffix': '_metadata',
    'file_format': 'npy',  # NumPy array format
    'metadata_format': 'json'  # JSON for metadata
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_ROOT,
        RAW_EEG_DIR,
        RAW_EEG_DIR / 'AD',
        RAW_EEG_DIR / 'HC',
        PROCESSED_EEG_DIR,
        EEG_FEATURES_DIR,
        DRUG_EMBEDDINGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return directories

def get_group_path(group: str) -> Path:
    """Get the path to raw EEG data for a specific group.
    
    Args:
        group: Group identifier ('AD' or 'HC')
    
    Returns:
        Path to group's raw EEG directory
    
    Raises:
        ValueError: If group is not recognized
    """
    if group not in EEG_GROUPS:
        raise ValueError(f"Unknown group: {group}. Must be one of {list(EEG_GROUPS.keys())}")
    
    return RAW_EEG_DIR / group

def validate_config():
    """Validate configuration parameters.
    
    Returns:
        bool: True if configuration is valid
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate frequency bands
    bands = list(FREQUENCY_BANDS.values())
    for i, (low, high) in enumerate(bands):
        if low >= high:
            raise ValueError(f"Invalid frequency band {i}: low ({low}) >= high ({high})")
        if i > 0 and low < bands[i-1][1]:
            raise ValueError(f"Frequency bands overlap: {bands[i-1]} and ({low}, {high})")
    
    # Validate filter config
    if FILTER_CONFIG['low_freq'] >= FILTER_CONFIG['high_freq']:
        raise ValueError("Filter low_freq must be < high_freq")
    
    # Validate epoch config
    if EPOCH_CONFIG['tmin'] >= EPOCH_CONFIG['tmax']:
        raise ValueError("Epoch tmin must be < tmax")
    
    return True

# Initialize directories on import
ensure_directories()



