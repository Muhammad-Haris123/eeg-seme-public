"""Utility modules for EEG-Guided Digital Brain Twin Framework."""

from .config import (
    PROJECT_ROOT,
    DATA_ROOT,
    RAW_EEG_DIR,
    PROCESSED_EEG_DIR,
    EEG_FEATURES_DIR,
    DRUG_EMBEDDINGS_DIR,
    ensure_directories,
    get_group_path,
    validate_config
)

__all__ = [
    'PROJECT_ROOT',
    'DATA_ROOT',
    'RAW_EEG_DIR',
    'PROCESSED_EEG_DIR',
    'EEG_FEATURES_DIR',
    'DRUG_EMBEDDINGS_DIR',
    'ensure_directories',
    'get_group_path',
    'validate_config'
]



