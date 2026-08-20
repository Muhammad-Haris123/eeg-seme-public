"""
Configuration for Phase 2: Digital Twin CVAE Model.

All hyperparameters and model settings are centralized here.

Author: Research Team
Date: 2026
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths (from Phase 1)
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_EEG_DIR = DATA_ROOT / "processed_eeg"
EEG_FEATURES_DIR = DATA_ROOT / "eeg_features"
DRUG_EMBEDDINGS_DIR = DATA_ROOT / "drug_embeddings"

# Output paths (Phase 2)
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
CHECKPOINTS_5FOLD_DIR = MODELS_DIR / "checkpoints_5fold"
LATENTS_DIR = MODELS_DIR / "latents"
SIMULATIONS_DIR = MODELS_DIR / "simulations"
LATENTS_FULL_DIR = MODELS_DIR / "latents_full"
SIMULATIONS_FULL_DIR = MODELS_DIR / "simulations_full"
LOGS_DIR = MODELS_DIR / "logs"
EVALUATION_DIR = MODELS_DIR / "evaluation"
XAI_DIR = EVALUATION_DIR / "xai"

# Model architecture hyperparameters
MODEL_CONFIG = {
    # EEG Encoder
    'eeg_encoder': {
        'psd_dim': 20,
        'n_channels': 19,
        'n_bands': 5,
        'eeg_latent_dim': 128,
        'hidden_dims': [256, 512]
    },
    
    # Drug Encoder
    'drug_encoder': {
        'input_dim': 384,  # ChemBERTa dimension
        'drug_latent_dim': 64,
        'hidden_dims': [256, 128]
    },
    
    # Fusion Module
    'fusion': {
        'fused_dim': 256,
        'use_attention': False  # Set to True for cross-attention (future)
    },
    
    # CVAE
    'cvae': {
        'latent_dim': 128,
        'hidden_dims': [512, 256]
    }
}

# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'beta_kl': 0.01,  # KL divergence weight
    'beta_condition': 0.1,  # Optional condition loss weight
    'use_condition_loss': False,  # Set to True to enable condition consistency loss
    'save_every': 10,  # Save checkpoint every N epochs
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
}

# Data configuration
DATA_CONFIG = {
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'random_seed': 42,
    'shuffle': True
}

# Inference configuration
INFERENCE_CONFIG = {
    'num_samples': 10,  # Number of samples per condition for simulation
    'temperature': 1.0  # Sampling temperature (for future use)
}

# Disease labels
DISEASE_LABELS = {
    'AD': 1,
    'HC': 0
}

# Drug IDs
DRUG_IDS = {
    'donepezil': 0,
    'memantine': 1
}

def ensure_directories():
    """Create all necessary directories."""
    directories = [
        MODELS_DIR,
        CHECKPOINTS_DIR,
        CHECKPOINTS_5FOLD_DIR,
        LATENTS_DIR,
        SIMULATIONS_DIR,
        LATENTS_FULL_DIR,
        SIMULATIONS_FULL_DIR,
        LOGS_DIR,
        EVALUATION_DIR,
        EVALUATION_DIR / "figures",
        XAI_DIR,
        XAI_DIR / "figures",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return directories

# Initialize directories
ensure_directories()

# ================================================
# Model architecture variants (added for paper upgrade)
# ================================================
MODEL_VARIANTS = {
    'baseline_mlp': {
        'use_gnn_encoder': False,
        'use_enhanced_drug': False,
        'use_cross_attention': False,
        'use_constrained_loss': False,
        'description': 'Original MLP model (FYP submission)'
    },
    'constrained_mlp': {
        'use_gnn_encoder': False,
        'use_enhanced_drug': False,
        'use_cross_attention': False,
        'use_constrained_loss': True,
        'description': 'MLP + biophysical constraints (Phase 5 result)'
    },
    'gnn_attention': {
        'use_gnn_encoder': True,
        'use_enhanced_drug': True,
        'use_cross_attention': True,
        'use_constrained_loss': True,
        'description': 'Full upgrade: GNN + enhanced drug + cross-attention + constraints'
    },
}

