"""
Inference Script for Phase 2: Generate Simulated Post-Drug EEG.

Generates:
- Latent representations for all subjects
- Simulated EEG features:
  - Pre-drug (baseline)
  - Post-Donepezil
  - Post-Memantine

Author: Research Team
Date: 2026
"""

import sys
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from .train_phase2 import DigitalTwinModel
from .data_loader import create_data_loaders, load_drug_embeddings, get_split_info, create_all_subjects_loader
from .config import (
    MODEL_CONFIG,
    CHECKPOINTS_DIR,
    LATENTS_DIR,
    SIMULATIONS_DIR,
    LATENTS_FULL_DIR,
    SIMULATIONS_FULL_DIR,
    DISEASE_LABELS,
    DRUG_IDS
)


def load_trained_model(checkpoint_path: Path, device: torch.device) -> DigitalTwinModel:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Initialize model
    model = DigitalTwinModel(config=MODEL_CONFIG).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    return model


def extract_latent_representations(
    model: DigitalTwinModel,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> dict:
    """
    Extract latent representations for all subjects.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device
    
    Returns:
        Dictionary with latent representations
    """
    model.eval()
    
    all_eeg_latents = []
    all_drug_latents = []
    all_cvae_mus = []
    all_cvae_logvars = []
    all_cvae_zs = []
    all_disease_labels = []
    all_subject_indices = []
    all_drug_ids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting latents"):
            # Move to device
            psd = batch['psd'].to(device)
            band_powers = batch['band_powers'].to(device)
            coherence = batch['coherence'].to(device)
            plv = batch['plv'].to(device)
            drug_embedding = batch['drug_embedding'].to(device)
            disease_label = batch['disease_label'].to(device)
            
            # Forward pass with latent return
            reconstructed, mu, logvar, z, eeg_latent, drug_latent = model(
                psd, band_powers, coherence, plv,
                drug_embedding, disease_label,
                return_latent=True
            )
            
            # Collect latents
            all_eeg_latents.append(eeg_latent.cpu().numpy())
            all_drug_latents.append(drug_latent.cpu().numpy())
            all_cvae_mus.append(mu.cpu().numpy())
            all_cvae_logvars.append(logvar.cpu().numpy())
            all_cvae_zs.append(z.cpu().numpy())
            all_disease_labels.append(disease_label.cpu().numpy())
            all_subject_indices.append(batch['subject_idx'].numpy())
            all_drug_ids.append(batch['drug_id'].numpy())
    
    # Concatenate all
    latents = {
        'eeg_latent': np.concatenate(all_eeg_latents, axis=0),
        'drug_latent': np.concatenate(all_drug_latents, axis=0),
        'cvae_mu': np.concatenate(all_cvae_mus, axis=0),
        'cvae_logvar': np.concatenate(all_cvae_logvars, axis=0),
        'cvae_z': np.concatenate(all_cvae_zs, axis=0),
        'disease_labels': np.concatenate(all_disease_labels, axis=0),
        'subject_indices': np.concatenate(all_subject_indices, axis=0),
        'drug_ids': np.concatenate(all_drug_ids, axis=0)
    }
    
    return latents


def simulate_post_drug_eeg(
    model: DigitalTwinModel,
    eeg_features: dict,
    drug_embeddings: dict,
    disease_label: int,
    device: torch.device,
    num_samples: int = 10
) -> dict:
    """
    Simulate post-drug EEG features.
    
    Args:
        model: Trained model
        eeg_features: Dictionary with psd, band_powers, coherence, plv
        drug_embeddings: Dictionary mapping drug names to embeddings
        disease_label: Disease label (AD=1, HC=0)
        device: Device
        num_samples: Number of samples to generate
    
    Returns:
        Dictionary with simulated features for each drug condition
    """
    model.eval()
    
    # Prepare EEG features
    psd = torch.FloatTensor(eeg_features['psd']).unsqueeze(0).to(device)
    band_powers = torch.FloatTensor(eeg_features['band_powers']).unsqueeze(0).to(device)
    coherence = torch.FloatTensor(eeg_features['coherence']).unsqueeze(0).to(device)
    plv = torch.FloatTensor(eeg_features['plv']).unsqueeze(0).to(device)
    disease_label_tensor = torch.tensor([disease_label], dtype=torch.float32).to(device)
    
    # Get EEG latent (baseline)
    with torch.no_grad():
        eeg_latent = model.eeg_encoder(psd, band_powers, coherence, plv)
    
    # Get CVAE latent (from baseline)
    no_drug_embedding = torch.zeros(1, 384).to(device)
    drug_latent_baseline = model.drug_encoder(no_drug_embedding)
    fused_baseline = model.fusion(eeg_latent, drug_latent_baseline, disease_label_tensor)
    mu, logvar = model.cvae.encode(fused_baseline)
    z_baseline = model.cvae.reparameterize(mu, logvar)
    
    simulations = {}
    
    # Simulate for each drug
    for drug_name, drug_embedding_array in drug_embeddings.items():
        drug_embedding = torch.FloatTensor(drug_embedding_array).unsqueeze(0).to(device)
        drug_latent = model.drug_encoder(drug_embedding)
        
        # Condition: drug_latent + disease_label
        condition = torch.cat([drug_latent, disease_label_tensor.unsqueeze(1)], dim=1)
        
        # Sample from latent
        z_samples = []
        for _ in range(num_samples):
            z_sample = model.cvae.reparameterize(mu, logvar)
            z_samples.append(z_sample)
        
        z_samples = torch.cat(z_samples, dim=0)
        condition_expanded = condition.repeat(num_samples, 1)
        
        # Decode
        with torch.no_grad():
            simulated_features = model.cvae.decode(z_samples, condition_expanded)
        
        simulations[drug_name] = simulated_features.cpu().numpy()
    
    # Also simulate baseline (no drug)
    no_drug_condition = torch.cat([drug_latent_baseline, disease_label_tensor.unsqueeze(1)], dim=1)
    z_baseline_samples = z_baseline.repeat(num_samples, 1)
    no_drug_condition_expanded = no_drug_condition.repeat(num_samples, 1)
    
    with torch.no_grad():
        baseline_features = model.cvae.decode(z_baseline_samples, no_drug_condition_expanded)
    
    simulations['baseline'] = baseline_features.cpu().numpy()
    
    return simulations


def generate_all_simulations(
    model: DigitalTwinModel,
    test_loader: torch.utils.data.DataLoader,
    drug_embeddings: dict,
    device: torch.device,
    num_samples: int = 10
) -> dict:
    """
    Generate simulations for all test subjects.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        drug_embeddings: Drug embeddings dictionary
        device: Device
        num_samples: Number of samples per condition
    
    Returns:
        Dictionary with all simulations
    """
    model.eval()
    
    all_simulations = {
        'baseline': [],
        'donepezil': [],
        'memantine': []
    }
    
    all_subject_info = []
    seen_subjects = set()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating simulations"):
            subject_indices = batch['subject_idx'].numpy()
            unique_subjects = np.unique(subject_indices)
            for subj_idx in unique_subjects:
                if subj_idx in seen_subjects:
                    continue
                seen_subjects.add(subj_idx)
                # Get first sample for this subject (baseline)
                mask = subject_indices == subj_idx
                idx = np.where(mask)[0][0]
                
                # Extract features
                eeg_features = {
                    'psd': batch['psd'][idx].numpy(),
                    'band_powers': batch['band_powers'][idx].numpy(),
                    'coherence': batch['coherence'][idx].numpy(),
                    'plv': batch['plv'][idx].numpy()
                }
                
                disease_label = batch['disease_label'][idx].item()
                
                # Simulate
                simulations = simulate_post_drug_eeg(
                    model, eeg_features, drug_embeddings,
                    disease_label, device, num_samples
                )
                
                all_simulations['baseline'].append(simulations['baseline'])
                all_simulations['donepezil'].append(simulations.get('donepezil', simulations.get('baseline')))
                all_simulations['memantine'].append(simulations.get('memantine', simulations.get('baseline')))
                
                all_subject_info.append({
                    'subject_idx': int(subj_idx),
                    'disease_label': int(disease_label),
                    'group': 'AD' if disease_label == 1 else 'HC'
                })
    
    # Convert to numpy arrays
    for key in all_simulations:
        all_simulations[key] = np.array(all_simulations[key])
    
    return {
        'simulations': all_simulations,
        'subject_info': all_subject_info
    }


def main():
    """Main inference function."""
    print("="*70)
    print("PHASE 2: INFERENCE - GENERATING SIMULATED POST-DRUG EEG")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    checkpoint_path = CHECKPOINTS_DIR / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = CHECKPOINTS_DIR / "checkpoint_latest.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINTS_DIR}")
    
    model = load_trained_model(checkpoint_path, device)
    
    # Load drug embeddings
    print("\nLoading drug embeddings...")
    drug_embeddings = load_drug_embeddings()
    print(f"Loaded embeddings for: {list(drug_embeddings.keys())}")
    
    # Create data loaders (subject-level split; test subjects only for primary output)
    print("\nCreating data loaders (subject-level split)...")
    _, _, test_loader = create_data_loaders(
        batch_size=16,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        random_seed=42,
        shuffle=False,
        split_by_subject=True,
    )
    split_info = get_split_info(train_split=0.8, val_split=0.1, test_split=0.1, random_seed=42)
    print(f"  Test subjects: {split_info['n_test_subjects']} (AD: {split_info['n_AD_test']}, HC: {split_info['n_HC_test']})")

    # Extract latent representations (test subjects only)
    print("\nExtracting latent representations (test set only)...")
    latents = extract_latent_representations(model, test_loader, device)

    # Save latents
    LATENTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nSaving latent representations...")
    for key, value in latents.items():
        save_path = LATENTS_DIR / f"{key}.npy"
        np.save(save_path, value)
        print(f"  Saved {key}: {value.shape} -> {save_path}")

    # Save latent metadata (include split info)
    latent_metadata = {
        'shapes': {key: list(value.shape) for key, value in latents.items()},
        'description': {
            'eeg_latent': 'EEG encoder output',
            'drug_latent': 'Drug encoder output',
            'cvae_mu': 'CVAE latent mean',
            'cvae_logvar': 'CVAE latent log-variance',
            'cvae_z': 'CVAE sampled latent',
            'disease_labels': 'Disease labels (AD=1, HC=0)',
            'subject_indices': 'Global subject IDs (test set only)',
            'drug_ids': 'Drug IDs (-1=baseline, 0=donepezil, 1=memantine)'
        },
        **{k: v for k, v in split_info.items() if k != 'test_subject_global_ids'},
    }
    with open(LATENTS_DIR / "latent_metadata.json", 'w') as f:
        json.dump(latent_metadata, f, indent=2)

    # Generate simulations (test subjects only)
    print("\nGenerating simulated post-drug EEG (test set only)...")
    simulations = generate_all_simulations(
        model, test_loader, drug_embeddings, device, num_samples=10
    )

    # Save simulations
    SIMULATIONS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nSaving simulations...")
    for condition, sim_data in simulations['simulations'].items():
        save_path = SIMULATIONS_DIR / f"simulated_{condition}.npy"
        np.save(save_path, sim_data)
        print(f"  Saved {condition}: {sim_data.shape} -> {save_path}")

    # Save simulation metadata (include split info)
    sim_metadata = {
        'conditions': list(simulations['simulations'].keys()),
        'shapes': {k: list(v.shape) for k, v in simulations['simulations'].items()},
        'subject_info': simulations['subject_info'],
        'num_samples_per_condition': 10,
        **{k: v for k, v in split_info.items() if k != 'test_subject_global_ids'},
    }
    with open(SIMULATIONS_DIR / "simulation_metadata.json", 'w') as f:
        json.dump(sim_metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE (test subjects only)")
    print("="*70)
    print(f"Latents saved to: {LATENTS_DIR}")
    print(f"Simulations saved to: {SIMULATIONS_DIR}")
    print("\nGenerated simulations for:")
    print("  - Baseline (pre-drug)")
    print("  - Post-Donepezil")
    print("  - Post-Memantine")

    # Optional: full-cohort inference (all 66 subjects)
    if "--all-subjects" in sys.argv:
        print("\n" + "-"*70)
        print("Running full-cohort inference (all subjects)...")
        all_loader = create_all_subjects_loader(batch_size=16)
        latents_full = extract_latent_representations(model, all_loader, device)
        LATENTS_FULL_DIR.mkdir(parents=True, exist_ok=True)
        for key, value in latents_full.items():
            np.save(LATENTS_FULL_DIR / f"{key}.npy", value)
        meta_full = {
            'shapes': {k: list(v.shape) for k, v in latents_full.items()},
            'full_cohort': True,
            'n_subjects': latents_full['cvae_mu'].shape[0],
        }
        with open(LATENTS_FULL_DIR / "latent_metadata.json", 'w') as f:
            json.dump(meta_full, f, indent=2)
        sims_full = generate_all_simulations(model, all_loader, drug_embeddings, device, num_samples=10)
        SIMULATIONS_FULL_DIR.mkdir(parents=True, exist_ok=True)
        for cond, data in sims_full['simulations'].items():
            np.save(SIMULATIONS_FULL_DIR / f"simulated_{cond}.npy", data)
        with open(SIMULATIONS_FULL_DIR / "simulation_metadata.json", 'w') as f:
            json.dump({
                'conditions': list(sims_full['simulations'].keys()),
                'shapes': {k: list(v.shape) for k, v in sims_full['simulations'].items()},
                'subject_info': sims_full['subject_info'],
                'num_samples_per_condition': 10,
                'full_cohort': True,
            }, f, indent=2)
        print(f"Full-cohort outputs saved to {LATENTS_FULL_DIR} and {SIMULATIONS_FULL_DIR}")


if __name__ == "__main__":
    main()

