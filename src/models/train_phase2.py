"""
Training Script for Phase 2: Digital Twin CVAE.

Trains the complete model architecture:
1. EEG Encoder
2. Drug Encoder
3. Fusion Module
4. Conditional VAE

Saves:
- Model checkpoints
- Training loss curves
- Latent representations

Author: Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from .eeg_encoder import EEGEncoder
from .drug_encoder import DrugEncoder
from .fusion import FusionModule
from .cvae import CVAE
from .data_loader import create_data_loaders, load_drug_embeddings
from .losses import compute_total_loss
from .config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    DATA_CONFIG,
    CHECKPOINTS_DIR,
    LATENTS_DIR,
    LOGS_DIR
)


class DigitalTwinModel(nn.Module):
    """
    Complete Digital Twin Model.
    
    Combines all components:
    - EEG Encoder
    - Drug Encoder
    - Fusion Module
    - CVAE
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize complete model.
        
        Args:
            config: Model configuration dictionary
        """
        super(DigitalTwinModel, self).__init__()
        
        if config is None:
            config = MODEL_CONFIG
        
        # EEG Encoder
        self.eeg_encoder = EEGEncoder(**config['eeg_encoder'])
        
        # Drug Encoder
        self.drug_encoder = DrugEncoder(**config['drug_encoder'])
        
        # Fusion Module
        eeg_latent_dim = config['eeg_encoder']['eeg_latent_dim']
        drug_latent_dim = config['drug_encoder']['drug_latent_dim']
        self.fusion = FusionModule(
            eeg_latent_dim=eeg_latent_dim,
            drug_latent_dim=drug_latent_dim,
            fused_dim=config['fusion']['fused_dim'],
            use_attention=config['fusion']['use_attention']
        )
        
        # CVAE
        # Calculate output dimension (same as EEG feature dimension)
        psd_dim = config['eeg_encoder']['psd_dim']
        n_channels = config['eeg_encoder']['n_channels']
        n_bands = config['eeg_encoder']['n_bands']
        
        # PSD features
        psd_features = n_channels * psd_dim
        # Band powers
        band_power_features = n_channels * n_bands
        # Connectivity (upper triangle per band)
        connectivity_features = n_bands * (n_channels * (n_channels - 1) // 2) * 2  # coherence + PLV
        
        output_dim = psd_features + band_power_features + connectivity_features
        
        condition_dim = drug_latent_dim + 1  # drug_latent + disease_label
        
        self.cvae = CVAE(
            fused_dim=config['fusion']['fused_dim'],
            latent_dim=config['cvae']['latent_dim'],
            output_dim=output_dim,
            condition_dim=condition_dim,
            hidden_dims=config['cvae']['hidden_dims']
        )
    
    def forward(
        self,
        psd: torch.Tensor,
        band_powers: torch.Tensor,
        coherence: torch.Tensor,
        plv: torch.Tensor,
        drug_embedding: torch.Tensor,
        disease_label: torch.Tensor,
        return_latent: bool = False
    ) -> tuple:
        """
        Forward pass through complete model.
        
        Args:
            psd: PSD features (batch, channels, frequencies)
            band_powers: Band power features (batch, channels, bands)
            coherence: Coherence matrices (batch, bands, channels, channels)
            plv: PLV matrices (batch, bands, channels, channels)
            drug_embedding: Drug embeddings (batch, 384)
            disease_label: Disease labels (batch, 1)
            return_latent: Whether to return latent distribution
        
        Returns:
            If return_latent: (reconstructed, mu, logvar, z, eeg_latent, drug_latent)
            Else: (reconstructed, mu, logvar, eeg_latent, drug_latent)
        """
        # Encode EEG
        eeg_latent = self.eeg_encoder(psd, band_powers, coherence, plv)
        
        # Encode drug
        drug_latent = self.drug_encoder(drug_embedding)
        
        # Fuse
        fused_repr = self.fusion(eeg_latent, drug_latent, disease_label)
        
        # Condition: drug_latent + disease_label
        condition = torch.cat([drug_latent, disease_label.unsqueeze(1) if disease_label.dim() == 1 else disease_label], dim=1)
        
        # CVAE forward
        if return_latent:
            reconstructed, mu, logvar, z = self.cvae(fused_repr, condition, return_latent=True)
            return reconstructed, mu, logvar, z, eeg_latent, drug_latent
        else:
            reconstructed, mu, logvar = self.cvae(fused_repr, condition, return_latent=False)
            return reconstructed, mu, logvar, eeg_latent, drug_latent


def prepare_target_features(
    psd: torch.Tensor,
    band_powers: torch.Tensor,
    coherence: torch.Tensor,
    plv: torch.Tensor
) -> torch.Tensor:
    """
    Prepare target features for reconstruction loss.
    
    Flattens all EEG features into a single vector (same format as CVAE output).
    
    Args:
        psd: (batch, channels, frequencies)
        band_powers: (batch, channels, bands)
        coherence: (batch, bands, channels, channels)
        plv: (batch, bands, channels, channels)
    
    Returns:
        Flattened features (batch, feature_dim)
    """
    batch_size = psd.shape[0]
    
    # Flatten PSD
    psd_flat = psd.view(batch_size, -1)
    
    # Flatten band powers
    band_powers_flat = band_powers.view(batch_size, -1)
    
    # Extract connectivity features (upper triangle)
    n_bands = coherence.shape[1]
    n_channels = coherence.shape[2]
    triu_indices = torch.triu_indices(n_channels, n_channels, offset=1)
    
    connectivity_features = []
    for band_idx in range(n_bands):
        coh_triu = coherence[:, band_idx, triu_indices[0], triu_indices[1]]
        plv_triu = plv[:, band_idx, triu_indices[0], triu_indices[1]]
        connectivity_features.append(torch.cat([coh_triu, plv_triu], dim=1))
    
    connectivity_flat = torch.cat(connectivity_features, dim=1)
    
    # Concatenate all
    target = torch.cat([psd_flat, band_powers_flat, connectivity_flat], dim=1)
    
    return target


def train_epoch(
    model: DigitalTwinModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: dict
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    loss_recon = 0.0
    loss_kl = 0.0
    loss_condition = 0.0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move to device
        psd = batch['psd'].to(device)
        band_powers = batch['band_powers'].to(device)
        coherence = batch['coherence'].to(device)
        plv = batch['plv'].to(device)
        drug_embedding = batch['drug_embedding'].to(device)
        disease_label = batch['disease_label'].to(device)
        
        # Prepare target
        target = prepare_target_features(psd, band_powers, coherence, plv)
        
        # Forward pass
        if config['use_condition_loss']:
            # Need latent z for condition loss
            reconstructed, mu, logvar, z, eeg_latent, drug_latent = model(
                psd, band_powers, coherence, plv,
                drug_embedding, disease_label,
                return_latent=True
            )
        else:
            reconstructed, mu, logvar, eeg_latent, drug_latent = model(
                psd, band_powers, coherence, plv,
                drug_embedding, disease_label,
                return_latent=False
            )
            z = None
        
        # Compute loss
        losses = compute_total_loss(
            reconstructed=reconstructed,
            target=target,
            mu=mu,
            logvar=logvar,
            beta_kl=config['beta_kl'],
            beta_condition=config['beta_condition'],
            use_condition_loss=config['use_condition_loss'],
            latent=z,
            disease_labels=disease_label
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += losses['total'].item()
        loss_recon += losses['reconstruction'].item()
        loss_kl += losses['kl'].item()
        if 'condition' in losses:
            loss_condition += losses['condition'].item()
    
    n_batches = len(train_loader)
    
    return {
        'total': total_loss / n_batches,
        'reconstruction': loss_recon / n_batches,
        'kl': loss_kl / n_batches,
        'condition': loss_condition / n_batches if config['use_condition_loss'] else 0.0
    }


def validate(
    model: DigitalTwinModel,
    val_loader: DataLoader,
    device: torch.device,
    config: dict
) -> dict:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    loss_recon = 0.0
    loss_kl = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Move to device
            psd = batch['psd'].to(device)
            band_powers = batch['band_powers'].to(device)
            coherence = batch['coherence'].to(device)
            plv = batch['plv'].to(device)
            drug_embedding = batch['drug_embedding'].to(device)
            disease_label = batch['disease_label'].to(device)
            
            # Prepare target
            target = prepare_target_features(psd, band_powers, coherence, plv)
            
            # Forward pass
            reconstructed, mu, logvar, _, _ = model(
                psd, band_powers, coherence, plv,
                drug_embedding, disease_label,
                return_latent=False
            )
            
            # Compute loss
            losses = compute_total_loss(
                reconstructed=reconstructed,
                target=target,
                mu=mu,
                logvar=logvar,
                beta_kl=config['beta_kl'],
                beta_condition=0.0,  # No condition loss in validation
                use_condition_loss=False
            )
            
            total_loss += losses['total'].item()
            loss_recon += losses['reconstruction'].item()
            loss_kl += losses['kl'].item()
    
    n_batches = len(val_loader)
    
    return {
        'total': total_loss / n_batches,
        'reconstruction': loss_recon / n_batches,
        'kl': loss_kl / n_batches
    }


def save_checkpoint(
    model: DigitalTwinModel,
    optimizer: optim.Optimizer,
    epoch: int,
    losses: dict,
    checkpoint_dir: Path
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Also save latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    print(f"Saved checkpoint: {checkpoint_path}")


def plot_losses(train_losses: list, val_losses: list, save_path: Path):
    """Plot training and validation losses."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Total loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, [l['total'] for l in train_losses], label='Train', marker='o')
    plt.plot(epochs, [l['total'] for l in val_losses], label='Val', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True)
    
    # Reconstruction loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, [l['reconstruction'] for l in train_losses], label='Train', marker='o')
    plt.plot(epochs, [l['reconstruction'] for l in val_losses], label='Val', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # KL loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, [l['kl'] for l in train_losses], label='Train', marker='o')
    plt.plot(epochs, [l['kl'] for l in val_losses], label='Val', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence Loss')
    plt.title('KL Divergence Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """Main training function."""
    print("="*70)
    print("PHASE 2: DIGITAL TWIN CVAE TRAINING")
    print("="*70)
    
    # Configuration
    model_config = MODEL_CONFIG
    train_config = TRAINING_CONFIG
    data_config = DATA_CONFIG
    
    # Device
    device = torch.device(train_config['device'])
    print(f"\nUsing device: {device}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=train_config['batch_size'],
        train_split=data_config['train_split'],
        val_split=data_config['val_split'],
        test_split=data_config['test_split'],
        random_seed=data_config['random_seed'],
        shuffle=data_config['shuffle']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = DigitalTwinModel(config=model_config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # Training loop
    print(f"\nStarting training for {train_config['num_epochs']} epochs...")
    print("="*70)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(1, train_config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{train_config['num_epochs']}")
        print("-"*70)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, train_config)
        train_losses.append(train_loss)
        
        print(f"Train Loss: {train_loss['total']:.4f} "
              f"(Recon: {train_loss['reconstruction']:.4f}, "
              f"KL: {train_loss['kl']:.4f})")
        
        # Validate
        val_loss = validate(model, val_loader, device, train_config)
        val_losses.append(val_loss)
        
        print(f"Val Loss: {val_loss['total']:.4f} "
              f"(Recon: {val_loss['reconstruction']:.4f}, "
              f"KL: {val_loss['kl']:.4f})")
        
        # Save checkpoint
        if epoch % train_config['save_every'] == 0 or epoch == train_config['num_epochs']:
            save_checkpoint(model, optimizer, epoch, val_loss, CHECKPOINTS_DIR)
        
        # Save best model
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            best_path = CHECKPOINTS_DIR / "checkpoint_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': val_loss
            }, best_path)
            print(f"New best model saved (val loss: {best_val_loss:.4f})")
    
    # Plot losses
    print("\nPlotting loss curves...")
    plot_losses(train_losses, val_losses, LOGS_DIR / "training_losses.png")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {
            'model': model_config,
            'training': train_config,
            'data': data_config
        }
    }
    
    with open(LOGS_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {CHECKPOINTS_DIR}")
    print(f"Logs saved to: {LOGS_DIR}")


if __name__ == "__main__":
    main()

