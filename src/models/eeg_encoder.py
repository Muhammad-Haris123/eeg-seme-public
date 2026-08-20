"""
EEG Encoder Module.

Encodes EEG features (PSD + connectivity) into a compact latent representation.

Architecture: MLP-based encoder (extendable to CNN/GNN in future work)

Input: EEG features aggregated across epochs
  - PSD: (subjects, channels, frequencies)
  - Band powers: (subjects, channels, bands)
  - Connectivity: (subjects, bands, channels, channels)

Output: EEG latent vector (eeg_latent_dim)

Author: Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGEncoder(nn.Module):
    """
    Encodes EEG features into latent representation.
    
    Currently implements MLP-based encoding.
    Design is extendable to CNN/GNN architectures.
    """
    
    def __init__(
        self,
        psd_dim: int = 20,  # Number of frequency bins
        n_channels: int = 19,
        n_bands: int = 5,
        eeg_latent_dim: int = 128,
        hidden_dims: list = [256, 512]
    ):
        """
        Initialize EEG Encoder.
        
        Args:
            psd_dim: Number of PSD frequency bins
            n_channels: Number of EEG channels
            n_bands: Number of frequency bands
            eeg_latent_dim: Dimension of output latent vector
            hidden_dims: List of hidden layer dimensions
        """
        super(EEGEncoder, self).__init__()
        
        self.psd_dim = psd_dim
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.eeg_latent_dim = eeg_latent_dim
        
        # Calculate input dimension
        # PSD: (channels, frequencies) -> flattened
        psd_features = n_channels * psd_dim
        
        # Band powers: (channels, bands) -> flattened
        band_power_features = n_channels * n_bands
        
        # Connectivity: (bands, channels, channels) -> upper triangle per band
        # For each band: n_channels * (n_channels - 1) / 2 (upper triangle)
        # We concatenate coherence + PLV, so multiply by 2
        triu_size = n_channels * (n_channels - 1) // 2
        connectivity_features = n_bands * triu_size * 2  # coherence + PLV
        
        input_dim = psd_features + band_power_features + connectivity_features
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, eeg_latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def extract_connectivity_features(self, coherence: torch.Tensor, plv: torch.Tensor) -> torch.Tensor:
        """
        Extract upper triangle of connectivity matrices.
        
        Args:
            coherence: (batch, bands, channels, channels)
            plv: (batch, bands, channels, channels)
        
        Returns:
            Flattened connectivity features (batch, features)
        """
        batch_size = coherence.shape[0]
        n_bands = coherence.shape[1]
        n_channels = coherence.shape[2]
        
        # Extract upper triangle (excluding diagonal) for each band
        triu_indices = torch.triu_indices(n_channels, n_channels, offset=1)
        
        connectivity_features = []
        
        for band_idx in range(n_bands):
            # Coherence upper triangle
            coh_triu = coherence[:, band_idx, triu_indices[0], triu_indices[1]]
            # PLV upper triangle
            plv_triu = plv[:, band_idx, triu_indices[0], triu_indices[1]]
            # Concatenate
            connectivity_features.append(torch.cat([coh_triu, plv_triu], dim=1))
        
        # Concatenate across bands
        connectivity = torch.cat(connectivity_features, dim=1)
        
        return connectivity
    
    def forward(
        self,
        psd: torch.Tensor,
        band_powers: torch.Tensor,
        coherence: torch.Tensor,
        plv: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode EEG features to latent representation.
        
        Args:
            psd: Power Spectral Density (batch, channels, frequencies)
            band_powers: Band power features (batch, channels, bands)
            coherence: Coherence matrices (batch, bands, channels, channels)
            plv: Phase Locking Value matrices (batch, bands, channels, channels)
        
        Returns:
            EEG latent vector (batch, eeg_latent_dim)
        """
        batch_size = psd.shape[0]
        
        # Flatten PSD: (batch, channels, frequencies) -> (batch, channels * frequencies)
        psd_flat = psd.view(batch_size, -1)
        
        # Flatten band powers: (batch, channels, bands) -> (batch, channels * bands)
        band_powers_flat = band_powers.view(batch_size, -1)
        
        # Extract connectivity features
        connectivity_flat = self.extract_connectivity_features(coherence, plv)
        
        # Concatenate all features
        eeg_features = torch.cat([psd_flat, band_powers_flat, connectivity_flat], dim=1)
        
        # Encode to latent
        eeg_latent = self.encoder(eeg_features)
        
        return eeg_latent

