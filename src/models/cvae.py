"""
Conditional Variational Autoencoder (CVAE) Module.

Generative model that learns disease-aware, drug-conditioned latent representations
for simulating post-drug EEG states.

Architecture:
  - Encoder: Maps fused representation to latent distribution (μ, logσ²)
  - Decoder: Reconstructs EEG features from latent + conditions

Loss: L = L_reconstruction + β * L_KL

Author: Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder for EEG simulation.
    
    Conditions on:
      - Disease label (AD vs HC)
      - Drug information
    
    Learns patient-specific latent brain representations.
    """
    
    def __init__(
        self,
        fused_dim: int = 256,
        latent_dim: int = 128,
        output_dim: int = None,  # Will be set based on EEG feature dimensions
        condition_dim: int = 65,  # drug_latent_dim (64) + disease_dim (1)
        hidden_dims: list = [512, 256]
    ):
        """
        Initialize CVAE.
        
        Args:
            fused_dim: Dimension of fused input representation
            latent_dim: Dimension of latent space
            output_dim: Dimension of reconstructed EEG features
            condition_dim: Dimension of condition (drug + disease)
            hidden_dims: List of hidden layer dimensions for encoder/decoder
        """
        super(CVAE, self).__init__()
        
        self.fused_dim = fused_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        
        # Encoder: fused -> (μ, logσ²)
        encoder_layers = []
        prev_dim = fused_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.encoder_base = nn.Sequential(*encoder_layers)
        
        # Output heads for mean and log-variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder: (latent + condition) -> reconstructed features
        decoder_input_dim = latent_dim + condition_dim
        
        decoder_layers = []
        prev_dim = decoder_input_dim
        
        # Reverse hidden_dims for decoder
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        if output_dim is not None:
            decoder_layers.append(nn.Linear(prev_dim, output_dim))
            self.decoder = nn.Sequential(*decoder_layers)
            self.decoder_output = None  # Not used when decoder is sequential
        else:
            # Will be set later
            self.decoder_base = nn.Sequential(*decoder_layers[:-4])  # Remove last layer
            self.decoder_output = None
    
    def set_output_dim(self, output_dim: int):
        """Set output dimension and create output layer."""
        if self.decoder_output is None:
            # Get the dimension before output layer
            last_hidden_dim = 512  # Assuming first hidden_dim
            self.decoder_output = nn.Linear(last_hidden_dim, output_dim)
        else:
            # Update existing output layer
            self.decoder_output = nn.Linear(self.decoder_output.in_features, output_dim)
    
    def encode(self, fused_repr: torch.Tensor) -> tuple:
        """
        Encode fused representation to latent distribution.
        
        Args:
            fused_repr: Fused representation (batch, fused_dim)
        
        Returns:
            Tuple of (μ, logσ²) both of shape (batch, latent_dim)
        """
        h = self.encoder_base(fused_repr)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean (batch, latent_dim)
            logvar: Log-variance (batch, latent_dim)
        
        Returns:
            Sampled latent vector (batch, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector + condition to reconstructed features.
        
        Args:
            z: Latent vector (batch, latent_dim)
            condition: Condition vector (batch, condition_dim) - [drug_latent, disease_label]
        
        Returns:
            Reconstructed EEG features (batch, output_dim)
        """
        # Concatenate latent and condition
        decoder_input = torch.cat([z, condition], dim=1)
        
        # Use sequential decoder (decoder_output is None when output_dim is set)
        output = self.decoder(decoder_input)
        
        return output
    
    def forward(
        self,
        fused_repr: torch.Tensor,
        condition: torch.Tensor,
        return_latent: bool = False
    ) -> tuple:
        """
        Forward pass through CVAE.
        
        Args:
            fused_repr: Fused representation (batch, fused_dim)
            condition: Condition vector (batch, condition_dim)
            return_latent: Whether to return latent distribution
        
        Returns:
            If return_latent: (reconstructed, mu, logvar, z)
            Else: (reconstructed, mu, logvar)
        """
        # Encode
        mu, logvar = self.encode(fused_repr)
        
        # Sample from latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decode(z, condition)
        
        if return_latent:
            return reconstructed, mu, logvar, z
        else:
            return reconstructed, mu, logvar
    
    def sample(
        self,
        condition: torch.Tensor,
        num_samples: int = 1,
        z: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Sample from the model (for simulation).
        
        Args:
            condition: Condition vector (batch, condition_dim)
            num_samples: Number of samples per condition
            z: Optional latent vector (if None, samples from prior)
        
        Returns:
            Sampled EEG features (batch * num_samples, output_dim)
        """
        batch_size = condition.shape[0]
        
        if z is None:
            # Sample from prior N(0, I)
            z = torch.randn(batch_size * num_samples, self.latent_dim, device=condition.device)
            # Expand condition
            condition_expanded = condition.repeat(num_samples, 1)
        else:
            condition_expanded = condition
        
        # Decode
        with torch.no_grad():
            sampled = self.decode(z, condition_expanded)
        
        return sampled

