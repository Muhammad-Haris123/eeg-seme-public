"""
Fusion Module.

Fuses EEG latent, drug latent, and disease label into a unified representation.

Architecture: Simplified concatenation (extendable to Cross-Attention)

Input:
  - EEG latent: (batch, eeg_latent_dim)
  - Drug latent: (batch, drug_latent_dim)
  - Disease label: (batch, 1) - AD=1, HC=0

Output: Fused representation (batch, fused_dim)

Author: Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    """
    Fuses multi-modal representations.
    
    Currently implements concatenation-based fusion.
    Design is extendable to Cross-Attention mechanism.
    """
    
    def __init__(
        self,
        eeg_latent_dim: int = 128,
        drug_latent_dim: int = 64,
        disease_dim: int = 1,
        fused_dim: int = 256,
        use_attention: bool = False
    ):
        """
        Initialize Fusion Module.
        
        Args:
            eeg_latent_dim: Dimension of EEG latent vector
            drug_latent_dim: Dimension of drug latent vector
            disease_dim: Dimension of disease label (1 for binary)
            fused_dim: Dimension of fused representation
            use_attention: Whether to use attention (future extension)
        """
        super(FusionModule, self).__init__()
        
        self.eeg_latent_dim = eeg_latent_dim
        self.drug_latent_dim = drug_latent_dim
        self.disease_dim = disease_dim
        self.fused_dim = fused_dim
        self.use_attention = use_attention
        
        # Concatenation-based fusion
        concat_dim = eeg_latent_dim + drug_latent_dim + disease_dim
        
        if use_attention:
            # Future: Cross-Attention implementation
            raise NotImplementedError("Cross-Attention fusion not yet implemented")
        else:
            # Simple MLP fusion
            self.fusion = nn.Sequential(
                nn.Linear(concat_dim, fused_dim),
                nn.BatchNorm1d(fused_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(fused_dim, fused_dim)
            )
    
    def forward(
        self,
        eeg_latent: torch.Tensor,
        drug_latent: torch.Tensor,
        disease_label: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multi-modal representations.
        
        Args:
            eeg_latent: EEG latent vector (batch, eeg_latent_dim)
            drug_latent: Drug latent vector (batch, drug_latent_dim)
            disease_label: Disease label (batch, 1) - AD=1, HC=0
        
        Returns:
            Fused representation (batch, fused_dim)
        """
        # Ensure disease_label is the right shape
        if disease_label.dim() == 1:
            disease_label = disease_label.unsqueeze(1)
        
        # Concatenate all representations
        fused_input = torch.cat([eeg_latent, drug_latent, disease_label], dim=1)
        
        # Apply fusion
        fused_output = self.fusion(fused_input)
        
        return fused_output

