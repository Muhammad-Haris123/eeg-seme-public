"""
Drug Encoder Module.

Encodes drug molecular embeddings (ChemBERTa) into drug latent representation.

Architecture: Feed-forward Neural Network

Input: ChemBERTa embeddings (384,)
Output: Drug latent vector (drug_latent_dim)

Author: Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugEncoder(nn.Module):
    """
    Encodes drug embeddings into latent representation.
    
    Simple feed-forward network that processes ChemBERTa embeddings.
    """
    
    def __init__(
        self,
        input_dim: int = 384,  # ChemBERTa embedding dimension
        drug_latent_dim: int = 64,
        hidden_dims: list = [256, 128]
    ):
        """
        Initialize Drug Encoder.
        
        Args:
            input_dim: Dimension of input drug embeddings (ChemBERTa)
            drug_latent_dim: Dimension of output latent vector
            hidden_dims: List of hidden layer dimensions
        """
        super(DrugEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.drug_latent_dim = drug_latent_dim
        
        # Build feed-forward layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, drug_latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, drug_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode drug embeddings to latent representation.
        
        Args:
            drug_embeddings: ChemBERTa embeddings (batch, 384)
        
        Returns:
            Drug latent vector (batch, drug_latent_dim)
        """
        drug_latent = self.encoder(drug_embeddings)
        return drug_latent

