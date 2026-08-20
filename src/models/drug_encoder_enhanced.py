"""
Enhanced drug encoder with pharmacology-guided gating.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnhancedDrugEncoder(nn.Module):
    """
    Encodes 416-D enhanced drug embeddings (384 ChemBERTa + 32 pharma prior).

    Output shape is (batch, 64) by default, matching the existing DrugEncoder.
    """

    def __init__(
        self,
        chemberta_dim: int = 384,
        pharma_dim: int = 32,
        drug_latent_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.chemberta_dim = chemberta_dim
        self.pharma_dim = pharma_dim
        self.input_dim = chemberta_dim + pharma_dim
        self.drug_latent_dim = drug_latent_dim

        self.chem_stream = nn.Sequential(
            nn.Linear(chemberta_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
        )
        self.pharma_stream = nn.Sequential(
            nn.Linear(pharma_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 32),
        )
        self.gate = nn.Sequential(
            nn.Linear(32, 64),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, drug_latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 416) concatenated ChemBERTa(384) + pharma_prior(32)
        Returns:
            (batch, drug_latent_dim), default (batch, 64)
        """
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch, {self.input_dim}), got {tuple(x.shape)}"
            )

        chem_input = x[:, : self.chemberta_dim]
        pharma_input = x[:, self.chemberta_dim :]

        chem_out = self.chem_stream(chem_input)  # (batch, 64)
        pharma_out = self.pharma_stream(pharma_input)  # (batch, 32)

        gate_weights = self.gate(pharma_out)  # (batch, 64)
        gated_chem = chem_out * gate_weights  # (batch, 64)

        fused = torch.cat([gated_chem, pharma_out], dim=-1)  # (batch, 96)
        return self.fusion(fused)  # (batch, drug_latent_dim)
