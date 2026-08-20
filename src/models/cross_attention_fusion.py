"""
Cross-modal attention fusion for EEG + drug latent representations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    """
    Query = drug latent, Key/Value = EEG latent tokens.
    Returns fused representation with same output shape as FusionModule.
    """

    def __init__(
        self,
        eeg_latent_dim: int = 128,
        drug_latent_dim: int = 64,
        attn_dim: int = 64,
        fusion_dim: int = 256,
        n_tokens: int = 8,
        n_disease_classes: int = 2,
    ):
        super().__init__()
        if eeg_latent_dim % n_tokens != 0:
            raise ValueError(
                f"eeg_latent_dim ({eeg_latent_dim}) must be divisible by n_tokens ({n_tokens})"
            )
        self.n_tokens = n_tokens
        self.token_dim = eeg_latent_dim // n_tokens
        self.attn_dim = attn_dim

        self.W_q = nn.Linear(drug_latent_dim, attn_dim)
        self.W_k = nn.Linear(self.token_dim, attn_dim)
        self.W_v = nn.Linear(self.token_dim, attn_dim)

        self.disease_emb = nn.Embedding(n_disease_classes, 16)
        concat_dim = attn_dim + drug_latent_dim + 16
        self.output_mlp = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, fusion_dim),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        eeg_latent: torch.Tensor,
        drug_latent: torch.Tensor,
        disease_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        eeg_latent: (batch, eeg_latent_dim)
        drug_latent: (batch, drug_latent_dim)
        disease_label: (batch,) or (batch,1)
        """
        bsz = eeg_latent.size(0)
        eeg_tokens = eeg_latent.view(bsz, self.n_tokens, self.token_dim)

        Q = self.W_q(drug_latent).unsqueeze(1)  # (batch,1,attn_dim)
        K = self.W_k(eeg_tokens)  # (batch,n_tokens,attn_dim)
        V = self.W_v(eeg_tokens)  # (batch,n_tokens,attn_dim)

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(float(self.attn_dim))  # (batch,1,n_tokens)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, V).squeeze(1)  # (batch,attn_dim)
        self.last_attention_weights = weights.detach()

        if disease_label.dim() > 1:
            disease_label = disease_label.squeeze(1)
        disease_label = disease_label.long().clamp(min=0, max=1)
        dis_emb = self.disease_emb(disease_label)  # (batch,16)

        fused = torch.cat([context, drug_latent, dis_emb], dim=-1)
        return self.output_mlp(fused)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Returns shape (batch, 1, n_tokens) from last forward pass."""
        return self.last_attention_weights

