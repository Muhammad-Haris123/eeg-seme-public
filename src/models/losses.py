"""
Loss Functions for Phase 2 CVAE Training.

Implements:
- Reconstruction loss (MSE)
- KL divergence loss
- Optional condition consistency loss

Total: L = L_reconstruction + β * L_KL (+ optional condition loss)

Author: Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def reconstruction_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Reconstruction loss (MSE).
    
    Args:
        reconstructed: Reconstructed EEG features (batch, output_dim)
        target: Target EEG features (batch, output_dim)
        reduction: 'mean' or 'sum'
    
    Returns:
        Reconstruction loss
    """
    return F.mse_loss(reconstructed, target, reduction=reduction)


def kl_divergence_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    KL divergence loss (regularization term).
    
    Encourages latent distribution to be close to standard normal N(0, I).
    
    Args:
        mu: Mean of latent distribution (batch, latent_dim)
        logvar: Log-variance of latent distribution (batch, latent_dim)
        reduction: 'mean' or 'sum'
    
    Returns:
        KL divergence loss
    """
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:
        return kl


def condition_consistency_loss(
    latent: torch.Tensor,
    disease_labels: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Optional condition consistency loss.
    
    Encourages latent representations to be separable by disease condition.
    This is a lightweight auxiliary loss.
    
    Args:
        latent: Latent vectors (batch, latent_dim)
        disease_labels: Disease labels (batch, 1) - AD=1, HC=0
        reduction: 'mean' or 'sum'
    
    Returns:
        Condition consistency loss
    """
    # Simple approach: encourage AD and HC latents to have different means
    ad_mask = (disease_labels == 1).squeeze()
    hc_mask = (disease_labels == 0).squeeze()
    
    if ad_mask.sum() == 0 or hc_mask.sum() == 0:
        # No loss if only one class in batch
        return torch.tensor(0.0, device=latent.device)
    
    ad_mean = latent[ad_mask].mean(dim=0)
    hc_mean = latent[hc_mask].mean(dim=0)
    
    # Encourage separation (negative distance = loss)
    # We want them to be far apart, so we minimize negative distance
    separation = torch.norm(ad_mean - hc_mean)
    loss = -separation  # Negative because we want to maximize separation
    
    return loss


def compute_total_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float = 0.01,
    beta_condition: float = 0.1,
    use_condition_loss: bool = False,
    latent: torch.Tensor = None,
    disease_labels: torch.Tensor = None
) -> Dict[str, torch.Tensor]:
    """
    Compute total loss for CVAE training.
    
    Args:
        reconstructed: Reconstructed features (batch, output_dim)
        target: Target features (batch, output_dim)
        mu: Latent mean (batch, latent_dim)
        logvar: Latent log-variance (batch, latent_dim)
        beta_kl: Weight for KL divergence
        beta_condition: Weight for condition loss
        use_condition_loss: Whether to use condition consistency loss
        latent: Latent vectors (for condition loss)
        disease_labels: Disease labels (for condition loss)
    
    Returns:
        Dictionary with individual losses and total loss
    """
    # Reconstruction loss
    loss_recon = reconstruction_loss(reconstructed, target)
    
    # KL divergence loss
    loss_kl = kl_divergence_loss(mu, logvar)
    
    # Total loss
    total_loss = loss_recon + beta_kl * loss_kl
    
    losses = {
        'reconstruction': loss_recon,
        'kl': loss_kl,
        'total': total_loss
    }
    
    # Optional condition loss
    if use_condition_loss and latent is not None and disease_labels is not None:
        loss_condition = condition_consistency_loss(latent, disease_labels)
        total_loss = total_loss + beta_condition * loss_condition
        losses['condition'] = loss_condition
        losses['total'] = total_loss
    
    return losses


# ============================================================================
# Biophysical pharmacodynamic constraints (Phase: constrained training)
# Added below existing functions to preserve backward compatibility.
# ============================================================================

# Canonical band power indices within the 2185-D feature vector
# Layout: 19 channels × 5 bands at positions [380:475]
BAND_INDICES = {
    'delta': (380, 399),   # 19 channels of delta power
    'theta': (399, 418),   # 19 channels of theta power
    'alpha': (418, 437),   # 19 channels of alpha power
    'beta':  (437, 456),   # 19 channels of beta power
    'gamma': (456, 475),   # 19 channels of gamma power
}
CONNECTIVITY_INDICES = (475, 1330)  # coherence block


def _zero_like_scalar(reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(0.0, device=reference.device, dtype=reference.dtype)


def compute_band_direction_constraint(
    reconstruction: torch.Tensor,
    baseline_features: torch.Tensor,
    drug_name: str,
    weight: float = 0.1
) -> torch.Tensor:
    """
    Soft directionality constraint for delta/theta/alpha/beta band changes.
    """
    if drug_name == 'baseline':
        return _zero_like_scalar(reconstruction)
    if baseline_features is None:
        return _zero_like_scalar(reconstruction)

    from src.drugs.pharmacological_embedding import get_eeg_effect_prior

    expected = get_eeg_effect_prior(drug_name)
    total_penalty = _zero_like_scalar(reconstruction)

    for band in ('delta', 'theta', 'alpha', 'beta'):
        start, end = BAND_INDICES[band]
        rec_band = reconstruction[:, start:end]
        base_band = baseline_features[:, start:end]

        actual_change = rec_band.mean() - base_band.mean()
        expected_dir = float(expected.get(band, 0.0))

        if expected_dir > 0:
            # Should increase; penalize decrease
            penalty = F.relu(-actual_change)
        elif expected_dir < 0:
            # Should decrease; penalize increase
            penalty = F.relu(actual_change)
        else:
            penalty = _zero_like_scalar(reconstruction)

        penalty = penalty * abs(expected_dir)
        total_penalty = total_penalty + penalty

    return total_penalty * float(weight)


def compute_connectivity_constraint(
    reconstruction: torch.Tensor,
    baseline_features: torch.Tensor,
    drug_name: str,
    weight: float = 0.05
) -> torch.Tensor:
    """
    Soft connectivity directionality constraint on coherence block [475:1330].
    """
    if drug_name == 'baseline':
        return _zero_like_scalar(reconstruction)
    if baseline_features is None:
        return _zero_like_scalar(reconstruction)

    c_start, c_end = CONNECTIVITY_INDICES
    rec_conn = reconstruction[:, c_start:c_end]
    base_conn = baseline_features[:, c_start:c_end]
    actual_change = rec_conn.mean() - base_conn.mean()

    # AD drugs should improve connectivity (increase coherence)
    penalty = F.relu(-actual_change)
    return penalty * float(weight)


def compute_constrained_total_loss(
    reconstruction: torch.Tensor,
    target_features: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    drug_name: str,
    baseline_features: torch.Tensor = None,
    beta: float = 0.01,
    constraint_weight: float = 0.1,
    epoch: int = 0,
    warmup_epochs: int = 5
) -> Dict[str, torch.Tensor]:
    """
    Full constrained loss:
    L = L_MSE + beta * L_KL + alpha_c * L_band + alpha_c * L_conn
    """
    # Keep MSE and KL logic aligned with existing losses.
    loss_mse = reconstruction_loss(reconstruction, target_features, reduction='mean')
    loss_kl = kl_divergence_loss(mu, log_var, reduction='mean')

    if epoch < warmup_epochs:
        alpha_c = 0.0
    else:
        alpha_c = float(constraint_weight) * min(1.0, float(epoch - warmup_epochs) / 5.0)

    if baseline_features is None or drug_name == 'baseline':
        loss_band = _zero_like_scalar(reconstruction)
        loss_conn = _zero_like_scalar(reconstruction)
    else:
        # Pass unit weights here; global scaling handled by alpha_c.
        loss_band = compute_band_direction_constraint(
            reconstruction, baseline_features, drug_name, weight=1.0
        )
        loss_conn = compute_connectivity_constraint(
            reconstruction, baseline_features, drug_name, weight=1.0
        )

    total = loss_mse + float(beta) * loss_kl + alpha_c * loss_band + alpha_c * loss_conn
    return {
        'total': total,
        'mse': loss_mse,
        'kl': loss_kl,
        'band_constraint': loss_band,
        'connectivity_constraint': loss_conn,
        'constraint_alpha': alpha_c,
    }

