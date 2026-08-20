"""
Digital Twin wrapper for Explainable AI (Captum, SHAP, occlusion).

Provides deterministic CVAE forwards (z = mu, no sampling), scalar losses for IG/DeepLIFT,
latent probes, decoder freeze hooks, checkpoint resolution, and intermediate activations.

Complexity: O(batch) per forward; activation hooks add O(encoder depth) memory for stored tensors.

Author: FYP Digital Twin XAI
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.train_phase2 import DigitalTwinModel, prepare_target_features
from src.models.config import MODEL_CONFIG, CHECKPOINTS_DIR, CHECKPOINTS_5FOLD_DIR

logger = logging.getLogger(__name__)

# Flatten layout (must match prepare_target_features)
N_CHANNELS = MODEL_CONFIG["eeg_encoder"]["n_channels"]
N_BANDS = MODEL_CONFIG["eeg_encoder"]["n_bands"]
PSD_DIM = MODEL_CONFIG["eeg_encoder"]["psd_dim"]
PSD_SIZE = N_CHANNELS * PSD_DIM
BAND_SIZE = N_CHANNELS * N_BANDS
TRIU_SIZE = N_CHANNELS * (N_CHANNELS - 1) // 2
CONNECTIVITY_SIZE = N_BANDS * TRIU_SIZE * 2
EEG_FLAT_SIZE = PSD_SIZE + BAND_SIZE + CONNECTIVITY_SIZE
DRUG_DIM = MODEL_CONFIG["drug_encoder"]["input_dim"]

STANDARD_CHANNELS: Tuple[str, ...] = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
)


def unflatten_eeg(flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map (batch, 2185) PSD+bands+conn packed vector to model tensors."""
    batch = flat.shape[0]
    psd_flat = flat[:, :PSD_SIZE]
    band_flat = flat[:, PSD_SIZE : PSD_SIZE + BAND_SIZE]
    conn_flat = flat[:, PSD_SIZE + BAND_SIZE :]
    psd = psd_flat.view(batch, N_CHANNELS, PSD_DIM)
    band_powers = band_flat.view(batch, N_CHANNELS, N_BANDS)
    triu_idx = torch.triu_indices(N_CHANNELS, N_CHANNELS, offset=1)
    coherence = torch.zeros(batch, N_BANDS, N_CHANNELS, N_CHANNELS, device=flat.device, dtype=flat.dtype)
    plv = torch.zeros(batch, N_BANDS, N_CHANNELS, N_CHANNELS, device=flat.device, dtype=flat.dtype)
    for b in range(N_BANDS):
        start = b * TRIU_SIZE * 2
        coh_triu = conn_flat[:, start : start + TRIU_SIZE]
        plv_triu = conn_flat[:, start + TRIU_SIZE : start + 2 * TRIU_SIZE]
        coherence[:, b, triu_idx[0], triu_idx[1]] = coh_triu
        coherence[:, b, triu_idx[1], triu_idx[0]] = coh_triu
        coherence[:, b].diagonal(0, 1, 2).fill_(1.0)
        plv[:, b, triu_idx[0], triu_idx[1]] = plv_triu
        plv[:, b, triu_idx[1], triu_idx[0]] = plv_triu
        plv[:, b].diagonal(0, 1, 2).fill_(1.0)
    return psd, band_powers, coherence, plv


def resolve_checkpoint_path(project_root: Optional[Path] = None) -> Optional[Path]:
    """
    Try checkpoints in order: best → final_full → fold_0..fold_4 best → latest.

    Returns:
        Path to checkpoint or None if none exist.

    Complexity: O(k) filesystem checks for k candidate files.
    """
    root = project_root or Path(__file__).resolve().parent.parent.parent
    ckpt_dir = root / "models" / "checkpoints"
    fold_dir = root / "models" / "checkpoints_5fold"
    candidates: List[Path] = [
        ckpt_dir / "checkpoint_best.pt",
        ckpt_dir / "checkpoint_final_full.pt",
    ]
    for i in range(5):
        candidates.append(fold_dir / f"fold_{i}_best.pt")
    candidates.append(ckpt_dir / "checkpoint_latest.pt")
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_digital_twin_for_xai(
    device: torch.device,
    project_root: Optional[Path] = None,
) -> Tuple[DigitalTwinModel, Path]:
    """
    Load DigitalTwinModel with robust checkpoint fallback.

    Raises:
        FileNotFoundError: if no checkpoint found (caller may catch and random-init wrapper).

    Complexity: O(model_size) I/O + load_state_dict.
    """
    root = project_root or Path(__file__).resolve().parent.parent.parent
    ckpt = resolve_checkpoint_path(root)
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found in models/checkpoints or checkpoints_5fold")

    model = DigitalTwinModel(config=MODEL_CONFIG).to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded DigitalTwinModel from %s", ckpt)
    return model, ckpt


class DigitalTwinWrapper(nn.Module):
    """
    XAI-facing wrapper around DigitalTwinModel.

    Deterministic rule: CVAE latent always uses **mu** from encode(); never samples z during XAI.

    Attributes:
        feature_dim: Flattened EEG feature size (2185).
        drug_dim: ChemBERTa dimension (384).
    """

    feature_dim: int = EEG_FLAT_SIZE
    drug_dim: int = DRUG_DIM

    def __init__(
        self,
        model: DigitalTwinModel,
        *,
        warn_random_init: bool = False,
    ):
        super().__init__()
        self.model = model
        self._decoder_frozen = False
        self._warn_random_init = warn_random_init
        if warn_random_init:
            warnings.warn(
                "DigitalTwinWrapper initialized without a trained checkpoint — "
                "attributions are not scientifically meaningful.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def latent_dim(self) -> int:
        return self.model.cvae.latent_dim

    # --- Core deterministic forward ---

    def encode_decode_deterministic(
        self,
        psd: torch.Tensor,
        band_powers: torch.Tensor,
        coherence: torch.Tensor,
        plv: torch.Tensor,
        drug_embedding: torch.Tensor,
        disease_label: torch.Tensor,
        *,
        enable_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode fused input to (mu, logvar), decode using **z = mu** only.

        Args:
            enable_grad: If False, runs under torch.no_grad() (activations still computed).

        Returns:
            reconstructed, mu, logvar

        Complexity: O(batch * param forward).
        """
        with torch.set_grad_enabled(enable_grad):
            eeg_latent = self.model.eeg_encoder(psd, band_powers, coherence, plv)
            drug_latent = self.model.drug_encoder(drug_embedding)
            dl = disease_label.unsqueeze(1) if disease_label.dim() == 1 else disease_label
            fused = self.model.fusion(eeg_latent, drug_latent, dl)
            mu, logvar = self.model.cvae.encode(fused)
            z = mu
            condition = torch.cat([drug_latent, dl], dim=1)
            reconstructed = self.model.cvae.decode(z, condition)
        return reconstructed, mu, logvar

    def forward(
        self,
        eeg_flat: torch.Tensor,
        drug_embedding: torch.Tensor,
        disease_label: Optional[torch.Tensor] = None,
        *,
        enable_grad: bool = True,
    ) -> torch.Tensor:
        """
        Legacy-compatible forward: flattened EEG + drug -> reconstruction (deterministic).

        Matches prior explainability code expecting full recon vector.

        Args:
            eeg_flat: (batch, 2185)
            drug_embedding: (batch, 384)
            disease_label: (batch,) or (batch,1); defaults to zeros (HC) if None.

        Returns:
            (batch, 2185) reconstruction.

        Example:
            >>> w = DigitalTwinWrapper(model)
            >>> r = w(torch.randn(2, 2185), torch.randn(2, 384), torch.zeros(2))
        """
        if disease_label is None:
            disease_label = torch.zeros(eeg_flat.shape[0], device=eeg_flat.device, dtype=eeg_flat.dtype)
        psd, bp, coh, plv = unflatten_eeg(eeg_flat)
        recon, _, _ = self.encode_decode_deterministic(
            psd, bp, coh, plv, drug_embedding, disease_label, enable_grad=enable_grad
        )
        return recon

    def forward_reconstruction(
        self,
        eeg_tensor: torch.Tensor,
        drug_tensor: torch.Tensor,
        disease_label: Optional[torch.Tensor] = None,
        *,
        enable_grad: bool = True,
    ) -> torch.Tensor:
        """
        Scalar **mean** MSE between deterministic reconstruction and input EEG features.

        IG/DeepLIFT target: single differentiable scalar per batch (mean over batch and features).

        Args:
            eeg_tensor: (batch, 2185) input features (also serves as target).
            drug_tensor: (batch, 384)

        Returns:
            1-D tensor of shape ``(1,)`` containing mean squared error (not 0-dim) so Captum
            gradient helpers can index ``outputs[0]`` without raising on scalar tensors.

        Raises:
            ValueError: if batch sizes mismatch.

        Complexity: O(batch * 2185).
        """
        if eeg_tensor.shape[0] != drug_tensor.shape[0]:
            raise ValueError("eeg_tensor and drug_tensor batch sizes must match")
        if disease_label is None:
            disease_label = torch.zeros(eeg_tensor.shape[0], device=eeg_tensor.device, dtype=eeg_tensor.dtype)
        recon = self.forward(eeg_tensor, drug_tensor, disease_label, enable_grad=enable_grad)
        target = eeg_tensor.detach()
        loss = F.mse_loss(recon, target, reduction="mean")
        # Captum gradient.py indexes outputs[0] then torch.unbind(outputs); 0-dim tensors fail.
        return loss.unsqueeze(0)

    def forward_latent(
        self,
        eeg_tensor: torch.Tensor,
        drug_tensor: torch.Tensor,
        disease_label: Optional[torch.Tensor] = None,
        *,
        enable_grad: bool = True,
    ) -> torch.Tensor:
        """
        Return CVAE latent mean mu (deterministic latent state), shape (batch, latent_dim).

        Complexity: O(batch * encoder + fusion + CVAE encode).
        """
        if disease_label is None:
            disease_label = torch.zeros(eeg_tensor.shape[0], device=eeg_tensor.device, dtype=eeg_tensor.dtype)
        psd, bp, coh, plv = unflatten_eeg(eeg_tensor)
        dl = disease_label.unsqueeze(1) if disease_label.dim() == 1 else disease_label
        with torch.set_grad_enabled(enable_grad):
            eeg_latent = self.model.eeg_encoder(psd, bp, coh, plv)
            drug_latent = self.model.drug_encoder(drug_tensor)
            fused = self.model.fusion(eeg_latent, drug_latent, dl)
            mu, _ = self.model.cvae.encode(fused)
        return mu

    def forward_drug_sensitivity(
        self,
        eeg_tensor: torch.Tensor,
        drug_tensor: torch.Tensor,
        disease_label: Optional[torch.Tensor] = None,
        *,
        enable_grad: bool = True,
    ) -> torch.Tensor:
        """
        Full per-feature reconstruction vector (same as ``forward``).

        Used when SHAP/occlusion need vector outputs per perturbed sample.

        Complexity: Same as forward.
        """
        return self.forward(eeg_tensor, drug_tensor, disease_label, enable_grad=enable_grad)

    def freeze_decoder(self) -> None:
        """Freeze CVAE decoder parameters (encoder-only attribution experiments)."""
        for p in self.model.cvae.decoder.parameters():
            p.requires_grad = False
        self._decoder_frozen = True

    def unfreeze_decoder(self) -> None:
        """Restore decoder gradients."""
        for p in self.model.cvae.decoder.parameters():
            p.requires_grad = True
        self._decoder_frozen = False

    def get_intermediate_activations(
        self,
        eeg_flat: torch.Tensor,
        drug_embedding: torch.Tensor,
        disease_label: Optional[torch.Tensor] = None,
        *,
        enable_grad: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward and collect intermediate tensors from EEG encoder MLP stack.

        Keys include ``eeg_encoder_input`` (concatenated flattened features), then
        ``eeg_encoder_layer_<i>`` for each submodule in ``EEGEncoder.encoder``.

        Args:
            enable_grad: If True, tensors retain grad (for layer-wise attribution experiments).

        Returns:
            Mapping layer_name -> activation tensor.

        Complexity: O(depth) forward; memory O(depth * batch * width).
        """
        if disease_label is None:
            disease_label = torch.zeros(eeg_flat.shape[0], device=eeg_flat.device, dtype=eeg_flat.dtype)
        psd, bp, coh, plv = unflatten_eeg(eeg_flat)
        out: Dict[str, torch.Tensor] = {}
        with torch.set_grad_enabled(enable_grad):
            enc = self.model.eeg_encoder
            batch = eeg_flat.shape[0]
            psd_flat = psd.view(batch, -1)
            bp_flat = bp.view(batch, -1)
            cf = enc.extract_connectivity_features(coh, plv)
            feats = torch.cat([psd_flat, bp_flat, cf], dim=1)
            out["eeg_encoder_input"] = feats.clone()
            x = feats
            for i, layer in enumerate(enc.encoder):
                x = layer(x)
                out[f"eeg_encoder_layer_{i}_{layer.__class__.__name__}"] = x.clone()
            eeg_latent = x
            dl = disease_label.unsqueeze(1) if disease_label.dim() == 1 else disease_label
            drug_latent = self.model.drug_encoder(drug_embedding)
            out["drug_latent"] = drug_latent.clone()
            fused = self.model.fusion(eeg_latent, drug_latent, dl)
            out["fusion_output"] = fused.clone()
            mu, logvar = self.model.cvae.encode(fused)
            out["cvae_mu"] = mu.clone()
            out["cvae_logvar"] = logvar.clone()
        return out


def build_wrapper(
    device: torch.device,
    *,
    allow_untrained: bool = False,
) -> Tuple[DigitalTwinWrapper, Optional[Path]]:
    """
    Construct ``DigitalTwinWrapper`` with checkpoint when available.

    Args:
        allow_untrained: If True and no checkpoint, return randomly initialized model with warning.

    Returns:
        (wrapper, checkpoint_path or None)

    Raises:
        FileNotFoundError: if no checkpoint and ``allow_untrained`` is False.
    """
    try:
        model, ckpt = load_digital_twin_for_xai(device)
        return DigitalTwinWrapper(model), ckpt
    except FileNotFoundError:
        if not allow_untrained:
            raise
        warnings.warn("No checkpoint — using randomly initialized weights for smoke tests only.")
        model = DigitalTwinModel(config=MODEL_CONFIG).to(device)
        return DigitalTwinWrapper(model, warn_random_init=True), None
