"""
Latent Neural ODE for continuous drug-response trajectories.

Uses a pretrained constrained CVAE encoder/decoder (frozen) and trains
only the ODE dynamics function that evolves latent EEG state over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


DEFAULT_DAYS = [0, 1, 3, 7, 14, 21, 30]
BAND_SLICES = {
    "delta": slice(380, 399),
    "theta": slice(399, 418),
    "alpha": slice(418, 437),
    "beta": slice(437, 456),
    "connectivity": slice(475, 1330),
}


class DrugODEFunction(nn.Module):
    """
    Neural ODE dynamics: dz/dt = f(z, drug_latent, t).

    Signature forward(t, z) is required by torchdiffeq.
    Drug conditioning is injected via set_drug_latent() before odeint().
    """

    def __init__(self, latent_dim: int = 128, drug_latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.drug_latent_dim = drug_latent_dim
        input_dim = latent_dim + drug_latent_dim + 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.xavier_normal_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)
        self._drug_latent: Optional[torch.Tensor] = None

    def set_drug_latent(self, drug_latent: torch.Tensor) -> None:
        self._drug_latent = drug_latent

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self._drug_latent is None:
            raise RuntimeError("Call set_drug_latent() before integrating the ODE.")
        batch_size = z.shape[0]
        # torchdiffeq may pass a 0-d tensor; expand to (batch, 1)
        t_vec = torch.ones(batch_size, 1, device=z.device, dtype=z.dtype) * t
        drug = self._drug_latent
        if drug.shape[0] != batch_size:
            if drug.shape[0] == 1:
                drug = drug.expand(batch_size, -1)
            else:
                raise ValueError(f"drug_latent batch {drug.shape[0]} != z batch {batch_size}")
        inp = torch.cat([z, drug, t_vec], dim=-1)
        return self.net(inp)


def euler_odeint(func, z0: torch.Tensor, t_span: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
    """Forward-Euler fallback when torchdiffeq is unavailable."""
    trajectory = [z0]
    z = z0
    t_points = [float(x) for x in t_span.detach().cpu().tolist()]
    current_t = t_points[0]
    next_save_idx = 1

    while next_save_idx < len(t_points):
        target_t = t_points[next_save_idx]
        while current_t < target_t - 1e-8:
            step = min(dt, target_t - current_t)
            t_tensor = torch.tensor(current_t, dtype=z.dtype, device=z.device)
            dz = func(t_tensor, z)
            z = z + step * dz
            current_t += step
        trajectory.append(z.clone())
        next_save_idx += 1

    return torch.stack(trajectory, dim=0)


class LatentODEModel(nn.Module):
    """
    Continuous drug-response twin: frozen CVAE encode/decode + trainable ODE.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        drug_latent_dim: int = 64,
        hidden_dim: int = 256,
        ode_method: str = "dopri5",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.drug_latent_dim = drug_latent_dim
        self.ode_method = ode_method
        self.ode_func = DrugODEFunction(latent_dim, drug_latent_dim, hidden_dim)

        self.full_model = None
        self.eeg_encoder = None
        self.drug_encoder = None
        self.fusion = None
        self.cvae = None
        self.condition_dim = drug_latent_dim + 1

    def load_pretrained_components(self, checkpoint_path, device: str | torch.device = "cpu") -> None:
        from src.models.config import MODEL_CONFIG
        from src.models.train_phase2 import DigitalTwinModel

        device = torch.device(device)
        checkpoint_path = Path(checkpoint_path)
        print(f"Loading pretrained CVAE components from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = DigitalTwinModel(config=MODEL_CONFIG).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        self.full_model = model
        self.eeg_encoder = model.eeg_encoder
        self.drug_encoder = model.drug_encoder
        self.fusion = model.fusion
        self.cvae = model.cvae
        self.condition_dim = model.cvae.condition_dim

        for module in (self.eeg_encoder, self.drug_encoder, self.fusion, self.cvae):
            for p in module.parameters():
                p.requires_grad = False
            module.eval()

        print(
            f"Frozen components loaded (epoch={ckpt.get('epoch', '?')}). "
            f"Trainable ODE params: {sum(p.numel() for p in self.ode_func.parameters()):,}"
        )

    def _build_condition(self, drug_latent: torch.Tensor, disease_label: torch.Tensor) -> torch.Tensor:
        if disease_label.dim() == 1:
            disease = disease_label.unsqueeze(1)
        else:
            disease = disease_label
        return torch.cat([drug_latent, disease], dim=1)

    def encode_z0_from_flat(
        self,
        eeg_features: torch.Tensor,
        disease_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode flat 2185-D baseline features to CVAE latent mu under no-drug condition.
        Uses frozen EEG encoder MLP + fusion + CVAE encode.
        """
        assert self.eeg_encoder is not None and self.cvae is not None
        eeg_latent = self.eeg_encoder.encoder(eeg_features)
        zeros = torch.zeros(eeg_features.shape[0], 384, device=eeg_features.device, dtype=eeg_features.dtype)
        drug_latent = self.drug_encoder(zeros)
        fused = self.fusion(eeg_latent, drug_latent, disease_label)
        mu, _ = self.cvae.encode(fused)
        return mu

    def decode_latent(
        self,
        z: torch.Tensor,
        drug_latent: torch.Tensor,
        disease_label: torch.Tensor,
    ) -> torch.Tensor:
        condition = self._build_condition(drug_latent, disease_label)
        return self.cvae.decode(z, condition)

    def _integrate(
        self,
        z0: torch.Tensor,
        drug_latent: torch.Tensor,
        t_normalized: torch.Tensor,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        method = method or self.ode_method
        self.ode_func.set_drug_latent(drug_latent)
        use_torchdiffeq = HAS_TORCHDIFFEQ and method != "euler"
        if use_torchdiffeq:
            try:
                return odeint(self.ode_func, z0, t_normalized, method=method, rtol=1e-3, atol=1e-4)
            except Exception as exc:
                print(f"[warn] torchdiffeq failed ({exc}); falling back to Euler.")
        return euler_odeint(self.ode_func, z0, t_normalized, dt=0.01)

    @torch.no_grad()
    def simulate_trajectory(
        self,
        eeg_features: Optional[torch.Tensor] = None,
        drug_embedding: Optional[torch.Tensor] = None,
        disease_label: Optional[torch.Tensor] = None,
        time_points_days: Optional[Sequence[int]] = None,
        method: Optional[str] = None,
        z0: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Simulate drug response trajectory.

        Provide either (eeg_features, drug_embedding, disease_label) or
        (z0, drug_embedding, disease_label).
        """
        if time_points_days is None:
            time_points_days = list(DEFAULT_DAYS)
        if drug_embedding is None or disease_label is None:
            raise ValueError("drug_embedding and disease_label are required.")
        if z0 is None:
            if eeg_features is None:
                raise ValueError("Provide eeg_features or z0.")
            z0 = self.encode_z0_from_flat(eeg_features, disease_label)

        drug_latent = self.drug_encoder(drug_embedding)
        t_days = torch.tensor(list(time_points_days), dtype=z0.dtype, device=z0.device)
        t_norm = t_days / 30.0
        z_traj = self._integrate(z0, drug_latent, t_norm, method=method)  # (T, B, 128)

        T, B, _ = z_traj.shape
        features = []
        bands = {k: [] for k in BAND_SLICES}
        for t_idx in range(T):
            feat_t = self.decode_latent(z_traj[t_idx], drug_latent, disease_label)
            features.append(feat_t)
            for name, sl in BAND_SLICES.items():
                bands[name].append(feat_t[:, sl].mean(dim=1))

        feature_traj = torch.stack(features, dim=0)  # (T, B, 2185)
        band_np = {k: torch.stack(v, dim=0).detach().cpu().numpy() for k, v in bands.items()}

        # Latent velocity ||z(t+1)-z(t)||, pad last with repeat
        diffs = torch.norm(z_traj[1:] - z_traj[:-1], dim=-1)  # (T-1, B)
        if diffs.shape[0] == 0:
            vel = torch.zeros(T, B, device=z0.device)
        else:
            vel = torch.cat([diffs, diffs[-1:]], dim=0)

        return {
            "time_points_days": list(time_points_days),
            "latent_trajectory": z_traj.detach().cpu().numpy(),
            "feature_trajectory": feature_traj.detach().cpu().numpy(),
            "band_trajectories": band_np,
            "latent_velocity": vel.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def compare_drug_trajectories(
        self,
        eeg_features: Optional[torch.Tensor],
        disease_label: torch.Tensor,
        baseline_emb: torch.Tensor,
        donepezil_emb: torch.Tensor,
        memantine_emb: torch.Tensor,
        time_points_days: Optional[Sequence[int]] = None,
        z0: Optional[torch.Tensor] = None,
    ) -> dict:
        if time_points_days is None:
            time_points_days = list(DEFAULT_DAYS)

        results = {}
        for name, emb in [
            ("baseline", baseline_emb),
            ("donepezil", donepezil_emb),
            ("memantine", memantine_emb),
        ]:
            if emb.dim() == 1:
                emb_b = emb.unsqueeze(0).expand(disease_label.shape[0], -1)
            else:
                emb_b = emb
            results[name] = self.simulate_trajectory(
                eeg_features=eeg_features,
                drug_embedding=emb_b,
                disease_label=disease_label,
                time_points_days=time_points_days,
                z0=z0,
            )

        base_z = results["baseline"]["latent_trajectory"]
        done_z = results["donepezil"]["latent_trajectory"]
        mem_z = results["memantine"]["latent_trajectory"]

        def _mean_l2(a, b):
            return np.linalg.norm(a - b, axis=-1).mean(axis=1)

        results["divergence"] = {
            "donepezil_from_baseline": _mean_l2(done_z, base_z),
            "memantine_from_baseline": _mean_l2(mem_z, base_z),
            "donepezil_from_memantine": _mean_l2(done_z, mem_z),
        }
        return results


def train_ode(
    model: LatentODEModel,
    train_data: dict,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
) -> dict:
    """
    Train only DrugODEFunction so z(t=1) matches CVAE drug latents,
    with smoothness regularization and baseline near-identity dynamics.
    """
    device = torch.device(device)
    model = model.to(device)
    model.ode_func.train()
    if model.cvae is not None:
        model.cvae.eval()
        model.drug_encoder.eval()

    z0 = torch.tensor(train_data["z0_baseline"], dtype=torch.float32, device=device)
    z_done = torch.tensor(train_data["z_target_donepezil"], dtype=torch.float32, device=device)
    z_mem = torch.tensor(train_data["z_target_memantine"], dtype=torch.float32, device=device)
    disease = torch.tensor(train_data["disease_labels"], dtype=torch.float32, device=device)

    emb_done = torch.tensor(train_data["drug_embedding_donepezil"], dtype=torch.float32, device=device)
    emb_mem = torch.tensor(train_data["drug_embedding_memantine"], dtype=torch.float32, device=device)
    emb_base = torch.tensor(train_data["drug_embedding_baseline"], dtype=torch.float32, device=device)

    with torch.no_grad():
        drug_lat_done = model.drug_encoder(emb_done.unsqueeze(0)).squeeze(0)
        drug_lat_mem = model.drug_encoder(emb_mem.unsqueeze(0)).squeeze(0)
        drug_lat_base = model.drug_encoder(emb_base.unsqueeze(0)).squeeze(0)

    optimizer = torch.optim.Adam(model.ode_func.parameters(), lr=lr)
    t_span = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32, device=device)
    n = z0.shape[0]
    history = []

    last_done = last_mem = last_base = last_smooth = 0.0

    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n, device=device)
        epoch_losses = {"done": [], "mem": [], "base": [], "smooth": [], "total": []}

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            zb = z0[idx]
            zb_done = z_done[idx]
            zb_mem = z_mem[idx]
            # disease unused in latent endpoint loss but kept for API parity
            _ = disease[idx]

            optimizer.zero_grad()
            total = torch.tensor(0.0, device=device)
            smooth_acc = torch.tensor(0.0, device=device)

            # Donepezil
            dlat = drug_lat_done.unsqueeze(0).expand(zb.shape[0], -1)
            traj = model._integrate(zb, dlat, t_span, method=model.ode_method)
            done_loss = F.mse_loss(traj[-1], zb_done)
            mid_t = t_span[1]
            mid_z = traj[1]
            model.ode_func.set_drug_latent(dlat)
            vel = model.ode_func(mid_t, mid_z)
            smooth = (vel ** 2).mean()
            total = total + done_loss + 0.01 * smooth
            smooth_acc = smooth_acc + smooth

            # Memantine
            mlat = drug_lat_mem.unsqueeze(0).expand(zb.shape[0], -1)
            traj_m = model._integrate(zb, mlat, t_span, method=model.ode_method)
            mem_loss = F.mse_loss(traj_m[-1], zb_mem)
            model.ode_func.set_drug_latent(mlat)
            vel_m = model.ode_func(t_span[1], traj_m[1])
            smooth_m = (vel_m ** 2).mean()
            total = total + mem_loss + 0.01 * smooth_m
            smooth_acc = smooth_acc + smooth_m

            # Baseline should stay near z0
            blat = drug_lat_base.unsqueeze(0).expand(zb.shape[0], -1)
            traj_b = model._integrate(zb, blat, t_span, method=model.ode_method)
            base_loss = F.mse_loss(traj_b[-1], zb)
            total = total + 0.5 * base_loss

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.ode_func.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses["done"].append(float(done_loss.item()))
            epoch_losses["mem"].append(float(mem_loss.item()))
            epoch_losses["base"].append(float(base_loss.item()))
            epoch_losses["smooth"].append(float((smooth_acc / 2).item()))
            epoch_losses["total"].append(float(total.item()))

        last_done = float(np.mean(epoch_losses["done"]))
        last_mem = float(np.mean(epoch_losses["mem"]))
        last_base = float(np.mean(epoch_losses["base"]))
        last_smooth = float(np.mean(epoch_losses["smooth"]))
        history.append(
            {
                "epoch": epoch,
                "done_loss": last_done,
                "mem_loss": last_mem,
                "base_loss": last_base,
                "smooth": last_smooth,
                "total": float(np.mean(epoch_losses["total"])),
            }
        )
        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(
                f"Epoch {epoch}: Donep_loss={last_done:.4f}  Mem_loss={last_mem:.4f}  "
                f"Base_loss={last_base:.4f}  Smooth={last_smooth:.4f}"
            )

    return {
        "history": history,
        "final": {
            "done_loss": last_done,
            "mem_loss": last_mem,
            "base_loss": last_base,
            "smooth": last_smooth,
        },
    }
