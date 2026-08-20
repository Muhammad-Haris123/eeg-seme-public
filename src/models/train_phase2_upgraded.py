"""
Upgraded Digital Twin training with architecture variants.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.drugs.pharmacological_embedding import build_pharmacodynamic_vector
from src.models.config import DATA_CONFIG, MODEL_CONFIG, MODEL_VARIANTS, TRAINING_CONFIG
from src.models.cvae import CVAE
from src.models.data_loader import create_data_loaders_for_fold, get_subject_level_kfold, load_drug_embeddings
from src.models.drug_encoder import DrugEncoder
from src.models.drug_encoder_enhanced import EnhancedDrugEncoder
from src.models.eeg_encoder import EEGEncoder
from src.models.eeg_encoder_gnn import get_gnn_encoder
from src.models.fusion import FusionModule
from src.models.cross_attention_fusion import CrossModalAttentionFusion
from src.models.losses import compute_constrained_total_loss, compute_total_loss
from src.models.train_phase2 import prepare_target_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class UpgradedDigitalTwinModel(nn.Module):
    """
    Configurable model supporting baseline/gnn/attention/enhanced-drug variants.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.use_gnn_encoder = bool(config.get("use_gnn_encoder", False))
        self.use_enhanced_drug = bool(config.get("use_enhanced_drug", False))
        self.use_cross_attention = bool(config.get("use_cross_attention", False))

        eeg_cfg = MODEL_CONFIG["eeg_encoder"]
        drug_cfg = MODEL_CONFIG["drug_encoder"]
        fusion_cfg = MODEL_CONFIG["fusion"]
        cvae_cfg = MODEL_CONFIG["cvae"]

        self.eeg_latent_dim = int(eeg_cfg["eeg_latent_dim"])
        self.drug_latent_dim = int(drug_cfg["drug_latent_dim"])
        self.fusion_dim = int(fusion_cfg["fused_dim"])
        self.psd_dim = int(eeg_cfg["psd_dim"])
        self.n_channels = int(eeg_cfg["n_channels"])
        self.n_bands = int(eeg_cfg["n_bands"])

        if self.use_gnn_encoder:
            self.eeg_encoder = get_gnn_encoder(input_dim=2185, latent_dim=self.eeg_latent_dim)
        else:
            self.eeg_encoder = EEGEncoder(**eeg_cfg)

        if self.use_enhanced_drug:
            self.drug_encoder = EnhancedDrugEncoder(
                chemberta_dim=384,
                pharma_dim=32,
                drug_latent_dim=self.drug_latent_dim,
            )
            self.drug_input_dim = 416
            # cached priors for baseline/donepezil/memantine by drug_id
            self.register_buffer(
                "_prior_lookup",
                torch.stack(
                    [
                        torch.tensor(build_pharmacodynamic_vector("baseline"), dtype=torch.float32),
                        torch.tensor(build_pharmacodynamic_vector("donepezil"), dtype=torch.float32),
                        torch.tensor(build_pharmacodynamic_vector("memantine"), dtype=torch.float32),
                    ],
                    dim=0,
                ),
            )
        else:
            self.drug_encoder = DrugEncoder(**drug_cfg)
            self.drug_input_dim = 384
            self._prior_lookup = None

        if self.use_cross_attention:
            self.fusion = CrossModalAttentionFusion(
                eeg_latent_dim=self.eeg_latent_dim,
                drug_latent_dim=self.drug_latent_dim,
                fusion_dim=self.fusion_dim,
            )
            self.has_attention = True
        else:
            self.fusion = FusionModule(
                eeg_latent_dim=self.eeg_latent_dim,
                drug_latent_dim=self.drug_latent_dim,
                fused_dim=self.fusion_dim,
                use_attention=False,
            )
            self.has_attention = False

        psd_features = self.n_channels * self.psd_dim
        band_power_features = self.n_channels * self.n_bands
        connectivity_features = self.n_bands * (self.n_channels * (self.n_channels - 1) // 2) * 2
        output_dim = psd_features + band_power_features + connectivity_features
        condition_dim = self.drug_latent_dim + 1
        self.cvae = CVAE(
            fused_dim=self.fusion_dim,
            latent_dim=cvae_cfg["latent_dim"],
            output_dim=output_dim,
            condition_dim=condition_dim,
            hidden_dims=cvae_cfg["hidden_dims"],
        )

    def _flatten_eeg_features(
        self, psd: torch.Tensor, band_powers: torch.Tensor, coherence: torch.Tensor, plv: torch.Tensor
    ) -> torch.Tensor:
        return prepare_target_features(psd, band_powers, coherence, plv)

    def _prepare_drug_embedding(self, drug_embedding: torch.Tensor, drug_id: torch.Tensor | None) -> torch.Tensor:
        if not self.use_enhanced_drug:
            return drug_embedding
        if drug_embedding.shape[1] == 416:
            return drug_embedding
        if drug_embedding.shape[1] != 384:
            raise ValueError(f"Expected drug embedding dim 384/416, got {drug_embedding.shape[1]}")
        if drug_id is None:
            raise ValueError("drug_id is required for enhanced drug mode when input is 384-D.")
        # map -1,0,1 => lookup rows 0,1,2
        idx = torch.where(drug_id < 0, torch.zeros_like(drug_id), drug_id + 1).long().clamp(0, 2)
        prior = self._prior_lookup[idx].to(drug_embedding.device)
        return torch.cat([drug_embedding, prior], dim=1)

    def forward(
        self,
        psd: torch.Tensor,
        band_powers: torch.Tensor,
        coherence: torch.Tensor,
        plv: torch.Tensor,
        drug_embedding: torch.Tensor,
        disease_label: torch.Tensor,
        return_latent: bool = False,
        drug_id: torch.Tensor | None = None,
    ) -> tuple:
        if self.use_gnn_encoder:
            eeg_flat = self._flatten_eeg_features(psd, band_powers, coherence, plv)
            eeg_latent = self.eeg_encoder(eeg_flat)
        else:
            eeg_latent = self.eeg_encoder(psd, band_powers, coherence, plv)

        drug_input = self._prepare_drug_embedding(drug_embedding, drug_id)
        drug_latent = self.drug_encoder(drug_input)

        fused = self.fusion(eeg_latent, drug_latent, disease_label)
        condition = torch.cat(
            [drug_latent, disease_label.unsqueeze(1) if disease_label.dim() == 1 else disease_label],
            dim=1,
        )

        if return_latent:
            reconstructed, mu, logvar, z = self.cvae(fused, condition, return_latent=True)
            return reconstructed, mu, logvar, z, eeg_latent, drug_latent
        reconstructed, mu, logvar = self.cvae(fused, condition, return_latent=False)
        return reconstructed, mu, logvar, eeg_latent, drug_latent

    def get_attention_weights(self):
        if self.has_attention:
            return self.fusion.get_attention_weights()
        return None


def _build_subject_baseline_cache(loader) -> Dict[int, np.ndarray]:
    cache = {}
    base = loader.dataset.base
    for subject_id in range(base.n_subjects):
        psd_arr, bp_arr, coh_arr, plv_arr, _, local_idx = base._get_group_arrays(subject_id)
        psd = torch.tensor(psd_arr[local_idx], dtype=torch.float32).unsqueeze(0)
        bp = torch.tensor(bp_arr[local_idx], dtype=torch.float32).unsqueeze(0)
        coh = torch.tensor(coh_arr[local_idx], dtype=torch.float32).unsqueeze(0)
        plv = torch.tensor(plv_arr[local_idx], dtype=torch.float32).unsqueeze(0)
        cache[int(subject_id)] = prepare_target_features(psd, bp, coh, plv).squeeze(0).numpy().astype(np.float32)
    return cache


def _lookup_baseline_batch(subject_idx: torch.Tensor, cache: Dict[int, np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack([cache[int(s)] for s in subject_idx.cpu().numpy()], axis=0)
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _train_epoch_variant(
    model: UpgradedDigitalTwinModel,
    train_loader,
    optimizer,
    device,
    epoch: int,
    use_constrained_loss: bool,
    baseline_cache: Dict[int, np.ndarray] | None,
    constraint_weight: float,
    warmup_epochs: int,
) -> Dict[str, float]:
    model.train()
    sums = {"total": 0.0, "mse": 0.0, "kl": 0.0, "band": 0.0, "conn": 0.0}
    n_batches = 0
    beta_kl = TRAINING_CONFIG["beta_kl"]
    for batch in train_loader:
        psd = batch["psd"].to(device)
        bp = batch["band_powers"].to(device)
        coh = batch["coherence"].to(device)
        plv = batch["plv"].to(device)
        drug = batch["drug_embedding"].to(device)
        disease = batch["disease_label"].to(device)
        drug_id = batch["drug_id"].to(device)
        subj = batch["subject_idx"].to(device)
        target = prepare_target_features(psd, bp, coh, plv)

        reconstructed, mu, logvar, _, _ = model(
            psd, bp, coh, plv, drug, disease, return_latent=False, drug_id=drug_id
        )

        if use_constrained_loss and baseline_cache is not None:
            baseline = _lookup_baseline_batch(subj, baseline_cache, device)
            total = torch.tensor(0.0, device=device)
            mse = torch.tensor(0.0, device=device)
            kl = torch.tensor(0.0, device=device)
            band = torch.tensor(0.0, device=device)
            conn = torch.tensor(0.0, device=device)
            bsz = float(drug_id.shape[0])
            for did in torch.unique(drug_id).tolist():
                mask = drug_id == int(did)
                if int(mask.sum()) == 0:
                    continue
                frac = float(mask.sum().item()) / bsz
                drug_name = "baseline" if int(did) < 0 else ("donepezil" if int(did) == 0 else "memantine")
                losses = compute_constrained_total_loss(
                    reconstruction=reconstructed[mask],
                    target_features=target[mask],
                    mu=mu[mask],
                    log_var=logvar[mask],
                    drug_name=drug_name,
                    baseline_features=baseline[mask],
                    beta=beta_kl,
                    constraint_weight=constraint_weight,
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
                )
                total = total + losses["total"] * frac
                mse = mse + losses["mse"] * frac
                kl = kl + losses["kl"] * frac
                band = band + losses["band_constraint"] * frac
                conn = conn + losses["connectivity_constraint"] * frac
        else:
            losses = compute_total_loss(
                reconstructed=reconstructed,
                target=target,
                mu=mu,
                logvar=logvar,
                beta_kl=beta_kl,
                beta_condition=TRAINING_CONFIG.get("beta_condition", 0.0),
                use_condition_loss=False,
            )
            total = losses["total"]
            mse = losses["reconstruction"]
            kl = losses["kl"]
            band = torch.tensor(0.0, device=device)
            conn = torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        sums["total"] += float(total.item())
        sums["mse"] += float(mse.item())
        sums["kl"] += float(kl.item())
        sums["band"] += float(band.item())
        sums["conn"] += float(conn.item())
        n_batches += 1

    for k in sums:
        sums[k] /= max(n_batches, 1)
    return sums


@torch.no_grad()
def _validate_epoch_variant(
    model: UpgradedDigitalTwinModel,
    val_loader,
    device,
    epoch: int,
    use_constrained_loss: bool,
    baseline_cache: Dict[int, np.ndarray] | None,
    constraint_weight: float,
    warmup_epochs: int,
) -> Dict[str, float]:
    model.eval()
    sums = {"total": 0.0, "mse": 0.0, "kl": 0.0, "band": 0.0, "conn": 0.0}
    n_batches = 0
    beta_kl = TRAINING_CONFIG["beta_kl"]
    for batch in val_loader:
        psd = batch["psd"].to(device)
        bp = batch["band_powers"].to(device)
        coh = batch["coherence"].to(device)
        plv = batch["plv"].to(device)
        drug = batch["drug_embedding"].to(device)
        disease = batch["disease_label"].to(device)
        drug_id = batch["drug_id"].to(device)
        subj = batch["subject_idx"].to(device)
        target = prepare_target_features(psd, bp, coh, plv)
        reconstructed, mu, logvar, _, _ = model(
            psd, bp, coh, plv, drug, disease, return_latent=False, drug_id=drug_id
        )

        if use_constrained_loss and baseline_cache is not None:
            baseline = _lookup_baseline_batch(subj, baseline_cache, device)
            total = torch.tensor(0.0, device=device)
            mse = torch.tensor(0.0, device=device)
            kl = torch.tensor(0.0, device=device)
            band = torch.tensor(0.0, device=device)
            conn = torch.tensor(0.0, device=device)
            bsz = float(drug_id.shape[0])
            for did in torch.unique(drug_id).tolist():
                mask = drug_id == int(did)
                if int(mask.sum()) == 0:
                    continue
                frac = float(mask.sum().item()) / bsz
                drug_name = "baseline" if int(did) < 0 else ("donepezil" if int(did) == 0 else "memantine")
                losses = compute_constrained_total_loss(
                    reconstruction=reconstructed[mask],
                    target_features=target[mask],
                    mu=mu[mask],
                    log_var=logvar[mask],
                    drug_name=drug_name,
                    baseline_features=baseline[mask],
                    beta=beta_kl,
                    constraint_weight=constraint_weight,
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
                )
                total = total + losses["total"] * frac
                mse = mse + losses["mse"] * frac
                kl = kl + losses["kl"] * frac
                band = band + losses["band_constraint"] * frac
                conn = conn + losses["connectivity_constraint"] * frac
        else:
            losses = compute_total_loss(
                reconstructed=reconstructed,
                target=target,
                mu=mu,
                logvar=logvar,
                beta_kl=beta_kl,
                beta_condition=TRAINING_CONFIG.get("beta_condition", 0.0),
                use_condition_loss=False,
            )
            total = losses["total"]
            mse = losses["reconstruction"]
            kl = losses["kl"]
            band = torch.tensor(0.0, device=device)
            conn = torch.tensor(0.0, device=device)

        sums["total"] += float(total.item())
        sums["mse"] += float(mse.item())
        sums["kl"] += float(kl.item())
        sums["band"] += float(band.item())
        sums["conn"] += float(conn.item())
        n_batches += 1

    for k in sums:
        sums[k] /= max(n_batches, 1)
    return sums


def train_upgraded_model(
    config: dict,
    n_splits: int = 5,
    epochs: int = 100,
    variant_name: str = "gnn_attention",
    constraint_weight: float = 0.1,
    warmup_epochs: int = 5,
) -> dict:
    """
    Train UpgradedDigitalTwinModel with 5-fold CV and save report/checkpoints.
    """
    use_constrained_loss = bool(config.get("use_constrained_loss", True))
    out_ckpt_dir = Path(f"models/checkpoints_{variant_name}")
    out_eval = Path(f"models/evaluation/{variant_name}_evaluation_report.json")
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_eval.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(TRAINING_CONFIG["device"])
    random_seed = DATA_CONFIG["random_seed"]
    batch_size = TRAINING_CONFIG["batch_size"]

    subject_ids, labels, test_folds = get_subject_level_kfold(n_splits=n_splits, random_seed=random_seed)
    per_fold = {}
    training_curves = {}
    timing = []

    for fold_idx in range(n_splits):
        test_ids = test_folds[fold_idx].tolist()
        train_val_ids = np.concatenate([test_folds[j] for j in range(n_splits) if j != fold_idx])
        train_ids, val_ids = train_test_split(
            train_val_ids.tolist(),
            test_size=0.1,
            stratify=labels[train_val_ids],
            random_state=random_seed + fold_idx,
        )
        train_loader, val_loader, test_loader = create_data_loaders_for_fold(
            train_ids, val_ids, test_ids, batch_size=batch_size, shuffle_train=True
        )
        baseline_cache = _build_subject_baseline_cache(train_loader) if use_constrained_loss else None

        model = UpgradedDigitalTwinModel(config=config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )
        best_path = out_ckpt_dir / f"fold_{fold_idx}_best.pt"
        best_val = float("inf")
        curves = []
        t0 = time.time()

        for epoch in range(epochs):
            tr = _train_epoch_variant(
                model,
                train_loader,
                optimizer,
                device,
                epoch=epoch,
                use_constrained_loss=use_constrained_loss,
                baseline_cache=baseline_cache,
                constraint_weight=constraint_weight,
                warmup_epochs=warmup_epochs,
            )
            va = _validate_epoch_variant(
                model,
                val_loader,
                device,
                epoch=epoch,
                use_constrained_loss=use_constrained_loss,
                baseline_cache=baseline_cache,
                constraint_weight=constraint_weight,
                warmup_epochs=warmup_epochs,
            )
            curves.append({"epoch": epoch + 1, "train": tr, "val": va})
            print(
                f"[{variant_name}] Fold {fold_idx} Epoch {epoch+1}: "
                f"MSE={tr['mse']:.4f} KL={tr['kl']:.4f} Band={tr['band']:.4f} Conn={tr['conn']:.4f} Total={tr['total']:.4f}"
            )
            if va["total"] < best_val:
                best_val = va["total"]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": va,
                        "variant_name": variant_name,
                        "config": config,
                    },
                    best_path,
                )

        timing.append(time.time() - t0)
        training_curves[f"fold_{fold_idx}"] = curves

        best_model = UpgradedDigitalTwinModel(config=config).to(device)
        ckpt = torch.load(best_path, map_location=device)
        best_model.load_state_dict(ckpt["model_state_dict"])
        rec = _compute_reconstruction_metrics_fold_upgraded(best_model, test_loader, device)
        acc = _compute_accuracy_fold_upgraded(
            train_loader, test_loader, best_model, device, random_seed=random_seed
        )
        per_fold[f"fold_{fold_idx}"] = {
            "reconstruction_mse": rec["reconstruction_mse"],
            "pearson_global": rec["pearson_global"],
            "pearson_per_feature_mean": rec["pearson_per_feature_mean"],
            "accuracy": acc,
            "n_test_subjects": len(test_ids),
            "n_train_subjects": len(train_ids),
            "checkpoint": str(best_path),
            "seconds": timing[-1],
        }

    mse_list = [per_fold[f"fold_{i}"]["reconstruction_mse"] for i in range(n_splits)]
    pearson_list = [per_fold[f"fold_{i}"]["pearson_global"] for i in range(n_splits)]
    acc_list = [per_fold[f"fold_{i}"]["accuracy"] for i in range(n_splits)]
    aggregated = {
        "reconstruction_mse_mean": float(np.mean(mse_list)),
        "reconstruction_mse_std": float(np.std(mse_list)),
        "pearson_global_mean": float(np.mean(pearson_list)),
        "pearson_global_std": float(np.std(pearson_list)),
        "accuracy_mean": float(np.mean(acc_list)),
        "accuracy_std": float(np.std(acc_list)),
        "training_time_per_fold_mean_s": float(np.mean(timing)),
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "variant_name": variant_name,
        "variant_config": config,
        "evaluation_design": {
            "description": "5-fold subject-level stratified cross-validation",
            "n_folds": n_splits,
            "random_seed": random_seed,
        },
        "per_fold": per_fold,
        "aggregated": aggregated,
        "constraint_curves": training_curves,
    }
    out_eval.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@torch.no_grad()
def _extract_latent_representations_upgraded(
    model: UpgradedDigitalTwinModel,
    data_loader,
    device: torch.device,
) -> dict:
    model.eval()
    all_mu = []
    all_disease = []
    all_subject_idx = []
    for batch in data_loader:
        psd = batch["psd"].to(device)
        bp = batch["band_powers"].to(device)
        coh = batch["coherence"].to(device)
        plv = batch["plv"].to(device)
        drug = batch["drug_embedding"].to(device)
        disease = batch["disease_label"].to(device)
        drug_id = batch["drug_id"].to(device)

        _, mu, _, _, _ = model(
            psd, bp, coh, plv, drug, disease, return_latent=False, drug_id=drug_id
        )
        all_mu.append(mu.cpu().numpy())
        all_disease.append(disease.cpu().numpy())
        all_subject_idx.append(batch["subject_idx"].numpy())
    return {
        "cvae_mu": np.concatenate(all_mu, axis=0),
        "disease_labels": np.concatenate(all_disease, axis=0),
        "subject_indices": np.concatenate(all_subject_idx, axis=0),
    }


@torch.no_grad()
def _compute_reconstruction_metrics_fold_upgraded(
    model: UpgradedDigitalTwinModel,
    test_loader,
    device: torch.device,
) -> dict:
    model.eval()
    all_targets = []
    all_recon = []
    for batch in test_loader:
        psd = batch["psd"].to(device)
        bp = batch["band_powers"].to(device)
        coh = batch["coherence"].to(device)
        plv = batch["plv"].to(device)
        drug = batch["drug_embedding"].to(device)
        disease = batch["disease_label"].to(device)
        drug_id = batch["drug_id"].to(device)
        target = prepare_target_features(psd, bp, coh, plv)
        reconstructed, _, _, _, _ = model(
            psd, bp, coh, plv, drug, disease, return_latent=False, drug_id=drug_id
        )
        all_targets.append(target.cpu().numpy())
        all_recon.append(reconstructed.cpu().numpy())
    targets = np.concatenate(all_targets, axis=0)
    recon = np.concatenate(all_recon, axis=0)
    feat_dim = targets.shape[1]
    mse = float(np.mean((targets - recon) ** 2))
    t_flat = targets.ravel()
    r_flat = recon.ravel()
    pearson_global = float(np.corrcoef(t_flat, r_flat)[0, 1]) if np.std(t_flat) > 1e-12 and np.std(r_flat) > 1e-12 else 0.0
    pearson_per_feat = []
    for j in range(feat_dim):
        t_j, r_j = targets[:, j], recon[:, j]
        if np.std(t_j) > 1e-12 and np.std(r_j) > 1e-12:
            pearson_per_feat.append(np.corrcoef(t_j, r_j)[0, 1])
        else:
            pearson_per_feat.append(0.0)
    pearson_mean = float(np.mean(np.abs(pearson_per_feat)))
    return {
        "reconstruction_mse": mse,
        "pearson_global": pearson_global,
        "pearson_per_feature_mean": pearson_mean,
        "n_test_samples": int(targets.shape[0]),
    }


def _compute_accuracy_fold_upgraded(
    train_loader,
    test_loader,
    model: UpgradedDigitalTwinModel,
    device: torch.device,
    random_seed: int = 42,
) -> float:
    lat_train = _extract_latent_representations_upgraded(model, train_loader, device)
    lat_test = _extract_latent_representations_upgraded(model, test_loader, device)
    mu_train = lat_train["cvae_mu"]
    mu_test = lat_test["cvae_mu"]
    y_train = lat_train["disease_labels"].ravel().astype(int)
    y_test = lat_test["disease_labels"].ravel().astype(int)
    subj_train = lat_train["subject_indices"].ravel()
    subj_test = lat_test["subject_indices"].ravel()
    _, idx_train = np.unique(subj_train, return_index=True)
    _, idx_test = np.unique(subj_test, return_index=True)
    X_train = mu_train[idx_train]
    X_test = mu_test[idx_test]
    y_train = y_train[idx_train]
    y_test = y_test[idx_test]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=500, random_state=random_seed)
    clf.fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)
    return float(np.mean(pred == y_test))

