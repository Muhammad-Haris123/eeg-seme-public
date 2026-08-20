"""
Explainable AI for Phase 2: Integrated Gradients (+ optional SmoothGrad), Feature Ablation,
optional SHAP GradientExplainer on a scalar reconstruction summary, and CAM-style heatmaps.

Outputs:
  models/evaluation/xai_report.json
  models/evaluation/xai_report_shap.json (if SHAP succeeds)
  models/evaluation/figures/xai_*.png

Run from project root:
  python -m src.explainability.explain_shap
"""

from __future__ import annotations

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import (
    EVALUATION_DIR,
    CHECKPOINTS_DIR,
)
from src.models.train_phase2 import prepare_target_features
from src.models.inference import load_trained_model
from src.models.data_loader import create_data_loaders
from src.explainability.integrated_gradients import _expand_labels_to_batch
from src.explainability.model_wrapper import (
    DigitalTwinWrapper,
    PSD_SIZE,
    BAND_SIZE,
    CONNECTIVITY_SIZE,
    EEG_FLAT_SIZE,
    DRUG_DIM,
)
from src.explainability.xai_visualizations import (
    save_psd_band_heatmaps,
    save_connectivity_band_maps,
    save_drug_embedding_importance,
)


def get_sample_batch(loader, max_samples=50, device="cpu"):
    """Collect up to max_samples (eeg_flat, drug_emb, disease_label) from loader."""
    eeg_flats = []
    drug_embs = []
    disease_labels = []
    for batch in loader:
        psd = batch["psd"]
        band_powers = batch["band_powers"]
        coherence = batch["coherence"]
        plv = batch["plv"]
        target = prepare_target_features(psd, band_powers, coherence, plv)
        eeg_flats.append(target.numpy())
        drug_embs.append(batch["drug_embedding"].numpy())
        disease_labels.append(batch["disease_label"].numpy())
        if sum(x.shape[0] for x in eeg_flats) >= max_samples:
            break
    eeg_flat = np.concatenate(eeg_flats, axis=0)[:max_samples]
    drug_emb = np.concatenate(drug_embs, axis=0)[:max_samples]
    disease_label = np.concatenate(disease_labels, axis=0)[:max_samples]
    return (
        torch.FloatTensor(eeg_flat).to(device),
        torch.FloatTensor(drug_emb).to(device),
        torch.FloatTensor(disease_label).to(device),
    )


def compute_train_baselines(train_loader, device, max_batches: int = 50):
    """Mean EEG feature vector and mean drug embedding on training batches (reference for IG)."""
    sum_eeg = torch.zeros(EEG_FLAT_SIZE, device=device)
    sum_drug = torch.zeros(DRUG_DIM, device=device)
    n = 0
    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break
        psd = batch["psd"].to(device)
        band_powers = batch["band_powers"].to(device)
        coherence = batch["coherence"].to(device)
        plv = batch["plv"].to(device)
        target = prepare_target_features(psd, band_powers, coherence, plv)
        drug = batch["drug_embedding"].to(device)
        bs = target.shape[0]
        sum_eeg += target.sum(dim=0)
        sum_drug += drug.sum(dim=0)
        n += bs
    if n == 0:
        return None, None
    return sum_eeg / n, sum_drug / n


def run_integrated_gradients(
    wrapper,
    eeg_flat,
    drug_emb,
    disease_label,
    device,
    baseline_eeg=None,
    baseline_drug=None,
    n_steps: int = 32,
):
    """Integrated Gradients; baselines default to zeros if None."""
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        return run_saliency_fallback(wrapper, eeg_flat, drug_emb, disease_label, device)

    wrapper.eval()
    if baseline_eeg is None:
        baseline_eeg = torch.zeros_like(eeg_flat)
    if baseline_drug is None:
        baseline_drug = torch.zeros_like(drug_emb)

    def _forward(e, d):
        dl = _expand_labels_to_batch(disease_label, e.shape[0])
        out = wrapper.forward_reconstruction(e, d, dl, enable_grad=True)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return out

    ig = IntegratedGradients(_forward)
    attr_eeg, attr_drug = ig.attribute(
        (eeg_flat, drug_emb),
        baselines=(baseline_eeg, baseline_drug),
        n_steps=n_steps,
    )
    return attr_eeg, attr_drug, None


def run_smoothgrad_integrated_gradients(
    wrapper,
    eeg_flat,
    drug_emb,
    disease_label,
    device,
    baseline_eeg,
    baseline_drug,
    n_steps: int = 32,
    nt_samples: int = 8,
    stdevs: float = 0.01,
):
    """NoiseTunnel + IG (SmoothGrad-style)."""
    try:
        from captum.attr import IntegratedGradients, NoiseTunnel
    except ImportError:
        return run_integrated_gradients(
            wrapper, eeg_flat, drug_emb, disease_label, device, baseline_eeg, baseline_drug, n_steps
        )

    wrapper.eval()
    if baseline_eeg is None:
        baseline_eeg = torch.zeros_like(eeg_flat)
    if baseline_drug is None:
        baseline_drug = torch.zeros_like(drug_emb)

    def _forward(e, d):
        dl = _expand_labels_to_batch(disease_label, e.shape[0])
        out = wrapper.forward_reconstruction(e, d, dl, enable_grad=True)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return out

    ig = IntegratedGradients(_forward)
    nt = NoiseTunnel(ig)
    # Tuple stdevs for multi-input (Captum)
    st = (stdevs, stdevs)
    attr_eeg, attr_drug = nt.attribute(
        (eeg_flat, drug_emb),
        baselines=(baseline_eeg, baseline_drug),
        n_steps=n_steps,
        nt_type="smoothgrad",
        nt_samples=nt_samples,
        stdevs=st,
    )
    return attr_eeg, attr_drug, None


def run_saliency_fallback(wrapper, eeg_flat, drug_emb, disease_label, device):
    """Gradient saliency when Captum is not available."""
    wrapper.eval()
    eeg_flat = eeg_flat.detach().requires_grad_(True)
    drug_emb = drug_emb.detach().requires_grad_(True)
    recon = wrapper(eeg_flat, drug_emb, disease_label)
    loss = recon.sum()
    loss.backward()
    return eeg_flat.grad.clone(), drug_emb.grad.clone(), None


def summarize_attributions(attr_eeg, attr_drug):
    """Summarize to block-level and top indices."""
    if attr_eeg is None:
        return {}
    ae = attr_eeg.detach().cpu().numpy()
    ad = attr_drug.detach().cpu().numpy()
    report = {
        "eeg_blocks": {
            "psd": float(np.mean(np.abs(ae[:, :PSD_SIZE]))),
            "band_powers": float(np.mean(np.abs(ae[:, PSD_SIZE : PSD_SIZE + BAND_SIZE]))),
            "connectivity": float(np.mean(np.abs(ae[:, PSD_SIZE + BAND_SIZE :]))),
        },
        "drug_embedding_mean_abs": float(np.mean(np.abs(ad))),
        "drug_embedding_std": float(np.std(ad)),
    }
    mean_abs_eeg = np.mean(np.abs(ae), axis=0)
    top_k = 20
    top_indices = np.argsort(mean_abs_eeg)[-top_k:][::-1].tolist()
    report["top_eeg_feature_indices"] = top_indices
    report["top_eeg_feature_means"] = [float(mean_abs_eeg[i]) for i in top_indices]
    block_names = []
    for i in top_indices:
        if i < PSD_SIZE:
            block_names.append("psd")
        elif i < PSD_SIZE + BAND_SIZE:
            block_names.append("band_powers")
        else:
            block_names.append("connectivity")
    report["top_eeg_blocks"] = block_names
    return report


def _feature_group_masks(device):
    """Four groups on EEG flat: PSD, bands, connectivity; drug fourth group."""
    m_eeg = torch.zeros(EEG_FLAT_SIZE, dtype=torch.long, device=device)
    m_eeg[:PSD_SIZE] = 0
    m_eeg[PSD_SIZE : PSD_SIZE + BAND_SIZE] = 1
    m_eeg[PSD_SIZE + BAND_SIZE :] = 2
    m_drug = torch.full((DRUG_DIM,), 3, dtype=torch.long, device=device)
    return m_eeg, m_drug


def run_feature_ablation_block_scores(
    wrapper,
    eeg_flat,
    drug_emb,
    disease_label,
    device,
    max_samples: int = 16,
):
    """
    Captum Feature Ablation with coarse groups (PSD / bands / connectivity / drug).
    Labels are fixed per batch row (closure) so only EEG + drug tensors are ablated.
    """
    try:
        from captum.attr import FeatureAblation
    except ImportError:
        return {"status": "skipped", "reason": "captum not installed"}

    n = min(max_samples, eeg_flat.shape[0])
    eeg_flat = eeg_flat[:n].detach().clone()
    drug_emb = drug_emb[:n].detach().clone()
    lbl = disease_label[:n].detach().clone()

    wrapper.eval()

    def forward_pair(eeg, drug):
        return wrapper(eeg, drug, lbl)

    fa = FeatureAblation(forward_pair)
    m_eeg, m_drug = _feature_group_masks(device)

    try:
        attrs = fa.attribute(
            (eeg_flat, drug_emb),
            feature_mask=(m_eeg.unsqueeze(0).expand(n, -1), m_drug.unsqueeze(0).expand(n, -1)),
            perturbations_per_eval=1,
        )
        ae = attrs[0].detach().cpu().numpy()
        ad = attrs[1].detach().cpu().numpy()
        group_means = {
            "psd": float(np.mean(np.abs(ae[:, :PSD_SIZE]))),
            "band_powers": float(np.mean(np.abs(ae[:, PSD_SIZE : PSD_SIZE + BAND_SIZE]))),
            "connectivity": float(np.mean(np.abs(ae[:, PSD_SIZE + BAND_SIZE :]))),
            "drug_embedding": float(np.mean(np.abs(ad))),
        }
        return {"status": "ok", "group_mean_abs_attribution": group_means}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


class ScalarOutputTwin(torch.nn.Module):
    """Maps reconstruction to a scalar per sample (sum) for SHAP gradient explainer."""

    def __init__(self, wrapper: DigitalTwinWrapper):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, eeg_flat, drug_emb, disease_label):
        r = self.wrapper(eeg_flat, drug_emb, disease_label)
        return r.sum(dim=1, keepdim=True)


def run_shap_gradient_explainer(
    scalar_model: ScalarOutputTwin,
    bg_eeg: torch.Tensor,
    bg_drug: torch.Tensor,
    bg_lbl: torch.Tensor,
    test_eeg: torch.Tensor,
    test_drug: torch.Tensor,
    test_lbl: torch.Tensor,
):
    """SHAP GradientExplainer on scalar sum(reconstruction)."""
    try:
        import shap
    except ImportError:
        return {"status": "skipped", "reason": "shap package not installed"}

    try:
        explainer_cls = getattr(shap, "GradientExplainer", None)
        if explainer_cls is None:
            return {"status": "skipped", "reason": "shap.GradientExplainer not found"}

        bg_eeg = bg_eeg.detach()
        bg_drug = bg_drug.detach()
        bg_lbl = bg_lbl.detach()
        explainer = explainer_cls(scalar_model, (bg_eeg, bg_drug, bg_lbl))

        sv = explainer.shap_values((test_eeg.detach(), test_drug.detach(), test_lbl.detach()))

        # sv may be list for multi-output or ndarray
        if isinstance(sv, list):
            sv_eeg = sv[0]
            sv_drug = sv[1] if len(sv) > 1 else None
        else:
            sv_eeg = sv
            sv_drug = None

        out = {
            "status": "ok",
            "method": "shap.GradientExplainer",
            "target": "sum(reconstruction) per sample",
            "n_background": int(bg_eeg.shape[0]),
            "n_explained": int(test_eeg.shape[0]),
            "eeg_shap_mean_abs_blocks": None,
        }
        if sv_eeg is not None:
            se = np.asarray(sv_eeg)
            if se.ndim == 3:
                se = se.mean(axis=0)
            elif se.ndim == 2:
                pass
            mean_abs = np.mean(np.abs(se), axis=0)
            out["eeg_shap_mean_abs_blocks"] = {
                "psd": float(np.mean(mean_abs[:PSD_SIZE])),
                "band_powers": float(np.mean(mean_abs[PSD_SIZE : PSD_SIZE + BAND_SIZE])),
                "connectivity": float(np.mean(mean_abs[PSD_SIZE + BAND_SIZE :])),
            }
            out["drug_shap_mean_abs"] = (
                float(np.mean(np.abs(np.asarray(sv_drug)))) if sv_drug is not None else None
            )
        return out
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def run():
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = EVALUATION_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = CHECKPOINTS_DIR / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = CHECKPOINTS_DIR / "checkpoint_latest.pt"
    model = load_trained_model(checkpoint_path, device)
    wrapper = DigitalTwinWrapper(model).to(device)

    train_loader, _, test_loader = create_data_loaders(
        batch_size=32,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        random_seed=42,
        shuffle=False,
        split_by_subject=True,
    )

    baseline_eeg, baseline_drug = compute_train_baselines(train_loader, device, max_batches=60)
    # Keep 1×D mean vectors for broadcasting
    be_row = baseline_eeg.unsqueeze(0) if baseline_eeg is not None else None
    bd_row = baseline_drug.unsqueeze(0) if baseline_drug is not None else None

    eeg_flat, drug_emb, disease_label = get_sample_batch(test_loader, max_samples=48, device=device)

    if be_row is not None:
        be = be_row.expand(eeg_flat.shape[0], -1)
        bd = bd_row.expand(drug_emb.shape[0], -1)
    else:
        be, bd = None, None

    use_smooth = True
    bez = be if be is not None else torch.zeros_like(eeg_flat)
    bdz = bd if bd is not None else torch.zeros_like(drug_emb)
    if use_smooth:
        attr_eeg, attr_drug, err = run_smoothgrad_integrated_gradients(
            wrapper,
            eeg_flat,
            drug_emb,
            disease_label,
            device,
            bez,
            bdz,
            n_steps=32,
            nt_samples=6,
        )
        ig_method = (
            "NoiseTunnel(smoothgrad) + IntegratedGradients (Captum); "
            + ("train-mean baseline" if be is not None else "zero baseline")
        )
    else:
        attr_eeg, attr_drug, err = run_integrated_gradients(
            wrapper, eeg_flat, drug_emb, disease_label, device, bez, bdz, n_steps=32
        )
        ig_method = "IntegratedGradients (Captum)"

    if err:
        summary = {"status": "error", "message": err, "timestamp": datetime.now().isoformat()}
        with open(EVALUATION_DIR / "xai_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    summary = summarize_attributions(attr_eeg, attr_drug)
    summary["method"] = ig_method
    summary["baseline"] = (
        "training_batch_mean(eeg_features), training_batch_mean(drug_embedding)"
        if baseline_eeg is not None
        else "zeros"
    )
    summary["n_samples"] = int(eeg_flat.shape[0])
    summary["timestamp"] = datetime.now().isoformat()
    summary["status"] = "ok"
    summary["interpretation_note"] = (
        "Grad-CAM applies to convolutional spatial maps; this model uses engineered features. "
        "Use xai_*_importance.png heatmaps as channel × feature importance (CAM-like summarization)."
    )
    summary["feature_ablation"] = run_feature_ablation_block_scores(
        wrapper, eeg_flat, drug_emb, disease_label, device, max_samples=16
    )
    summary["summary_text"] = (
        "EEG blocks (mean |IG|): PSD=%.4f, band_powers=%.4f, connectivity=%.4f. "
        "Drug embedding mean |IG|=%.4f. Top blocks: %s."
    ) % (
        summary["eeg_blocks"]["psd"],
        summary["eeg_blocks"]["band_powers"],
        summary["eeg_blocks"]["connectivity"],
        summary["drug_embedding_mean_abs"],
        ", ".join(summary["top_eeg_blocks"][:5]),
    )

    # Mean |attribution| over samples for spatial plots
    ae_np = np.mean(np.abs(attr_eeg.detach().cpu().numpy()), axis=0)
    ad_np = np.mean(np.abs(attr_drug.detach().cpu().numpy()), axis=0)
    summary["figures"] = {}
    summary["figures"].update(save_psd_band_heatmaps(ae_np, fig_dir, prefix="xai"))
    summary["figures"].update(save_connectivity_band_maps(ae_np, fig_dir, prefix="xai"))
    dchunk = save_drug_embedding_importance(ad_np, fig_dir, prefix="xai")
    if dchunk:
        summary["figures"]["drug_chunks"] = dchunk

    # Bar chart (legacy block comparison)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3))
        blocks = list(summary["eeg_blocks"].keys())
        vals = [summary["eeg_blocks"][b] for b in blocks]
        vals.append(summary["drug_embedding_mean_abs"])
        blocks_l = blocks + ["drug_embedding"]
        ax.bar(blocks_l, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        ax.set_ylabel("Mean |attribution|")
        ax.set_title("Feature importance (IG / SmoothGrad)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        bp = fig_dir / "xai_block_importance.png"
        plt.savefig(bp, dpi=150)
        plt.close(fig)
        summary["figures"]["block_bar"] = str(bp.name)
    except Exception as ex:
        summary["figure_note"] = str(ex)

    # SHAP GradientExplainer (subset)
    bg_eeg, bg_drug, bg_lbl = get_sample_batch(train_loader, max_samples=24, device=device)
    test_eeg, test_drug, test_lbl = get_sample_batch(test_loader, max_samples=8, device=device)
    scalar_model = ScalarOutputTwin(wrapper).to(device)
    shap_report = run_shap_gradient_explainer(
        scalar_model,
        bg_eeg,
        bg_drug,
        bg_lbl,
        test_eeg,
        test_drug,
        test_lbl,
    )
    summary["shap_gradient_explainer"] = shap_report
    with open(EVALUATION_DIR / "xai_report_shap.json", "w", encoding="utf-8") as f:
        json.dump(shap_report, f, indent=2)

    with open(EVALUATION_DIR / "xai_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("XAI report saved to", EVALUATION_DIR / "xai_report.json")
    print("SHAP sidecar:", EVALUATION_DIR / "xai_report_shap.json")
    return summary


if __name__ == "__main__":
    run()
