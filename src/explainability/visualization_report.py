"""
Visualization report: before/after simulations, scalp topomaps, comparative PSD.

Output: models/evaluation/visualization_report/ with plots and summary.txt.
Run: eeg_twin\\Scripts\\python.exe -u -c "from src.explainability.visualization_report import run; run()"
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Feature layout (match model_wrapper)
PSD_SIZE = 380
BAND_SIZE = 95
N_CHANNELS = 19
N_BANDS = 5
PSD_DIM = 20

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
# Simple 2D scalp positions (approximate 10-20)
CHANNEL_POS = {
    "Fp1": (-0.03, 0.12), "Fp2": (0.03, 0.12),
    "F7": (-0.08, 0.06), "F3": (-0.05, 0.06), "Fz": (0, 0.06), "F4": (0.05, 0.06), "F8": (0.08, 0.06),
    "T3": (-0.10, 0), "C3": (-0.06, 0), "Cz": (0, 0), "C4": (0.06, 0), "T4": (0.10, 0),
    "T5": (-0.08, -0.06), "P3": (-0.05, -0.06), "Pz": (0, -0.06), "P4": (0.05, -0.06), "T6": (0.08, -0.06),
    "O1": (-0.03, -0.12), "O2": (0.03, -0.12),
}


def unflatten_sim(flat):
    """flat (..., 2185) -> psd (..., 19, 20), band_powers (..., 19, 5)."""
    shape = flat.shape[:-1]
    psd = flat[..., :PSD_SIZE].reshape(*shape, N_CHANNELS, PSD_DIM)
    band = flat[..., PSD_SIZE:PSD_SIZE + BAND_SIZE].reshape(*shape, N_CHANNELS, N_BANDS)
    return psd, band


def run():
    from src.models.config import EVALUATION_DIR, SIMULATIONS_DIR
    report_dir = EVALUATION_DIR / "visualization_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    base = np.load(SIMULATIONS_DIR / "simulated_baseline.npy")
    done = np.load(SIMULATIONS_DIR / "simulated_donepezil.npy")
    mem = np.load(SIMULATIONS_DIR / "simulated_memantine.npy")
    with open(SIMULATIONS_DIR / "simulation_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    subject_info = meta["subject_info"]

    # Select 2-3 AD, 2-3 HC for detailed plots
    ad_indices = [i for i, s in enumerate(subject_info) if s.get("group") == "AD"][:3]
    hc_indices = [i for i, s in enumerate(subject_info) if s.get("group") == "HC"][:3]
    plot_indices = ad_indices + hc_indices

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Before-after: mean feature block per condition (one panel per subject)
    for idx in plot_indices:
        info = subject_info[idx]
        b = base[idx].mean(axis=0)
        d = done[idx].mean(axis=0)
        m = mem[idx].mean(axis=0)
        psd_b, band_b = unflatten_sim(b[np.newaxis, :])
        psd_d, band_d = unflatten_sim(d[np.newaxis, :])
        psd_m, band_m = unflatten_sim(m[np.newaxis, :])
        block_means_b = [psd_b.mean(), band_b.mean(), b[PSD_SIZE + BAND_SIZE:].mean()]
        block_means_d = [psd_d.mean(), band_d.mean(), d[PSD_SIZE + BAND_SIZE:].mean()]
        block_means_m = [psd_m.mean(), band_m.mean(), m[PSD_SIZE + BAND_SIZE:].mean()]
        fig, ax = plt.subplots(figsize=(6, 3))
        x = np.arange(3)
        w = 0.25
        ax.bar(x - w, block_means_b, w, label="Baseline")
        ax.bar(x, block_means_d, w, label="Donepezil")
        ax.bar(x + w, block_means_m, w, label="Memantine")
        ax.set_xticks(x)
        ax.set_xticklabels(["PSD", "Band powers", "Connectivity"])
        ax.set_ylabel("Mean value")
        ax.legend()
        ax.set_title(f"Subject {info.get('subject_idx', idx)} ({info.get('group', '')})")
        plt.tight_layout()
        plt.savefig(report_dir / f"before_after_subj{idx}.png", dpi=150)
        plt.close()

    # 2) Scalp topomap: alpha band (index 2) for one subject, three conditions
    def plot_topomap(values_19, title, path):
        # values_19: (19,)
        x = np.array([CHANNEL_POS[ch][0] for ch in CHANNEL_NAMES])
        y = np.array([CHANNEL_POS[ch][1] for ch in CHANNEL_NAMES])
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(x, y, c=values_19, s=120, cmap="viridis")
        for i, ch in enumerate(CHANNEL_NAMES):
            ax.annotate(ch, (x[i], y[i]), fontsize=7, ha="center")
        plt.colorbar(sc, ax=ax, label="Band power")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    for idx in plot_indices[:4]:  # first 4 subjects
        info = subject_info[idx]
        for cond_name, arr in [("baseline", base), ("donepezil", done), ("memantine", mem)]:
            sim = arr[idx].mean(axis=0)
            _, band = unflatten_sim(sim[np.newaxis, :])
            alpha_power = band[0, :, 2]  # (19,) alpha band
            plot_topomap(
                alpha_power,
                f"Alpha power - {cond_name} (subj {info.get('subject_idx', idx)})",
                report_dir / f"topomap_alpha_subj{idx}_{cond_name}.png",
            )

    # 3) Comparative PSD: mean over channels, baseline vs donepezil vs memantine (averaged over selected subjects)
    psd_base = []
    psd_done = []
    psd_mem = []
    for idx in plot_indices:
        for s in range(base.shape[1]):
            psd_b, _ = unflatten_sim(base[idx, s : s + 1, :])
            psd_d, _ = unflatten_sim(done[idx, s : s + 1, :])
            psd_m, _ = unflatten_sim(mem[idx, s : s + 1, :])
            psd_base.append(psd_b[0].mean(axis=0))
            psd_done.append(psd_d[0].mean(axis=0))
            psd_mem.append(psd_m[0].mean(axis=0))
    psd_base = np.array(psd_base).mean(axis=0)
    psd_done = np.array(psd_done).mean(axis=0)
    psd_mem = np.array(psd_mem).mean(axis=0)
    freqs = np.linspace(0.5, 40, PSD_DIM)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(freqs, psd_base, label="Baseline")
    ax.plot(freqs, psd_done, label="Donepezil")
    ax.plot(freqs, psd_mem, label="Memantine")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (mean over channels)")
    ax.legend()
    ax.set_title("Comparative PSD (selected subjects)")
    plt.tight_layout()
    plt.savefig(report_dir / "comparative_psd.png", dpi=150)
    plt.close()

    # Summary text
    summary = [
        "Phase 2 Visualization Report",
        "=" * 50,
        datetime.now().isoformat(),
        "",
        "Contents:",
        "- before_after_subj*.png: Mean feature blocks (PSD, band powers, connectivity) for baseline vs Donepezil vs Memantine per subject.",
        "- topomap_alpha_subj*_*.png: Alpha-band scalp layout (19 channels) for baseline, donepezil, memantine.",
        "- comparative_psd.png: PSD (mean over channels) for the three conditions.",
        "",
        "Channel layout: 10-20 subset (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz).",
    ]
    with open(report_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("Visualization report saved to", report_dir)
    return report_dir
