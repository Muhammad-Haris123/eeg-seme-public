"""Build CBM graphical abstract PNG (no placeholder text)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "graphical_abstract.png"


def _box(ax, xy, w, h, facecolor, edgecolor="#1f2937", lw=1.2):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        mutation_aspect=0.3,
    )
    ax.add_patch(patch)
    return patch


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.0, 3.4), dpi=220)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Panel 1: twin
    _box(ax, (0.25, 0.45), 3.4, 2.5, "#ecfdf5", "#047857")
    ax.text(1.95, 2.65, "Constrained EEG-drug CVAE", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#064e3b")
    ax.text(1.95, 2.15, "2185-D EEG + ChemBERTa\nDonepezil / Memantine",
            ha="center", va="center", fontsize=9, color="#134e4a")
    ax.text(1.95, 1.35, "Literature PD band + connectivity\nconstraints ($\\alpha_c{=}0.1$)",
            ha="center", va="center", fontsize=9, color="#134e4a")
    ax.text(1.95, 0.7, "Train $N{=}66$ (AD 37 / HC 29)",
            ha="center", va="center", fontsize=8.5, color="#065f46")

    # Panel 2: direction (locked fold-0 primary endpoint)
    _box(ax, (4.15, 0.45), 3.5, 2.5, "#eff6ff", "#1d4ed8")
    ax.text(5.9, 2.65, "Directional concordance", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1e3a8a")
    ax.text(5.9, 2.20, "Signs more stable than continuous $r$\n(not paired post-dose EEG)",
            ha="center", va="center", fontsize=8.5, color="#1e40af")
    ax.text(5.9, 1.50, "Locked seed-42 fold-0:\n$\\mathbf{10/10}$; $r{=}0.908/0.851/0.918$",
            ha="center", va="center", fontsize=9, color="#1e3a8a")
    ax.text(5.9, 0.75, "Fold + train-seed sensitive $r$\n(non-42 often ${\\approx}0.08$ to $0.43$)",
            ha="center", va="center", fontsize=8, color="#1d4ed8")

    # Panel 3: diagnosis + encoding
    _box(ax, (8.15, 0.45), 3.55, 2.5, "#fff7ed", "#c2410c")
    ax.text(9.925, 2.65, "Diagnosis fails externally", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#7c2d12")
    ax.text(9.925, 2.15, "Magnitude nulls (4 cohorts)\nEncoding: attenuated vs features",
            ha="center", va="center", fontsize=9, color="#9a3412")
    ax.text(9.925, 1.35, "Latent probe AUC $\\mathbf{0.579}$\n$\\theta/\\alpha$ mean-fold $\\mathbf{0.675}$",
            ha="center", va="center", fontsize=9, color="#7c2d12")
    ax.text(9.925, 0.7, "Best head OOF AUC $0.593$  |  CAUEEG $n{=}727$",
            ha="center", va="center", fontsize=8, color="#c2410c")

    for x0, x1 in ((3.65, 4.15), (7.65, 8.15)):
        ax.add_patch(
            FancyArrowPatch(
                (x0, 1.7),
                (x1, 1.7),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.6,
                color="#374151",
            )
        )

    ax.text(
        6.0,
        3.2,
        "External spine: directional/sign concordance more stable than continuous $r$; diagnostic magnitude no",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#111827",
        fontweight="bold",
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
