"""
Fig. 1 — Pharmacodynamically constrained EEG-drug CVAE architecture.

Layout mirrors the ChatGPT architecture draft (inputs → encoders → fusion →
CVAE → dual endpoints), with locked numbers from MODEL_CONFIG / fusion.py.

Outputs:
  paper/figures/fig1_architecture.png
  paper/figures/fig1_architecture.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "figures"

NAVY = "#1f3a5f"
SLATE = "#6b7280"
TEAL = "#0f766e"
WARM = "#9ca3af"
LIGHT = "#f8fafc"
EEG_FILL = "#eef2ff"
BAND_FILL = "#ecfdf5"
CONN_FILL = "#f5f3ff"
DRUG_FILL = "#ecfdf5"
TEAL_FILL = "#ccfbf1"
TEAL_SOFT = "#ecfdf5"
GRAY_FILL = "#f3f4f6"
GRAY_SOFT = "#e5e7eb"
CVAE_FILL = "#ffffff"
CONSTRAINT_FILL = "#f0fdfa"


def _rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.linewidth": 0,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )


def _box(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    edge=NAVY,
    face=LIGHT,
    lw=1.25,
    ls="-",
    fontsize=7.2,
    weight="medium",
    color=NAVY,
    ha="center",
    va="center",
    z=2,
):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.08",
        linewidth=lw,
        linestyle=ls,
        edgecolor=edge,
        facecolor=face,
        zorder=z,
    )
    ax.add_patch(p)
    if text:
        ax.text(
            x + w / 2 if ha == "center" else x + 0.08,
            y + h / 2,
            text,
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            linespacing=1.2,
            zorder=z + 1,
        )
    return p


def _arrow(ax, x1, y1, x2, y2, *, color=SLATE, lw=1.15, ms=11, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
    )


def _trapezoid(ax, x, y, w, h, *, taper="in", face="#dbeafe", edge=NAVY, lw=1.1):
    """MLP encoder/decoder glyph. taper='in' narrows to the right; 'out' widens."""
    if taper == "in":
        pts = [(x, y), (x, y + h), (x + w, y + 0.78 * h), (x + w, y + 0.22 * h)]
    else:
        pts = [(x, y + 0.22 * h), (x, y + 0.78 * h), (x + w, y + h), (x + w, y)]
    poly = Polygon(pts, closed=True, facecolor=face, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(poly)


def _latent_dots(ax, x, y, n=5, color=NAVY, r=0.055):
    for i in range(n):
        circ = plt.Circle((x, y - i * 0.16), r, color=color, zorder=3)
        ax.add_patch(circ)


def make_figure() -> Path:
    _rc()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Match GPT landscape aspect; double-column Elsevier width.
    fig_w = 19.0 / 2.54
    fig_h = 14.2 / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    W, H = 19.0, 14.2
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Title
    ax.text(
        W / 2,
        13.75,
        "Pharmacodynamically constrained EEG–drug CVAE digital twin (architecture)",
        ha="center",
        va="center",
        fontsize=10.5,
        color=NAVY,
        fontweight="bold",
    )

    # Train badge (top-left)
    _box(
        ax,
        0.25,
        12.55,
        2.55,
        0.85,
        "Train N=66\n(AD 37 / HC 29)",
        edge=NAVY,
        face=NAVY,
        color="white",
        fontsize=7.5,
        weight="bold",
    )

    # ---------- INPUT EEG (left) ----------
    eeg_x, eeg_y, eeg_w, eeg_h = 0.25, 8.35, 4.55, 3.95
    _box(ax, eeg_x, eeg_y, eeg_w, eeg_h, "", edge=NAVY, face=EEG_FILL, lw=1.4)
    ax.text(
        eeg_x + eeg_w / 2,
        eeg_y + eeg_h - 0.32,
        "Input EEG features (2185-D)",
        ha="center",
        va="center",
        fontsize=8.2,
        color=NAVY,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        eeg_x + eeg_w / 2,
        eeg_y + eeg_h - 0.62,
        "19 channels, 10–20",
        ha="center",
        va="center",
        fontsize=6.8,
        color=SLATE,
        zorder=4,
    )
    # Sub-feature boxes
    _box(
        ax,
        eeg_x + 0.18,
        eeg_y + 2.35,
        eeg_w - 0.36,
        0.85,
        "PSD  19×20 = 380\n20 bins over ~0.5–40 Hz",
        edge=SLATE,
        face="white",
        fontsize=6.6,
        color=NAVY,
    )
    _box(
        ax,
        eeg_x + 0.18,
        eeg_y + 1.30,
        eeg_w - 0.36,
        0.90,
        "Band powers  19×5 = 95\n(δ, θ, α, β, γ)",
        edge="#059669",
        face=BAND_FILL,
        fontsize=6.6,
        color="#065f46",
    )
    _box(
        ax,
        eeg_x + 0.18,
        eeg_y + 0.22,
        eeg_w - 0.36,
        0.95,
        "Coherence + PLV = 1710\n5 bands × 171 upper-tri × 2",
        edge="#7c3aed",
        face=CONN_FILL,
        fontsize=6.6,
        color="#5b21b6",
    )

    # EEG encoder
    enc_x, enc_y = 5.05, 9.85
    _box(
        ax,
        enc_x,
        enc_y,
        2.55,
        1.15,
        "EEG encoder (MLP)\nhidden [256, 512]",
        edge=NAVY,
        face="#dbeafe",
        fontsize=6.8,
        weight="semibold",
    )
    _trapezoid(ax, enc_x + 0.15, enc_y + 0.12, 0.55, 0.35, taper="in", face="#93c5fd")
    _arrow(ax, eeg_x + eeg_w + 0.02, 10.4, enc_x - 0.04, enc_y + 0.55)

    # EEG latent
    _box(
        ax,
        7.85,
        9.95,
        1.55,
        0.95,
        "EEG latent\n128-D",
        edge=NAVY,
        face=LIGHT,
        fontsize=7.0,
        weight="semibold",
    )
    _latent_dots(ax, 9.15, 10.55, n=4, color=NAVY, r=0.045)
    _arrow(ax, enc_x + 2.55 + 0.02, enc_y + 0.55, 7.85 - 0.04, 10.4)

    # ---------- DRUG PATH (center-top) ----------
    _box(
        ax,
        5.05,
        12.35,
        4.35,
        0.95,
        "Drug options:  Baseline (zeros)  ·  Donepezil  ·  Memantine",
        edge=TEAL,
        face=DRUG_FILL,
        fontsize=6.6,
        color=TEAL,
        weight="semibold",
    )
    _box(
        ax,
        9.60,
        12.40,
        2.35,
        0.85,
        "ChemBERTa\nembedding 384-D",
        edge=TEAL,
        face=TEAL_SOFT,
        fontsize=6.8,
        color=TEAL,
        weight="semibold",
    )
    _arrow(ax, 5.05 + 4.35 + 0.02, 12.82, 9.60 - 0.04, 12.82, color=TEAL)

    _box(
        ax,
        12.15,
        12.30,
        2.70,
        1.05,
        "Drug encoder (MLP)\nhidden [256, 128]",
        edge=TEAL,
        face="#d1fae5",
        fontsize=6.8,
        color=TEAL,
        weight="semibold",
    )
    _trapezoid(ax, 12.30, 12.40, 0.50, 0.32, taper="in", face="#6ee7b7", edge=TEAL)
    _arrow(ax, 9.60 + 2.35 + 0.02, 12.82, 12.15 - 0.04, 12.82, color=TEAL)

    _box(
        ax,
        15.10,
        12.35,
        1.70,
        0.95,
        "Drug latent\n64-D",
        edge=TEAL,
        face=LIGHT,
        fontsize=7.0,
        color=TEAL,
        weight="bold",
    )
    _latent_dots(ax, 16.50, 12.95, n=4, color=TEAL, r=0.045)
    _arrow(ax, 12.15 + 2.70 + 0.02, 12.82, 15.10 - 0.04, 12.82, color=TEAL)

    # Disease label
    _box(
        ax,
        15.10,
        11.15,
        2.55,
        0.85,
        "Disease label scalar\nAD=1 / HC=0",
        edge=SLATE,
        face=GRAY_FILL,
        fontsize=6.8,
        color="#374151",
        weight="semibold",
    )

    # ---------- FUSION ----------
    fus_x, fus_y, fus_w, fus_h = 7.55, 7.95, 6.35, 1.55
    _box(
        ax,
        fus_x,
        fus_y,
        fus_w,
        fus_h,
        "",
        edge=NAVY,
        face="#fff7ed",
        lw=1.5,
    )
    ax.text(
        fus_x + fus_w / 2,
        fus_y + fus_h - 0.35,
        "Fusion (concat, no attention) → 256-D",
        ha="center",
        va="center",
        fontsize=8.0,
        color=NAVY,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        fus_x + fus_w / 2,
        fus_y + 0.55,
        "concat 128 EEG + 64 drug + 1 disease (=193)  →  MLP  →  256-D",
        ha="center",
        va="center",
        fontsize=6.5,
        color=SLATE,
        zorder=4,
    )
    # tiny concat glyph
    for i, c in enumerate(["#3b82f6", "#10b981", "#f59e0b"]):
        ax.add_patch(plt.Circle((fus_x + 2.2 + i * 0.22, fus_y + 0.28), 0.07, color=c, zorder=4))
    ax.text(fus_x + 3.2, fus_y + 0.28, "…", color=SLATE, fontsize=8, va="center", zorder=4)

    # Arrows into fusion
    _arrow(ax, 7.85 + 0.75, 9.95 - 0.02, fus_x + 1.2, fus_y + fus_h + 0.02)  # EEG latent down
    _arrow(ax, 15.10 + 0.85, 12.35 - 0.02, fus_x + fus_w - 1.0, fus_y + fus_h + 0.02, color=TEAL, rad=-0.15)
    _arrow(ax, 15.10 + 0.4, 11.15 - 0.02, fus_x + fus_w - 0.5, fus_y + fus_h + 0.02, color=SLATE, rad=-0.1)

    # ---------- CVAE ----------
    cv_x, cv_y, cv_w, cv_h = 5.05, 3.55, 10.85, 4.10
    _box(ax, cv_x, cv_y, cv_w, cv_h, "", edge=NAVY, face=CVAE_FILL, lw=1.6)
    ax.text(
        cv_x + 0.25,
        cv_y + cv_h - 0.28,
        "CVAE (hidden [512, 256])",
        ha="left",
        va="center",
        fontsize=8.5,
        color=NAVY,
        fontweight="bold",
        zorder=4,
    )

    # CVAE encoder
    _box(
        ax,
        cv_x + 0.25,
        cv_y + 2.35,
        2.35,
        1.05,
        "Encoder (MLP)",
        edge=NAVY,
        face="#dbeafe",
        fontsize=7.2,
        weight="semibold",
    )
    _trapezoid(ax, cv_x + 0.40, cv_y + 2.48, 0.55, 0.35, taper="in")

    # Gaussian latent
    _box(
        ax,
        cv_x + 3.00,
        cv_y + 2.25,
        2.55,
        1.25,
        "Gaussian latent\n128-D   μ, logσ²",
        edge=NAVY,
        face="#eff6ff",
        fontsize=7.0,
        weight="semibold",
    )
    _latent_dots(ax, cv_x + 5.15, cv_y + 3.20, n=5, color="#2563eb", r=0.05)
    _arrow(ax, cv_x + 0.25 + 2.35 + 0.02, cv_y + 2.85, cv_x + 3.00 - 0.04, cv_y + 2.85)

    # PD constraint (dashed)
    _box(
        ax,
        cv_x + 2.85,
        cv_y + 0.85,
        3.05,
        1.20,
        "Pharmacodynamic constraint\n"
        "Donepezil: δ− θ− α+ β+ conn+\n"
        "Memantine: δ− θ− α+ β+ conn+",
        edge=TEAL,
        face=CONSTRAINT_FILL,
        lw=1.35,
        ls="--",
        fontsize=6.2,
        color=TEAL,
        weight="medium",
    )

    # Decoder
    _box(
        ax,
        cv_x + 5.95,
        cv_y + 2.35,
        2.35,
        1.05,
        "Decoder (MLP)",
        edge=NAVY,
        face="#dbeafe",
        fontsize=7.2,
        weight="semibold",
    )
    _trapezoid(ax, cv_x + 7.55, cv_y + 2.48, 0.55, 0.35, taper="out")
    _arrow(ax, cv_x + 3.00 + 2.55 + 0.02, cv_y + 2.85, cv_x + 5.95 - 0.04, cv_y + 2.85)

    # Reconstructed
    _box(
        ax,
        cv_x + 8.55,
        cv_y + 2.25,
        2.05,
        1.25,
        "Reconstructed\nEEG features\n2185-D",
        edge=NAVY,
        face=LIGHT,
        fontsize=6.8,
        weight="semibold",
    )
    _arrow(ax, cv_x + 5.95 + 2.35 + 0.02, cv_y + 2.85, cv_x + 8.55 - 0.04, cv_y + 2.85)

    # Loss badge
    ax.text(
        cv_x + cv_w / 2,
        cv_y + 0.45,
        r"$L = L_{\mathrm{MSE}} + \beta L_{\mathrm{KL}} + \alpha_c(L_{\mathrm{band}}+L_{\mathrm{conn}})$"
        "     "
        r"$\beta{=}0.01$;  $\alpha_c$ warmup 5 epochs then ramp; constraint weight 0.1",
        ha="center",
        va="center",
        fontsize=6.4,
        color=NAVY,
        zorder=4,
    )

    # Arrow fusion → CVAE
    _arrow(ax, fus_x + fus_w / 2, fus_y - 0.02, cv_x + cv_w / 2, cv_y + cv_h + 0.02, lw=1.35)

    # ---------- PATH (a) TEAL — left bottom ----------
    ax.text(
        0.35,
        3.15,
        "Path (a)  MECHANISM / DIRECTION",
        ha="left",
        va="center",
        fontsize=8.0,
        color=TEAL,
        fontweight="bold",
    )
    _box(
        ax,
        0.25,
        2.15,
        3.55,
        0.80,
        "Latent (drug-conditioned)\nfrom CVAE",
        edge=TEAL,
        face=TEAL_SOFT,
        fontsize=6.6,
        color=TEAL,
        weight="semibold",
    )
    _box(
        ax,
        4.05,
        2.15,
        2.85,
        0.80,
        "CVAE Decoder\n(shared)",
        edge=TEAL,
        face=TEAL_SOFT,
        fontsize=6.6,
        color=TEAL,
        weight="semibold",
    )
    _box(
        ax,
        7.15,
        2.15,
        3.35,
        0.80,
        "Simulated post-drug\nEEG (2185-D)",
        edge=TEAL,
        face=TEAL_SOFT,
        fontsize=6.6,
        color=TEAL,
        weight="semibold",
    )
    _box(
        ax,
        10.75,
        1.95,
        3.85,
        1.15,
        "Drug-response DIRECTION\nband/connectivity sign agreement\nvs training signature",
        edge=TEAL,
        face=TEAL_FILL,
        fontsize=6.8,
        color=TEAL,
        weight="bold",
        lw=1.6,
    )
    _arrow(ax, 0.25 + 3.55 + 0.02, 2.55, 4.05 - 0.04, 2.55, color=TEAL)
    _arrow(ax, 4.05 + 2.85 + 0.02, 2.55, 7.15 - 0.04, 2.55, color=TEAL)
    _arrow(ax, 7.15 + 3.35 + 0.02, 2.55, 10.75 - 0.04, 2.55, color=TEAL)
    # from CVAE latent down to path a
    _arrow(ax, cv_x + 4.2, cv_y - 0.02, 2.0, 2.15 + 0.80 + 0.02, color=TEAL, rad=0.05)

    # ---------- PATH (b) GRAY — bottom ----------
    ax.text(
        0.35,
        1.55,
        "Path (b)  DIAGNOSIS / MAGNITUDE",
        ha="left",
        va="center",
        fontsize=8.0,
        color="#6b7280",
        fontweight="bold",
    )
    _box(
        ax,
        0.25,
        0.35,
        2.85,
        0.95,
        "",
        edge=WARM,
        face=GRAY_FILL,
    )
    ax.text(
        0.25 + 1.425,
        0.35 + 0.475,
        "Baseline mean\n" + r"$\mu_{\mathrm{base}}$ (drug=0)",
        ha="center",
        va="center",
        fontsize=6.4,
        color="#4b5563",
        fontweight="semibold",
        zorder=4,
    )
    _box(
        ax,
        3.30,
        0.35,
        2.70,
        0.95,
        "",
        edge=WARM,
        face=GRAY_FILL,
    )
    ax.text(
        3.30 + 1.35,
        0.35 + 0.475,
        "Donepezil mean\n" + r"$\mu_{\mathrm{don}}$",
        ha="center",
        va="center",
        fontsize=6.4,
        color="#4b5563",
        fontweight="semibold",
        zorder=4,
    )
    _box(
        ax,
        6.20,
        0.35,
        2.70,
        0.95,
        "",
        edge=WARM,
        face=GRAY_FILL,
    )
    ax.text(
        6.20 + 1.35,
        0.35 + 0.475,
        "Memantine mean\n" + r"$\mu_{\mathrm{mem}}$",
        ha="center",
        va="center",
        fontsize=6.4,
        color="#4b5563",
        fontweight="semibold",
        zorder=4,
    )
    _box(
        ax,
        9.10,
        0.25,
        5.05,
        1.15,
        "",
        edge=WARM,
        face=GRAY_SOFT,
    )
    ax.text(
        9.10 + 2.525,
        0.25 + 0.575,
        r"$\bar{s}=\frac{1}{2}(\|\mu_{\mathrm{don}}-\mu_{\mathrm{base}}\|+\|\mu_{\mathrm{mem}}-\mu_{\mathrm{base}}\|)$",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#374151",
        zorder=4,
    )
    _box(
        ax,
        14.35,
        0.30,
        3.40,
        1.05,
        "Drug-response\nMAGNITUDE",
        edge=WARM,
        face=GRAY_SOFT,
        fontsize=7.5,
        color="#6b7280",
        weight="bold",
        lw=1.6,
    )
    # GPT-style row: three means → formula → magnitude endpoint
    _arrow(ax, 6.20 + 2.70 + 0.02, 0.82, 9.10 - 0.04, 0.82, color=WARM)
    _arrow(ax, 9.10 + 5.05 + 0.02, 0.82, 14.35 - 0.04, 0.82, color=WARM)
    # light guides from base/don into the shared formula path
    _arrow(ax, 0.25 + 2.85 + 0.02, 0.82, 6.20 - 0.04, 0.82, color=WARM, lw=0.9, ms=8)
    _arrow(ax, 3.30 + 2.70 + 0.02, 0.82, 6.20 - 0.04, 0.82, color=WARM, lw=0.9, ms=8)
    # from CVAE to path b
    _arrow(ax, cv_x + 5.5, cv_y - 0.02, 4.5, 0.35 + 0.95 + 0.08, color=WARM, rad=-0.08)

    png = OUT_DIR / "fig1_architecture.png"
    svg = OUT_DIR / "fig1_architecture.svg"
    fig.savefig(png, dpi=300, facecolor="white", edgecolor="none", pad_inches=0.15)
    fig.savefig(svg, format="svg", facecolor="white", edgecolor="none", pad_inches=0.15)
    plt.close(fig)
    return png


if __name__ == "__main__":
    out = make_figure()
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.svg')}")
