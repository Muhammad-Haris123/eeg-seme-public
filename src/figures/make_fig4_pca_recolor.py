"""
Fig. 4 — PCA overlap: training cohort vs CAUEEG features (recolored).

Regenerates the Layer 5e PCA scatter from the same data/transform as
`src/tuh/run_layer5e_caueeg.py::pca_overlap`, with paper palette and
Elsevier styling (no on-figure title).

Why not reuse models/validation/figures/layer5e_caueeg_pca.png:
  - default seaborn/matplotlib tab colors
  - missing PC1/PC2 axis labels
  - on-figure title (caption belongs in manuscript)
  - 95% training radius not named in the legend

Contextual color choice: CAUEEG uses teal accents because this panel
supports the Part B interpretation that magnitude nulls occur despite
good PCA overlap (overlap_fraction ≈ 0.611), not because of a failed
domain check. Training points use navy.

Outputs:
  paper/figures/fig4_pca.png
  paper/figures/fig4_pca.svg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from api.utils.feature_processor import flatten_features  # noqa: E402

OUT_DIR = PROJECT_ROOT / "paper" / "figures"
CAUEEG_DIR = PROJECT_ROOT / "data" / "caueeg_external"

NAVY = "#1f3a5f"
NAVY_LIGHT = "#5b7a9d"
TEAL = "#0f766e"
TEAL_MID = "#14b8a6"
TEAL_SOFT = "#5eead4"
SLATE = "#6b7280"
WARM = "#9ca3af"


def _load_train() -> tuple[np.ndarray, np.ndarray]:
    train, train_y = [], []
    for g, lab in (("AD", 1), ("HC", 0)):
        psd = np.load(PROJECT_ROOT / "data" / "eeg_features" / f"{g}_psd.npy")
        bp = np.load(PROJECT_ROOT / "data" / "eeg_features" / f"{g}_band_powers.npy")
        coh = np.load(PROJECT_ROOT / "data" / "eeg_features" / f"{g}_coherence.npy")
        plv = np.load(PROJECT_ROOT / "data" / "eeg_features" / f"{g}_plv.npy")
        for i in range(psd.shape[0]):
            train.append(flatten_features(psd[i], bp[i], coh[i], plv[i]))
            train_y.append(lab)
    return np.stack(train), np.asarray(train_y)


def _load_caueeg() -> tuple[np.ndarray, list[dict]]:
    report = json.loads((CAUEEG_DIR / "processing_report.json").read_text(encoding="utf-8"))
    flats, metas = [], []
    for rec in report["results"]:
        if not rec.get("ok"):
            continue
        flat = np.load(rec["feature_path"]).astype(np.float32)
        class_name = rec.get("class_name") or rec.get("group") or "Unknown"
        metas.append({"label": class_name.lower(), "class_name": class_name})
        flats.append(flat)
    return np.stack(flats, axis=0), metas


def make_figure() -> Path:
    train, train_y = _load_train()
    flats, metas = _load_caueeg()

    pca = PCA(2, random_state=42)
    t2 = pca.fit_transform(train)
    x2 = pca.transform(flats)
    center = t2.mean(0)
    r95 = float(np.percentile(np.linalg.norm(t2 - center, axis=1), 95))
    overlap = float(np.mean(np.linalg.norm(x2 - center, axis=1) <= r95))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    # Single-column width (~9 cm); taller for scatter readability
    fig_w = 9.0 / 2.54
    fig_h = 8.2 / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Training (navy family)
    ax.scatter(
        t2[train_y == 1, 0],
        t2[train_y == 1, 1],
        c=NAVY,
        s=28,
        alpha=0.75,
        marker="o",
        edgecolors="none",
        label="Train AD",
        zorder=3,
    )
    ax.scatter(
        t2[train_y == 0, 0],
        t2[train_y == 0, 1],
        c=NAVY_LIGHT,
        s=28,
        alpha=0.75,
        marker="o",
        edgecolors="none",
        label="Train HC",
        zorder=3,
    )

    # CAUEEG (teal family — good-overlap finding)
    caueeg_style = {
        "normal": (TEAL_SOFT, "Normal"),
        "mci": (TEAL_MID, "MCI"),
        "dementia": (TEAL, "Dementia"),
    }
    for lab, (col, pretty) in caueeg_style.items():
        mask = np.array([m["label"] == lab for m in metas])
        if not mask.any():
            continue
        ax.scatter(
            x2[mask, 0],
            x2[mask, 1],
            c=col,
            s=14,
            marker="^",
            alpha=0.55,
            edgecolors=TEAL,
            linewidths=0.25,
            label=f"CAUEEG {pretty}",
            zorder=2,
        )

    circ = Circle(
        center,
        r95,
        fill=False,
        ls="--",
        linewidth=1.2,
        edgecolor=WARM,
        label="Train 95% radius",
        zorder=1,
    )
    ax.add_patch(circ)

    ax.set_xlabel("PC1", fontsize=8, color=NAVY)
    ax.set_ylabel("PC2", fontsize=8, color=NAVY)
    ax.tick_params(colors=SLATE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)
    ax.set_aspect("equal", adjustable="datalim")

    # Legend: data classes + explicit 95% radius entry
    handles, labels = ax.get_legend_handles_labels()
    # Ensure radius appears even if patch order is odd
    ax.legend(
        handles,
        labels,
        frameon=True,
        fancybox=False,
        edgecolor="#e5e7eb",
        fontsize=6.0,
        loc="best",
        framealpha=0.95,
    )

    fig.tight_layout(pad=0.35)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "fig4_pca.png"
    svg = OUT_DIR / "fig4_pca.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"overlap_fraction_within_train_r95 = {overlap:.6f}")
    print(f"r95 = {r95:.6f}")
    print(f"n_train = {train.shape[0]}, n_caueeg = {flats.shape[0]}")
    return png


if __name__ == "__main__":
    out = make_figure()
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.svg')}")
