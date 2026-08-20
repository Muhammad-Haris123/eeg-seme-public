"""Supplementary training curves from stored constrained fold-0 logs.

Source: models/evaluation/constrained_evaluation_report.json -> constraint_curves.fold_0
Does not synthesize losses.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "models" / "evaluation" / "constrained_evaluation_report.json"
OUT = ROOT / "paper" / "figures" / "fig_s1_training_curves.png"


def main() -> Path:
    with SRC.open(encoding="utf-8") as f:
        rep = json.load(f)
    curves = rep["constraint_curves"]["fold_0"]
    epochs = np.array([c["epoch"] for c in curves], dtype=int)
    keys = ["total", "mse", "kl", "band", "conn"]
    train = {k: np.array([c["train"][k] for c in curves], dtype=float) for k in keys}
    val = {k: np.array([c["val"][k] for c in curves], dtype=float) for k in keys}
    alpha = np.array([c["train"]["alpha"] for c in curves], dtype=float)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(epochs, train["total"], color="#0f766e", lw=1.2, label="Train total")
    ax.plot(epochs, val["total"], color="#1f3a5f", lw=1.2, ls="--", label="Val total")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("A. Total loss (fold 0)")

    ax = axes[0, 1]
    ax.plot(epochs, train["mse"], color="#0f766e", lw=1.2, label="Train MSE")
    ax.plot(epochs, val["mse"], color="#1f3a5f", lw=1.2, ls="--", label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("B. Reconstruction MSE")

    ax = axes[1, 0]
    ax.plot(epochs, train["kl"], color="#0f766e", lw=1.2, label="Train KL")
    ax.plot(epochs, val["kl"], color="#1f3a5f", lw=1.2, ls="--", label="Val KL")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("C. KL term (unweighted)")

    ax = axes[1, 1]
    ax.plot(epochs, train["band"], color="#b45309", lw=1.1, label="Train L_band")
    ax.plot(epochs, train["conn"], color="#7c3aed", lw=1.1, label="Train L_conn")
    ax.plot(epochs, alpha, color="#6b7280", lw=1.1, ls=":", label=r"$\alpha_c$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Constraint / weight")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("D. Unweighted penalties and $\\alpha_c$")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".svg"))
    print("wrote", OUT)
    return OUT


if __name__ == "__main__":
    main()
