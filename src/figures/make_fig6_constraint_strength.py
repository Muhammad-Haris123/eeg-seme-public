"""
Fig. 6 — Constraint-strength (alpha_c) sweep: direction vs latent AUC.

Source (locked):
  models/validation/constraint_strength_sweep.json

Outputs:
  paper/figures/fig6_constraint_strength.{png,svg}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "figures"
SWEEP_JSON = PROJECT_ROOT / "models" / "validation" / "constraint_strength_sweep.json"

TEAL = "#0f766e"
NAVY = "#1f3a5f"
AMBER = "#b45309"
SLATE = "#6b7280"


def _parse_agree(s: str) -> float:
    a, b = s.split("/")
    return 100.0 * float(a) / float(b)


def _load() -> list[dict]:
    d = json.loads(SWEEP_JSON.read_text(encoding="utf-8"))
    rows = []
    for pt in d["points"]:
        ext = pt["external"]
        tuh = ext["tuh_direction"]
        pad = ext["padic_direction"]
        rows.append(
            {
                "alpha": float(pt["constraint_weight"]),
                "tuh_agree_pct": _parse_agree(tuh["direction_agreement_total"]),
                "padic_agree_pct": _parse_agree(pad["direction_agreement_total"]),
                "tuh_r": float(tuh["effect_magnitude_correlation"]),
                "padic_r": float(pad["effect_magnitude_correlation"]),
                "auc": float(ext["caueeg_latent_probe"]["roc_auc_mean"]),
                "tuh_agree": tuh["direction_agreement_total"],
                "padic_agree": pad["direction_agreement_total"],
            }
        )
    return rows


def make_figure() -> Path:
    rows = _load()
    x = np.array([r["alpha"] for r in rows], dtype=float)
    # mean of TUH and P-ADIC agreement % for primary curve
    agree = np.array(
        [0.5 * (r["tuh_agree_pct"] + r["padic_agree_pct"]) for r in rows], dtype=float
    )
    r_mean = np.array([0.5 * (r["tuh_r"] + r["padic_r"]) for r in rows], dtype=float)
    auc = np.array([r["auc"] for r in rows], dtype=float)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax1 = plt.subplots(figsize=(5.6, 3.6), dpi=300)
    ax2 = ax1.twinx()

    (l1,) = ax1.plot(
        x,
        agree,
        "o-",
        color=TEAL,
        lw=1.6,
        ms=6,
        label="Direction agreement (%; mean TUH, P-ADIC)",
        zorder=3,
    )
    (l2,) = ax1.plot(
        x,
        100.0 * np.clip(r_mean, 0, 1),
        "s--",
        color=NAVY,
        lw=1.2,
        ms=5,
        label="Effect-vector r (scaled ×100; mean TUH, P-ADIC)",
        zorder=2,
    )
    (l3,) = ax2.plot(
        x,
        auc,
        "D-",
        color=AMBER,
        lw=1.4,
        ms=5,
        label="CAUEEG latent probe AUC",
        zorder=3,
    )

    ax1.axvline(0.1, color=SLATE, ls=":", lw=0.9, alpha=0.8)
    ax1.text(0.1, 102.5, r"paper $\alpha_c{=}0.1$", ha="center", va="bottom", color=SLATE, fontsize=7)

    ax1.set_xlabel(r"Constraint weight $\alpha_c$")
    ax1.set_ylabel("Direction agreement (%) / $100\\times r$")
    ax2.set_ylabel("CAUEEG latent ROC-AUC")
    ax1.set_ylim(0, 110)
    ax2.set_ylim(0.45, 0.65)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{v:g}" for v in x])
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, frameon=False, loc="lower right", fontsize=7)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "fig6_constraint_strength.png"
    svg = OUT_DIR / "fig6_constraint_strength.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {png}")
    print(f"[wrote] {svg}")
    return png


if __name__ == "__main__":
    make_figure()
