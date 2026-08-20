"""
Fig. 2 — Cross-dataset drug-response direction agreement (correlation r).

Source (locked):
  models/validation/complete_validation_report_v3.json
    layer5.cross_dataset
    layer5b_ad_labeled_external.cross_dataset
    layer5d_padic_external.cross_dataset

Primary mirrors:
  data/ad_labeled_external/validation/layer5b_results.json
  models/validation/layer5d_padic_results.json

No CI/SE for effect_magnitude_correlation is present in these JSON files;
error bars are omitted intentionally.

Outputs:
  paper/figures/fig2_direction_agreement.png
  paper/figures/fig2_direction_agreement.svg
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "figures"
V3_JSON = PROJECT_ROOT / "models" / "validation" / "complete_validation_report_v3.json"

TEAL = "#0f766e"
NAVY = "#1f3a5f"
SLATE = "#6b7280"


def _load_rows() -> list[dict]:
    with open(V3_JSON, encoding="utf-8") as f:
        rep = json.load(f)

    tuh = rep["layer5"]["cross_dataset"]
    osf = rep["layer5b_ad_labeled_external"]["cross_dataset"]
    padic = rep["layer5d_padic_external"]["cross_dataset"]

    return [
        {
            "label": "TUH",
            "r": float(tuh["effect_magnitude_correlation"]),
            "agreement": tuh["direction_agreement_total"],
            "n": int(tuh["n_tuh_used"]),
            "json_path": "layer5.cross_dataset",
        },
        {
            "label": "OSF",
            "r": float(osf["effect_magnitude_correlation"]),
            "agreement": osf["direction_agreement_total"],
            "n": int(osf["n_used"]),
            "json_path": "layer5b_ad_labeled_external.cross_dataset",
        },
        {
            "label": "P-ADIC",
            "r": float(padic["effect_magnitude_correlation"]),
            "agreement": padic["direction_agreement_total"],
            "n": int(padic["n_used"]),
            "json_path": "layer5d_padic_external.cross_dataset",
        },
    ]


def make_figure() -> Path:
    rows = _load_rows()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    # Single-column Elsevier width (~9 cm)
    fig_w = 9.0 / 2.54
    fig_h = 7.2 / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(len(rows))
    r_vals = [row["r"] for row in rows]
    bar_w = 0.55

    bars = ax.bar(
        x,
        r_vals,
        width=bar_w,
        color=TEAL,
        edgecolor=NAVY,
        linewidth=0.9,
        zorder=3,
    )

    ax.axhline(0.0, color=SLATE, linewidth=0.9, linestyle="-", zorder=2)

    # Annotate agreement fraction above each bar
    for i, (bar, row) in enumerate(zip(bars, rows)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.025,
            row["agreement"],
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=TEAL,
            zorder=4,
        )
        # r value just inside / on bar for readability
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(height * 0.5, 0.08),
            f"r = {row['r']:.3f}",
            ha="center",
            va="center",
            fontsize=6.5,
            color="white",
            fontweight="semibold",
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row['label']}\n(n = {row['n']})" for row in rows],
        fontsize=8,
        color=NAVY,
    )
    ax.set_ylabel(
        "Effect-vector correlation $r$\n(drug-response direction)",
        fontsize=8,
        color=NAVY,
    )
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(-0.55, len(rows) - 0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)
    ax.tick_params(colors=SLATE)
    ax.yaxis.label.set_color(NAVY)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#d1d5db", zorder=0)

    fig.tight_layout(pad=0.35)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "fig2_direction_agreement.png"
    svg = OUT_DIR / "fig2_direction_agreement.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    # Console audit trail (not written into figure)
    for row in rows:
        print(
            f"{row['label']}: agreement={row['agreement']}  "
            f"r={row['r']:.6f}  n={row['n']}  key={row['json_path']}"
        )
    print("CI/SE: not present in source JSON; error bars omitted.")
    return png


if __name__ == "__main__":
    out = make_figure()
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.svg')}")
