"""
Fig. 3 — External diagnostic discrimination on twin drug-response magnitude.

Forest plot of Cohen's d for the primary contrast in each escalating cohort.

Source (locked): models/validation/complete_validation_report_v3.json
  TUH:    layer5.abnormal_vs_normal
  OSF:    layer5b_ad_labeled_external.discrimination
  P-ADIC: layer5d_padic_external.discrimination
  CAUEEG: layer5e_caueeg_external.discrimination.primary_dementia_vs_normal

Primary mirrors:
  data/ad_labeled_external/validation/layer5b_results.json
  models/validation/layer5d_padic_results.json
  models/validation/layer5e_caueeg_results.json

95% CI for Cohen's d is not stored in JSON. It is computed from the
reported d and group sizes with the large-sample variance for the
standardized mean difference (Hedges & Olkin form):

  SE(d) = sqrt( (n1+n2)/(n1*n2) + d^2 / (2*(n1+n2)) )
  95% CI = d ± 1.96 * SE(d)

This is an approximation attached to the reported Cohen's d; the primary
hypothesis test remains Mann-Whitney U (p stored in JSON).

Outputs:
  paper/figures/fig3_diagnostic_null.png
  paper/figures/fig3_diagnostic_null.svg
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "figures"
V3_JSON = PROJECT_ROOT / "models" / "validation" / "complete_validation_report_v3.json"

WARM = "#9ca3af"
NAVY = "#1f3a5f"
SLATE = "#6b7280"
EDGE = "#6b7280"


def cohens_d_ci(d: float, n1: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    """Approximate 95% CI for Cohen's d from d and group sizes."""
    se = np.sqrt((n1 + n2) / (n1 * n2) + (d ** 2) / (2.0 * (n1 + n2)))
    return float(d - z * se), float(d + z * se), float(se)


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    # three significant figures
    return f"p = {p:.3g}"


def _load_rows() -> list[dict]:
    with open(V3_JSON, encoding="utf-8") as f:
        rep = json.load(f)

    tuh = rep["layer5"]["abnormal_vs_normal"]
    osf = rep["layer5b_ad_labeled_external"]["discrimination"]
    padic = rep["layer5d_padic_external"]["discrimination"]
    caueeg = rep["layer5e_caueeg_external"]["discrimination"]["primary_dementia_vs_normal"]

    rows = [
        {
            "name": "TUH",
            "contrast": "Abn vs Nor",
            "n1": int(tuh["n_abnormal"]),
            "n2": int(tuh["n_normal"]),
            "n_label": f"n = {tuh['n_abnormal']}+{tuh['n_normal']}",
            "d": float(tuh["cohens_d"]),
            "p": float(tuh["mannwhitney_p"]),
            "caveat": False,
            "json_key": "layer5.abnormal_vs_normal",
        },
        {
            "name": "OSF",
            "contrast": "AD vs HC",
            "n1": int(osf["n_ad"]),
            "n2": int(osf["n_hc"]),
            "n_label": f"n = {osf['n_ad']}+{osf['n_hc']}",
            "d": float(osf["cohens_d"]),
            "p": float(osf["mannwhitney_p"]),
            "caveat": True,  # PASS_WITH_CAVEATS / HC n=12
            "json_key": "layer5b_ad_labeled_external.discrimination",
        },
        {
            "name": "P-ADIC",
            "contrast": "AD vs HC",
            "n1": int(padic["n_ad"]),
            "n2": int(padic["n_hc"]),
            "n_label": f"n = {padic['n_ad']}+{padic['n_hc']}",
            "d": float(padic["cohens_d"]),
            "p": float(padic["mannwhitney_p"]),
            "caveat": False,
            "json_key": "layer5d_padic_external.discrimination",
        },
        {
            "name": "CAUEEG",
            "contrast": "Dem vs Nor",
            "n1": int(caueeg["n_dementia"]),
            "n2": int(caueeg["n_normal"]),
            "n_label": f"n = {caueeg['n_dementia']}+{caueeg['n_normal']}",
            "d": float(caueeg["cohens_d"]),
            "p": float(caueeg["mannwhitney_p"]),
            "caveat": False,
            "json_key": "layer5e_caueeg_external.discrimination.primary_dementia_vs_normal",
        },
    ]

    for row in rows:
        lo, hi, se = cohens_d_ci(row["d"], row["n1"], row["n2"])
        row["ci_lo"] = lo
        row["ci_hi"] = hi
        row["se"] = se
    return rows


def make_figure() -> Path:
    rows = _load_rows()
    # Top → bottom: TUH, OSF, P-ADIC, CAUEEG
    y = np.arange(len(rows))[::-1]  # reverse so TUH is at top when plotted

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

    fig_w = 9.0 / 2.54
    fig_h = 8.0 / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.axvline(0.0, color=SLATE, linewidth=1.0, linestyle="-", zorder=1)

    for yi, row in zip(y, rows):
        ax.plot(
            [row["ci_lo"], row["ci_hi"]],
            [yi, yi],
            color=WARM,
            linewidth=1.8,
            solid_capstyle="round",
            zorder=2,
        )
        # whisker caps
        cap = 0.12
        ax.plot([row["ci_lo"], row["ci_lo"]], [yi - cap, yi + cap], color=WARM, linewidth=1.4, zorder=2)
        ax.plot([row["ci_hi"], row["ci_hi"]], [yi - cap, yi + cap], color=WARM, linewidth=1.4, zorder=2)

        if row["caveat"]:
            ax.plot(
                row["d"],
                yi,
                marker="o",
                markersize=7.5,
                markerfacecolor="white",
                markeredgecolor=EDGE,
                markeredgewidth=1.6,
                zorder=3,
                label="OSF PASS_WITH_CAVEATS\n(small HC n=12; interpret with caution)",
            )
        else:
            ax.plot(
                row["d"],
                yi,
                marker="o",
                markersize=7.5,
                markerfacecolor=WARM,
                markeredgecolor=EDGE,
                markeredgewidth=1.0,
                zorder=3,
            )

        # Annotation to the right of the CI
        ann = (
            f"d = {row['d']:.2f}, {_fmt_p(row['p'])}, "
            f"{row['n_label']}"
        )
        ax.text(
            max(row["ci_hi"], 0.05) + 0.08,
            yi,
            ann,
            va="center",
            ha="left",
            fontsize=6.3,
            color=NAVY,
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{row['name']}\n({row['contrast']})" for row in rows],
        fontsize=7.5,
        color=NAVY,
    )
    ax.set_xlabel("Cohen's $d$ (drug-response magnitude)", fontsize=8, color=NAVY)

    # X-limits: cover CIs with room for annotations
    xmin = min(r["ci_lo"] for r in rows) - 0.15
    xmax = max(r["ci_hi"] for r in rows) + 1.55
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)
    ax.tick_params(colors=SLATE)
    ax.xaxis.label.set_color(NAVY)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.6, color="#d1d5db", zorder=0)

    # Legend only for caveat marker
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower right",
            fontsize=5.8,
            frameon=True,
            fancybox=False,
            edgecolor="#e5e7eb",
            framealpha=0.95,
        )

    fig.tight_layout(pad=0.35)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "fig3_diagnostic_null.png"
    svg = OUT_DIR / "fig3_diagnostic_null.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    for row in rows:
        print(
            f"{row['name']}: d={row['d']:.6f}  CI=[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]  "
            f"p={row['p']:.6g}  n1={row['n1']} n2={row['n2']}  caveat={row['caveat']}  "
            f"key={row['json_key']}"
        )
    return png


if __name__ == "__main__":
    out = make_figure()
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.svg')}")
