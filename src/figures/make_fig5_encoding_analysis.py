"""
Fig. 5 — CAUEEG encoding analysis (central Results figure).

Horizontal bars: feature reference vs frozen-twin latent readouts.
Visual intent (5-second read): latents carry some signal above the
permutation null, less than theta/alpha, and stronger heads do not close
the gap.

Sources:
  models/validation/layer5e_caueeg_latent_probe_results.json
  models/validation/layer5e_caueeg_classifier_head_results.json

Outputs:
  paper/figures/fig5_encoding_analysis.png
  paper/figures/fig5_encoding_analysis.svg
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "figures"
PROBE_JSON = PROJECT_ROOT / "models" / "validation" / "layer5e_caueeg_latent_probe_results.json"
HEAD_JSON = PROJECT_ROOT / "models" / "validation" / "layer5e_caueeg_classifier_head_results.json"

NAVY = "#1f3a5f"
TEAL = "#0f766e"
TEAL_MID = "#0d9488"
TEAL_SOFT = "#5eead4"
AMBER = "#b45309"
AMBER_FILL = "#f59e0b"
SLATE = "#6b7280"
NULL_BAND = "#e5e7eb"
CHANCE = "#9ca3af"


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _load_rows() -> tuple[list[dict], float, float]:
    probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    head = json.loads(HEAD_JSON.read_text(encoding="utf-8"))

    null = probe["permutation_test_latent"]["null_auc_percentiles"]
    p5 = float(null["p5"])
    p95 = float(null["p95"])

    # Locked primary AUCs: feature & original probe = CV means;
    # nested heads = OOF (as in best_head / comparison_table).
    rows = [
        {
            "label": r"$\theta/\alpha$ feature probe" + "\n(reference ceiling)",
            "auc": float(probe["feature_probe_theta_alpha_dementia_vs_normal"]["cv"]["roc_auc"]["mean"]),
            "kind": "feature",
            "p": None,
            "json_key": "feature_probe_theta_alpha_dementia_vs_normal.cv.roc_auc.mean",
        },
        {
            "label": r"Tuned L2 logistic regression on $\mu_{\mathrm{base}}$",
            "auc": float(head["best_head"]["oof_roc_auc"]),
            "kind": "latent",
            "p": float(head["permutation_best_head"]["permutation_p_value"]),
            "json_key": "best_head.oof_roc_auc / permutation_best_head",
        },
        {
            "label": r"MLP on $\mu_{\mathrm{base}}$",
            "auc": float(head["heads"]["mlp"]["oof"]["roc_auc"]),
            "kind": "latent",
            "p": None,
            "json_key": "heads.mlp.oof.roc_auc",
        },
        {
            "label": r"HistGradientBoosting on $\mu_{\mathrm{base}}$",
            "auc": float(head["heads"]["hist_gradient_boosting"]["oof"]["roc_auc"]),
            "kind": "latent",
            "p": None,
            "json_key": "heads.hist_gradient_boosting.oof.roc_auc",
        },
        {
            "label": r"$\mu$+logvar control (tuned LogReg)",
            "auc": float(head["mu_logvar_control"]["oof"]["roc_auc"]),
            "kind": "latent",
            "p": None,
            "json_key": "mu_logvar_control.oof.roc_auc",
        },
        {
            "label": r"Untuned logistic regression on $\mu_{\mathrm{base}}$" + "\n(original latent probe)",
            "auc": float(probe["latent_probe_dementia_vs_normal"]["cv"]["roc_auc"]["mean"]),
            "kind": "latent",
            "p": float(probe["permutation_test_latent"]["permutation_p_value"]),
            "json_key": "latent_probe_dementia_vs_normal.cv.roc_auc.mean / permutation_test_latent",
        },
    ]
    return rows, p5, p95


def make_figure() -> Path:
    rows, p5, p95 = _load_rows()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.9,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )

    # Double-column Elsevier width (~19 cm)
    fig_w = 19.0 / 2.54
    fig_h = 10.5 / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    n = len(rows)
    # Top → bottom as specified
    y = np.arange(n)[::-1]
    aucs = [r["auc"] for r in rows]
    bar_h = 0.62

    # Permutation-null 95% band (latent-probe null) behind everything
    ax.axvspan(
        p5,
        p95,
        color=NULL_BAND,
        alpha=1.0,
        zorder=0,
        label=f"Permutation null 5th–95th pct\n([{p5:.3f}, {p95:.3f}])",
    )
    ax.axvline(0.5, color=CHANCE, linestyle="--", linewidth=1.3, zorder=1, label="Chance (AUC = 0.5)")

    colors = []
    for r in rows:
        if r["kind"] == "feature":
            colors.append(AMBER_FILL)
        else:
            # Slightly emphasize best head; others teal/navy family
            colors.append(TEAL if "Tuned L2" in r["label"] else TEAL_MID)

    # Soften non-best latent bars
    for i, r in enumerate(rows):
        if r["kind"] == "latent" and "Tuned L2" not in r["label"] and "Untuned" not in r["label"]:
            colors[i] = "#5f9ea0"  # muted teal-gray for weaker heads
        if "Untuned" in r["label"]:
            colors[i] = NAVY

    bars = ax.barh(
        y,
        aucs,
        height=bar_h,
        color=colors,
        edgecolor=NAVY,
        linewidth=0.8,
        zorder=3,
    )
    # Amber edge for feature bar
    bars[0].set_edgecolor(AMBER)
    bars[0].set_linewidth(1.2)

    # Annotations
    for yi, bar, row in zip(y, bars, rows):
        auc = row["auc"]
        if row["p"] is not None:
            txt = f"AUC = {auc:.3f}  ({_fmt_p(row['p'])})"
        else:
            txt = f"AUC = {auc:.3f}"
        # Place text just past bar end; if near right edge, inside bar
        if auc < 0.68:
            ax.text(
                auc + 0.008,
                yi,
                txt,
                va="center",
                ha="left",
                fontsize=8.5,
                color=NAVY,
                fontweight="semibold",
                zorder=4,
            )
        else:
            ax.text(
                auc - 0.01,
                yi,
                txt,
                va="center",
                ha="right",
                fontsize=8.5,
                color="white",
                fontweight="bold",
                zorder=4,
            )

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9, color=NAVY)
    ax.set_xlabel("ROC-AUC (Dementia vs Normal, CAUEEG)", fontsize=10, color=NAVY)
    ax.set_xlim(0.45, 0.75)
    ax.set_ylim(-0.7, n - 0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)
    ax.tick_params(colors=SLATE)
    ax.xaxis.label.set_color(NAVY)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.7, color="#d1d5db", zorder=0)

    # Compact legend for band + chance + color meaning
    legend_handles = [
        Patch(
            facecolor=NULL_BAND,
            edgecolor=SLATE,
            label=f"Latent-probe permutation null (p5–p95): {p5:.3f}–{p95:.3f}",
        ),
        Line2D([0], [0], color=CHANCE, linestyle="--", linewidth=1.3, label="Chance (AUC = 0.5)"),
        Patch(facecolor=AMBER_FILL, edgecolor=AMBER, label=r"Feature reference ($\theta/\alpha$)"),
        Patch(facecolor=TEAL, edgecolor=NAVY, label=r"Twin latent readouts ($\mu_{\mathrm{base}}$)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=7.5,
        frameon=True,
        fancybox=False,
        edgecolor="#e5e7eb",
        framealpha=0.96,
    )

    # Small gap callout: feature vs best latent
    feat_auc = rows[0]["auc"]
    best_auc = rows[1]["auc"]
    # Bracket between feature (top) and best latent (second)
    y_feat, y_best = y[0], y[1]
    x_bracket = 0.735
    ax.annotate(
        "",
        xy=(x_bracket, y_best),
        xytext=(x_bracket, y_feat),
        arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.2),
        zorder=5,
    )
    ax.text(
        x_bracket + 0.004,
        0.5 * (y_feat + y_best),
        f"gap\n{feat_auc - best_auc:.3f}",
        va="center",
        ha="left",
        fontsize=7,
        color=AMBER,
        fontweight="bold",
        zorder=5,
    )

    fig.tight_layout(pad=0.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "fig5_encoding_analysis.png"
    svg = OUT_DIR / "fig5_encoding_analysis.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    for r in rows:
        print(f"{r['label'].replace(chr(10), ' ')}: AUC={r['auc']:.6f} p={r['p']}  [{r['json_key']}]")
    print(f"null band p5={p5:.6f} p95={p95:.6f}")
    return png


if __name__ == "__main__":
    out = make_figure()
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.svg')}")
