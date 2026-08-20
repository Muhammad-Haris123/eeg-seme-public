"""
Classical-vs-latent encoding bake-off panel (SEME Encoding axis).

READ-ONLY of locked probe/head JSON. Emphasizes incremental-value contrast
for Path A Week 3 without new experiments.

Outputs:
  paper/figures/fig_bakeoff_encoding.png
  paper/figures/fig_bakeoff_encoding.svg
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
PROBE = ROOT / "models" / "validation" / "layer5e_caueeg_latent_probe_results.json"
HEAD = ROOT / "models" / "validation" / "layer5e_caueeg_classifier_head_results.json"
RAW = ROOT / "models" / "validation" / "layer5e_caueeg_raw2185_probe_results.json"
DELONG = ROOT / "models" / "validation" / "delong_oof_probe_results.json"

NAVY = "#1f3a5f"
TEAL = "#0f766e"
AMBER = "#b45309"
SLATE = "#6b7280"


def main() -> int:
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    head = json.loads(HEAD.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8")) if RAW.exists() else {}

    latent = float(probe["latent_probe_dementia_vs_normal"]["cv"]["roc_auc"]["mean"])
    theta = float(
        probe["feature_probe_theta_alpha_dementia_vs_normal"]["cv"]["roc_auc"]["mean"]
    )
    packed = float(
        raw.get("raw2185_logistic_dementia_vs_normal", {})
        .get("cv", {})
        .get("roc_auc", {})
        .get("mean", 0.699)
    )
    best = float(head["best_head"]["oof_roc_auc"])

    delong_note = ""
    if DELONG.exists():
        d = json.loads(DELONG.read_text(encoding="utf-8"))
        for c in d.get("comparisons", []):
            if c.get("model_a") == "latent_probe" and c.get("model_b") == "packed_2185":
                delong_note = f"DeLong latent vs 2185-D: z={c['z']:.2f}, p={c['p_two_sided']:.2g}"
                break

    labels = [
        "Packed 2185-D logistic\n(classical feature reference)",
        "Theta/alpha ratio\n(classical spectral reference)",
        "Best nested head on μ_base\n(OOF)",
        "Latent probe on μ_base\n(mean-fold)",
    ]
    values = [packed, theta, best, latent]
    colors = [AMBER, AMBER, TEAL, NAVY]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, color=colors, height=0.62, edgecolor="white", linewidth=0.6)
    ax.axvline(0.5, color=SLATE, linestyle="--", linewidth=1.0, label="Chance (0.5)")
    for yi, v in zip(y, values):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", ha="left", fontsize=9, color=NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.45, 0.78)
    ax.set_xlabel("ROC-AUC (CAUEEG Dementia vs Normal, n=727)", fontsize=10)
    ax.set_title(
        "SEME Encoding bake-off: classical features vs frozen-twin latents",
        fontsize=11,
        color=NAVY,
        pad=10,
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    if delong_note:
        ax.text(
            0.99,
            0.02,
            delong_note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=SLATE,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "fig_bakeoff_encoding.png"
    svg = OUT / "fig_bakeoff_encoding.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png}")
    print(f"[saved] {svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
