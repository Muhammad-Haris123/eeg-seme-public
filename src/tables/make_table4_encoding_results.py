"""
Table 4 — CAUEEG encoding analysis (pairs with Fig. 5).

Sources:
  models/validation/layer5e_caueeg_latent_probe_results.json
  models/validation/layer5e_caueeg_classifier_head_results.json

Outputs:
  paper/tables/table4_encoding_results.md
  paper/tables/table4_encoding_results.tex
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "tables"
PROBE_JSON = PROJECT_ROOT / "models" / "validation" / "layer5e_caueeg_latent_probe_results.json"
HEAD_JSON = PROJECT_ROOT / "models" / "validation" / "layer5e_caueeg_classifier_head_results.json"


def _fmt_auc(x: float) -> str:
    return f"{x:.3f}"


def _fmt_delta(x: float) -> str:
    # always signed, 3 decimals
    return f"{x:+.3f}"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "--"
    if p < 0.001:
        return "$p < 0.001$"
    return f"${p:.3f}$"


def _fmt_p_md(p: float | None) -> str:
    if p is None:
        return "--"
    if p < 0.001:
        return "p < 0.001"
    return f"{p:.3f}"


def _load_rows() -> list[dict]:
    probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    head = json.loads(HEAD_JSON.read_text(encoding="utf-8"))

    feat = probe["feature_probe_theta_alpha_dementia_vs_normal"]["cv"]["oof"]
    lat = probe["latent_probe_dementia_vs_normal"]["cv"]["oof"]
    ceil = float(feat["roc_auc"])

    rows = [
        {
            "model": r"$\theta/\alpha$ feature probe (reference)",
            "model_md": "θ/α feature probe (reference)",
            "input": "Band-power θ/α ratio",
            "auc": float(feat["roc_auc"]),
            "bal": float(feat["balanced_accuracy"]),
            "p": None,
            "json_key": "feature_probe_...cv.oof",
        },
        {
            "model": r"Untuned logistic regression",
            "model_md": "Untuned logistic regression",
            "input": r"$\mu_{\mathrm{base}}$",
            "input_md": "μ_base",
            "auc": float(lat["roc_auc"]),
            "bal": float(lat["balanced_accuracy"]),
            "p": float(probe["permutation_test_latent"]["permutation_p_value"]),
            "json_key": "latent_probe_...cv.oof / permutation_test_latent",
        },
        {
            "model": r"Tuned L2 logistic regression",
            "model_md": "Tuned L2 logistic regression",
            "input": r"$\mu_{\mathrm{base}}$",
            "input_md": "μ_base",
            "auc": float(head["best_head"]["oof_roc_auc"]),
            "bal": float(head["best_head"]["oof_balanced_accuracy"]),
            "p": float(head["permutation_best_head"]["permutation_p_value"]),
            "json_key": "best_head / permutation_best_head",
        },
        {
            "model": "MLP",
            "model_md": "MLP",
            "input": r"$\mu_{\mathrm{base}}$",
            "input_md": "μ_base",
            "auc": float(head["heads"]["mlp"]["oof"]["roc_auc"]),
            "bal": float(head["heads"]["mlp"]["oof"]["balanced_accuracy"]),
            "p": None,
            "json_key": "heads.mlp.oof",
        },
        {
            "model": "HistGradientBoosting",
            "model_md": "HistGradientBoosting",
            "input": r"$\mu_{\mathrm{base}}$",
            "input_md": "μ_base",
            "auc": float(head["heads"]["hist_gradient_boosting"]["oof"]["roc_auc"]),
            "bal": float(head["heads"]["hist_gradient_boosting"]["oof"]["balanced_accuracy"]),
            "p": None,
            "json_key": "heads.hist_gradient_boosting.oof",
        },
        {
            "model": r"Tuned LogReg ($\mu$+logvar control)",
            "model_md": "Tuned LogReg (μ+logvar control)",
            "input": r"$\mu_{\mathrm{base}}$ + logvar",
            "input_md": "μ_base + logvar",
            "auc": float(head["mu_logvar_control"]["oof"]["roc_auc"]),
            "bal": float(head["mu_logvar_control"]["oof"]["balanced_accuracy"]),
            "p": None,
            "json_key": "mu_logvar_control.oof",
        },
    ]

    for r in rows:
        r["delta"] = float(r["auc"] - ceil)
        if "input_md" not in r:
            r["input_md"] = r["input"]
    return rows


def write_markdown(path: Path, rows: list[dict]) -> None:
    caption = (
        "CAUEEG Dementia vs Normal encoding analysis (n = 727). "
        "OOF ROC-AUC and balanced accuracy are out-of-fold aggregates. "
        "Permutation p-values are reported where computed (original latent probe: "
        "200 label shuffles of 5-fold CV; best head: 200 shuffles of nested CV). "
        "Delta is OOF AUC minus the θ/α feature-probe OOF AUC (ceiling). "
        "Pairs with Fig. 5."
    )
    lines = [
        "# Table 4. Encoding analysis results (CAUEEG)",
        "",
        f"**Caption (place ABOVE the table, Elsevier):** {caption}",
        "",
        "| Model | Input representation | OOF ROC-AUC | Balanced accuracy | Permutation p-value | vs. theta/alpha ceiling |",
        "|-------|---------------------|-------------|-------------------|---------------------|-------------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_md']} | {r['input_md']} | {_fmt_auc(r['auc'])} | "
            f"{_fmt_auc(r['bal'])} | {_fmt_p_md(r['p'])} | {_fmt_delta(r['delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Source notes",
            "",
            "- `models/validation/layer5e_caueeg_latent_probe_results.json`",
            "- `models/validation/layer5e_caueeg_classifier_head_results.json`",
            "- Ceiling = feature_probe OOF ROC-AUC; deltas = row OOF AUC − ceiling",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(path: Path, rows: list[dict]) -> None:
    body = []
    for r in rows:
        body.append(
            f"  {r['model']} & {r['input']} & {_fmt_auc(r['auc'])} & "
            f"{_fmt_auc(r['bal'])} & {_fmt_p(r['p'])} & {_fmt_delta(r['delta'])} \\\\"
        )
    tex = r"""% Table 4 — Encoding analysis (pairs with Fig. 5)
% Caption ABOVE tabular. Requires booktabs.
\begin{table}[!t]
\centering
\caption{CAUEEG Dementia vs Normal encoding analysis ($n = 727$).
Out-of-fold (OOF) ROC-AUC and balanced accuracy are reported.
Permutation $p$-values are included where computed (original latent probe:
200 label shuffles of 5-fold CV; best head: 200 shuffles of nested CV).
The final column is OOF AUC minus the $\theta/\alpha$ feature-probe OOF AUC
(ceiling). Complements Fig.~\ref{fig:encoding-analysis}.}
\label{tab:encoding-results}
\footnotesize
\begin{tabular}{@{}llcccc@{}}
\toprule
Model & Input & OOF ROC-AUC & Bal.\ acc. & Perm.\ $p$ & vs.\ $\theta/\alpha$ \\
\midrule
"""
    tex += "\n".join(body) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    path.write_text(tex, encoding="utf-8")


def main() -> None:
    rows = _load_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / "table4_encoding_results.md"
    tex = OUT_DIR / "table4_encoding_results.tex"
    write_markdown(md, rows)
    write_latex(tex, rows)
    for r in rows:
        print(
            f"{r['json_key']}: AUC={r['auc']:.6f} bal={r['bal']:.6f} "
            f"p={r['p']} delta={r['delta']:+.6f}"
        )
    print(f"Wrote {md}")
    print(f"Wrote {tex}")


if __name__ == "__main__":
    main()
