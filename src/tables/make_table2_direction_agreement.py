"""
Table 2 — Cross-dataset drug-response direction agreement.

Sources:
  models/validation/complete_validation_report_v3.json
    layer5.cross_dataset
    layer5b_ad_labeled_external.cross_dataset
    layer5d_padic_external.cross_dataset
  Mirrors:
    data/ad_labeled_external/validation/layer5b_results.json
    models/validation/layer5d_padic_results.json

Outputs:
  paper/tables/table2_direction_agreement.md
  paper/tables/table2_direction_agreement.tex
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "tables"
V3_JSON = PROJECT_ROOT / "models" / "validation" / "complete_validation_report_v3.json"


def _load_rows() -> list[dict]:
    rep = json.loads(V3_JSON.read_text(encoding="utf-8"))
    tuh = rep["layer5"]["cross_dataset"]
    osf = rep["layer5b_ad_labeled_external"]["cross_dataset"]
    padic = rep["layer5d_padic_external"]["cross_dataset"]

    return [
        {
            "dataset": "TUH",
            "n": str(int(tuh["n_tuh_used"])),
            "agreement": tuh["direction_agreement_total"],
            "r": f"{float(tuh['effect_magnitude_correlation']):.3f}",
            "status": "1st confirmation",
            "json_key": "layer5.cross_dataset",
        },
        {
            "dataset": "OSF",
            "n": str(int(osf["n_used"])),
            "agreement": osf["direction_agreement_total"],
            "r": f"{float(osf['effect_magnitude_correlation']):.3f}",
            "status": "2nd confirmation (independent cohort)",
            "json_key": "layer5b_ad_labeled_external.cross_dataset",
        },
        {
            "dataset": "P-ADIC",
            "n": str(int(padic["n_used"])),
            "agreement": padic["direction_agreement_total"],
            "r": f"{float(padic['effect_magnitude_correlation']):.3f}",
            "status": "3rd confirmation (independent cohort, different acquisition)",
            "json_key": "layer5d_padic_external.cross_dataset",
        },
    ]


def write_markdown(path: Path, rows: list[dict]) -> None:
    caption = (
        "Band-level drug-response direction agreement between each external cohort "
        "and the training-cohort constrained simulations. Agreement is scored on five "
        "feature blocks × two drugs (Donepezil, Memantine; maximum 10/10). "
        "Correlation r is the Pearson correlation of mean drug-minus-baseline effect "
        "vectors (`effect_magnitude_correlation`). Replication status follows the "
        "Results Part A escalating-evidence sequence. CAUEEG is not included in the "
        "locked 10/10 direction claim."
    )
    lines = [
        "# Table 2. Drug-response direction agreement",
        "",
        f"**Caption (place ABOVE the table, Elsevier):** {caption}",
        "",
        "| Dataset | N | Direction agreement (fraction) | Correlation r | Replication status |",
        "|---------|---|-------------------------------|---------------|--------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['n']} | {r['agreement']} | {r['r']} | {r['status']} |"
        )
    lines.extend(
        [
            "",
            "## Source notes",
            "",
            "- Primary: `models/validation/complete_validation_report_v3.json`",
            "- TUH: `layer5.cross_dataset`",
            "- OSF: `layer5b_ad_labeled_external.cross_dataset` "
            "(mirror: `data/ad_labeled_external/validation/layer5b_results.json`)",
            "- P-ADIC: `layer5d_padic_external.cross_dataset` "
            "(mirror: `models/validation/layer5d_padic_results.json`)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(path: Path, rows: list[dict]) -> None:
    body = []
    for r in rows:
        # Escape status parentheses are fine; long status may need to wrap in p{} later
        body.append(
            f"  {r['dataset']} & {r['n']} & {r['agreement']} & {r['r']} & {r['status']} \\\\"
        )
    tex = r"""% Table 2 — Drug-response direction agreement
% Caption placement: ABOVE the tabular (Elsevier / CBM convention).
% Requires: \usepackage{booktabs}
\begin{table}[!t]
\centering
\caption{Locked fold-0 band-level drug-response direction agreement between each
external cohort and the training-cohort constrained signature. Agreement is scored
on five feature blocks $\times$ two drugs (Donepezil, Memantine; maximum 10/10).
Correlation $r$ is the Pearson correlation of mean drug-minus-baseline effect
vectors (\texttt{effect\_magnitude\_correlation}). These are the paper checkpoint
values, not five-fold re-simulation fold-0 scores. Replication status follows the
Results Part~A escalating-evidence sequence. CAUEEG is not included in the locked
10/10 direction claim.}
\label{tab:direction-agreement}
\begin{tabular}{@{}llllp{5.2cm}@{}}
\toprule
Dataset & N & Direction agreement & Correlation $r$ & Replication status \\
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
    md = OUT_DIR / "table2_direction_agreement.md"
    tex = OUT_DIR / "table2_direction_agreement.tex"
    write_markdown(md, rows)
    write_latex(tex, rows)
    for r in rows:
        print(f"{r['dataset']}: {r['agreement']}  r={r['r']}  n={r['n']}  [{r['json_key']}]")
    print(f"Wrote {md}")
    print(f"Wrote {tex}")


if __name__ == "__main__":
    main()
