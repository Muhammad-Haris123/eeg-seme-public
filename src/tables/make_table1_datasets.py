"""
Table 1 — External validation datasets (Methods / Results Part B).

Sources (metadata):
  models/validation/layer5_domain_shift_diagnosis.md
  models/validation/layer5b_ad_labeled_external_diagnosis.md
  models/validation/layer5d_padic_external_diagnosis.md
  models/validation/layer5e_caueeg_external_diagnosis.md
  models/validation/complete_validation_report_v3.json (n counts)

Outputs:
  paper/tables/table1_datasets.md
  paper/tables/table1_datasets.tex
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "tables"

# Concise cells (~6 words where possible). Roles match Results Part B arc.
ROWS = [
    {
        "dataset": "TUH",
        "n_total": "200",
        "n_by_group": "Abn 131 / Nor 69",
        "labels": "Abnormal vs normal (proxy)",
        "sampling_rate": "Mixed EDF $\\rightarrow$ 500 Hz",
        "montage": "19ch 10--20",
        "role": "First external test; proxy labels",
        # Markdown variants (no TeX)
        "sampling_rate_md": "Mixed EDF → 500 Hz",
        "montage_md": "19ch 10–20",
    },
    {
        "dataset": "OSF",
        "n_total": "92",
        "n_by_group": "AD 80 / HC 12",
        "labels": "Folder AD vs Healthy",
        "sampling_rate": "256 $\\rightarrow$ 500 Hz",
        "montage": "19ch 10--20",
        "role": "AD/HC labels; small HC",
        "sampling_rate_md": "256 → 500 Hz",
        "montage_md": "19ch 10–20",
    },
    {
        "dataset": "P-ADIC",
        "n_total": "145",
        "n_by_group": "AD 49 / HC 96",
        "labels": "NIA-AA AD vs HC",
        "sampling_rate": "200/500 $\\rightarrow$ 500 Hz",
        "montage": "19ch 10--20 (Nihon remap)",
        "role": "Larger AD/HC; confirms null",
        "sampling_rate_md": "200/500 → 500 Hz",
        "montage_md": "19ch 10–20 (Nihon remap)",
    },
    {
        "dataset": "CAUEEG",
        "n_total": "1122",
        "n_by_group": "Nor 436 / MCI 395 / Dem 291",
        "labels": "Clinical Normal/MCI/Dementia",
        "sampling_rate": "200 $\\rightarrow$ 500 Hz",
        "montage": "19ch AVG 10--20",
        "role": "Large-N clinical; encoding analysis",
        "sampling_rate_md": "200 → 500 Hz",
        "montage_md": "19ch AVG 10–20",
    },
]

CAPTION = (
    "External EEG datasets used for escalating validation of the constrained twin. "
    "Sampling rates are native acquisition followed by the common 500 Hz analysis rate. "
    "Roles follow the Results Part B narrative sequence."
)


def write_markdown(path: Path) -> None:
    lines = [
        "# Table 1. External validation datasets",
        "",
        f"**Caption (place ABOVE the table, Elsevier):** {CAPTION}",
        "",
        "| Dataset | N (total) | N by group | Labels | Sampling rate | Montage | Role in study |",
        "|---------|-----------|------------|--------|---------------|---------|---------------|",
    ]
    for r in ROWS:
        lines.append(
            f"| {r['dataset']} | {r['n_total']} | {r['n_by_group']} | {r['labels']} | "
            f"{r['sampling_rate_md']} | {r['montage_md']} | {r['role']} |"
        )
    lines.extend(
        [
            "",
            "## Source notes",
            "",
            "- TUH: `models/validation/layer5_domain_shift_diagnosis.md`; "
            "`complete_validation_report_v3.json` → `layer5`",
            "- OSF: `models/validation/layer5b_ad_labeled_external_diagnosis.md`; "
            "`layer5b_ad_labeled_external`",
            "- P-ADIC: `models/validation/layer5d_padic_external_diagnosis.md`; "
            "`layer5d_padic_external`",
            "- CAUEEG: `models/validation/layer5e_caueeg_external_diagnosis.md`; "
            "`layer5e_caueeg_external`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(path: Path) -> None:
    # booktabs, no vertical rules; caption above (elsarticle)
    body_rows = []
    for r in ROWS:
        body_rows.append(
            "  {dataset} & {n_total} & {n_by_group} & {labels} & "
            "{sampling_rate} & {montage} & {role} \\\\".format(**r)
        )
    tex = r"""% Table 1 — External validation datasets
% Caption placement: ABOVE the tabular (Elsevier / CBM convention).
% Requires: \usepackage{booktabs}
\begin{table*}[!t]
\centering
\caption{External EEG datasets used for escalating validation of the constrained twin.
Sampling rates list native acquisition then the common 500\,Hz analysis rate.
Roles follow the Results Part~B narrative sequence.}
\label{tab:datasets}
\begin{tabular}{@{}lllllll@{}}
\toprule
Dataset & N (total) & N by group & Labels & Sampling rate & Montage & Role in study \\
\midrule
"""
    tex += "\n".join(body_rows) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    path.write_text(tex, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / "table1_datasets.md"
    tex = OUT_DIR / "table1_datasets.tex"
    write_markdown(md)
    write_latex(tex)
    print(f"Wrote {md}")
    print(f"Wrote {tex}")


if __name__ == "__main__":
    main()
