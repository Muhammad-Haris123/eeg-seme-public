"""
Table 3 — Twin drug-response magnitude discrimination (all primary and
secondary contrasts).

Sources:
  models/validation/complete_validation_report_v3.json
    layer5.abnormal_vs_normal
    layer5b_ad_labeled_external (discrimination + sensitivity in mirror)
    layer5d_padic_external.discrimination / disease_label_ablation
    layer5e_caueeg_external.discrimination / disease_label_ablation
  data/ad_labeled_external/validation/layer5b_results.json
  models/validation/layer5d_padic_results.json
  models/validation/layer5e_caueeg_results.json

Outputs:
  paper/tables/table3_diagnostic_results.md
  paper/tables/table3_diagnostic_results.tex
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "tables"
V3_JSON = PROJECT_ROOT / "models" / "validation" / "complete_validation_report_v3.json"
L5B_JSON = PROJECT_ROOT / "data" / "ad_labeled_external" / "validation" / "layer5b_results.json"


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "$p < 0.001$"
    return f"${p:.3g}$"


def _fmt_p_md(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"{p:.3g}"


def _fmt_d(d: float) -> str:
    return f"{d:.2f}"


def _load_rows() -> list[dict]:
    rep = json.loads(V3_JSON.read_text(encoding="utf-8"))
    l5b = json.loads(L5B_JSON.read_text(encoding="utf-8"))

    tuh = rep["layer5"]["abnormal_vs_normal"]
    osf = rep["layer5b_ad_labeled_external"]["discrimination"]
    osf_sens = l5b["sensitivity_disease_label_fixed"]
    # Cohort-level verdict recorded by the validation run
    osf_verdict = l5b["honest_verdict"]["status"]  # PASS_WITH_CAVEATS

    padic = rep["layer5d_padic_external"]["discrimination"]
    padic_ab = rep["layer5d_padic_external"]["disease_label_ablation"]

    caueeg = rep["layer5e_caueeg_external"]["discrimination"]
    caueeg_ab = rep["layer5e_caueeg_external"]["disease_label_ablation"]
    kw = caueeg["kruskal_three_group"]

    rows: list[dict] = []

    def section(name: str) -> None:
        rows.append({"kind": "section", "dataset": name})

    def contrast(
        dataset: str,
        contrast: str,
        n_str: str,
        p: float | None,
        d_str: str,
        status: str,
        *,
        p_is_kruskal: bool = False,
    ) -> None:
        rows.append(
            {
                "kind": "data",
                "dataset": dataset,
                "contrast": contrast,
                "n": n_str,
                "p": p,
                "p_is_kruskal": p_is_kruskal,
                "d": d_str,
                "status": status,
            }
        )

    # --- TUH ---
    section("TUH")
    contrast(
        "TUH",
        "Abnormal vs normal",
        f"{tuh['n_abnormal']} / {tuh['n_normal']}",
        float(tuh["mannwhitney_p"]),
        _fmt_d(float(tuh["cohens_d"])),
        tuh["status"],  # FAIL
    )

    # --- OSF ---
    section("OSF")
    contrast(
        "OSF",
        "AD vs HC",
        f"{osf['n_ad']} / {osf['n_hc']}",
        float(osf["mannwhitney_p"]),
        _fmt_d(float(osf["cohens_d"])),
        osf_verdict,  # PASS_WITH_CAVEATS (honest_verdict); discrimination.status was PASS
    )
    contrast(
        "OSF",
        "AD vs HC (disease=0)",
        f"{osf['n_ad']} / {osf['n_hc']}",
        float(osf_sens["all_disease_0"]["p"]),
        _fmt_d(float(osf_sens["all_disease_0"]["cohens_d"])),
        "—",  # no status field in sensitivity JSON
    )
    contrast(
        "OSF",
        "AD vs HC (disease=1)",
        f"{osf['n_ad']} / {osf['n_hc']}",
        float(osf_sens["all_disease_1"]["p"]),
        _fmt_d(float(osf_sens["all_disease_1"]["cohens_d"])),
        "—",
    )

    # --- P-ADIC ---
    section("P-ADIC")
    contrast(
        "P-ADIC",
        "AD vs HC",
        f"{padic['n_ad']} / {padic['n_hc']}",
        float(padic["mannwhitney_p"]),
        _fmt_d(float(padic["cohens_d"])),
        padic["status"],
    )
    contrast(
        "P-ADIC",
        "AD vs HC (disease=0)",
        f"{padic_ab['disease_fixed_0']['n_ad']} / {padic_ab['disease_fixed_0']['n_hc']}",
        float(padic_ab["disease_fixed_0"]["mannwhitney_p"]),
        _fmt_d(float(padic_ab["disease_fixed_0"]["cohens_d"])),
        padic_ab["disease_fixed_0"]["status"],
    )
    contrast(
        "P-ADIC",
        "AD vs HC (disease=1)",
        f"{padic_ab['disease_fixed_1']['n_ad']} / {padic_ab['disease_fixed_1']['n_hc']}",
        float(padic_ab["disease_fixed_1"]["mannwhitney_p"]),
        _fmt_d(float(padic_ab["disease_fixed_1"]["cohens_d"])),
        padic_ab["disease_fixed_1"]["status"],
    )

    # --- CAUEEG ---
    section("CAUEEG")
    prim = caueeg["primary_dementia_vs_normal"]
    contrast(
        "CAUEEG",
        "Dementia vs Normal",
        f"{prim['n_dementia']} / {prim['n_normal']}",
        float(prim["mannwhitney_p"]),
        _fmt_d(float(prim["cohens_d"])),
        prim["status"],
    )
    dvm = caueeg["dementia_vs_mci"]
    contrast(
        "CAUEEG",
        "Dementia vs MCI",
        f"{dvm['n_dementia']} / {dvm['n_mci']}",
        float(dvm["mannwhitney_p"]),
        _fmt_d(float(dvm["cohens_d"])),
        dvm["status"],
    )
    mvn = caueeg["mci_vs_normal"]
    contrast(
        "CAUEEG",
        "MCI vs Normal",
        f"{mvn['n_mci']} / {mvn['n_normal']}",
        float(mvn["mannwhitney_p"]),
        _fmt_d(float(mvn["cohens_d"])),
        mvn["status"],
    )
    adv = caueeg["ad_tagged_dementia_vs_normal"]
    contrast(
        "CAUEEG",
        "AD-tagged Dementia vs Normal",
        f"{adv['n_dementia_ad_tag']} / {adv['n_normal']}",
        float(adv["mannwhitney_p"]),
        _fmt_d(float(adv["cohens_d"])),
        adv["status"],
    )
    contrast(
        "CAUEEG",
        "Normal / MCI / Dementia (Kruskal–Wallis)",
        f"{prim['n_normal']} / {dvm['n_mci']} / {prim['n_dementia']}",
        float(kw["p"]),
        f"H = {float(kw['H']):.2f}",
        caueeg["status"],  # FAIL (parent discrimination block)
        p_is_kruskal=True,
    )
    contrast(
        "CAUEEG",
        "Dementia vs Normal (disease=0)",
        f"{caueeg_ab['disease_fixed_0']['n_dementia']} / {caueeg_ab['disease_fixed_0']['n_normal']}",
        float(caueeg_ab["disease_fixed_0"]["mannwhitney_p"]),
        _fmt_d(float(caueeg_ab["disease_fixed_0"]["cohens_d"])),
        caueeg_ab["disease_fixed_0"]["status"],
    )
    contrast(
        "CAUEEG",
        "Dementia vs Normal (disease=1)",
        f"{caueeg_ab['disease_fixed_1']['n_dementia']} / {caueeg_ab['disease_fixed_1']['n_normal']}",
        float(caueeg_ab["disease_fixed_1"]["mannwhitney_p"]),
        _fmt_d(float(caueeg_ab["disease_fixed_1"]["cohens_d"])),
        caueeg_ab["disease_fixed_1"]["status"],
    )

    return rows


def write_markdown(path: Path, rows: list[dict]) -> None:
    caption = (
        "Twin latent drug-response magnitude discrimination across escalating external "
        "cohorts. Status strings are taken from the validation JSON (`FAIL`, "
        "`PASS_WITH_CAVEATS`). OSF primary uses `honest_verdict.status` "
        "(`PASS_WITH_CAVEATS`); OSF disease-label sensitivity rows have no status "
        "field (shown as —). Kruskal–Wallis row reports H instead of Cohen's d and "
        "Kruskal p in the p column."
    )
    lines = [
        "# Table 3. Diagnostic magnitude discrimination results",
        "",
        f"**Caption (place ABOVE the table, Elsevier):** {caption}",
        "",
        "**Placement judgment:** Prefer **Supplementary Table** (or main-text Table with "
        "primary contrasts only). Full table has 14 contrast rows + 4 dataset section "
        "breaks; dense for a single-column main text.",
        "",
        "| Dataset | Contrast | N (group A / group B) | Mann-Whitney p | Cohen's d | Status |",
        "|---------|----------|----------------------|----------------|-----------|--------|",
    ]
    for r in rows:
        if r["kind"] == "section":
            lines.append(f"| **{r['dataset']}** | | | | | |")
            continue
        p_col = "—" if r["p"] is None else _fmt_p_md(r["p"])
        if r.get("p_is_kruskal"):
            p_col = f"Kruskal {_fmt_p_md(r['p'])}"
        lines.append(
            f"| {r['dataset']} | {r['contrast']} | {r['n']} | {p_col} | {r['d']} | {r['status']} |"
        )
    lines.extend(
        [
            "",
            "## Source notes",
            "",
            "- TUH: `complete_validation_report_v3.json` → `layer5.abnormal_vs_normal`",
            "- OSF: `layer5b_ad_labeled_external.discrimination`; "
            "`data/ad_labeled_external/validation/layer5b_results.json` "
            "(`honest_verdict`, `sensitivity_disease_label_fixed`)",
            "- P-ADIC: `layer5d_padic_external.discrimination` + `disease_label_ablation`",
            "- CAUEEG: `layer5e_caueeg_external.discrimination` "
            "(pairwise + `kruskal_three_group`) + `disease_label_ablation`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(path: Path, rows: list[dict]) -> None:
    body: list[str] = []
    for r in rows:
        if r["kind"] == "section":
            body.append(r"  \multicolumn{6}{@{}l@{}}{\textbf{" + r["dataset"] + r"}} \\")
            body.append(r"  \midrule")
            continue
        if r["p"] is None:
            p_tex = "—"
        elif r.get("p_is_kruskal"):
            p_tex = "Kruskal " + _fmt_p(r["p"])
        else:
            p_tex = _fmt_p(r["p"])
        # Escape underscores in status
        status = r["status"].replace("_", r"\_")
        contrast = r["contrast"].replace("–", "--")
        body.append(
            f"  {r['dataset']} & {contrast} & {r['n']} & {p_tex} & {r['d']} & {status} \\\\"
        )

    tex = r"""% Table 3 — Diagnostic magnitude discrimination (all contrasts)
% Caption ABOVE tabular (Elsevier). Requires booktabs.
% Placement: consider Supplementary Material (14 contrast rows); see manifest.
\begin{table*}[!t]
\centering
\caption{Twin latent drug-response magnitude discrimination across escalating
external cohorts. Status strings are taken from the validation JSON
(\texttt{FAIL}, \texttt{PASS\_WITH\_CAVEATS}). The OSF primary row uses
\texttt{honest\_verdict.status}; OSF disease-label sensitivity rows have no
status field (shown as ---). The Kruskal--Wallis row reports $H$ instead of
Cohen's $d$ and Kruskal $p$ in the $p$ column.}
\label{tab:diagnostic-results}
\footnotesize
\begin{tabular}{@{}llp{2.4cm}lll@{}}
\toprule
Dataset & Contrast & N (A / B) & Mann--Whitney $p$ & Cohen's $d$ & Status \\
\midrule
"""
    tex += "\n".join(body) + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    path.write_text(tex, encoding="utf-8")


def main() -> None:
    rows = _load_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / "table3_diagnostic_results.md"
    tex = OUT_DIR / "table3_diagnostic_results.tex"
    write_markdown(md, rows)
    write_latex(tex, rows)
    n_data = sum(1 for r in rows if r["kind"] == "data")
    print(f"data rows: {n_data}")
    for r in rows:
        if r["kind"] == "section":
            print(f"--- {r['dataset']} ---")
        else:
            print(f"  {r['contrast']}: p={r['p']} d={r['d']} status={r['status']}")
    print(f"Wrote {md}")
    print(f"Wrote {tex}")


if __name__ == "__main__":
    main()
