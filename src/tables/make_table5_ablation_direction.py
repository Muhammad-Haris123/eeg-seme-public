"""
Table 5 — Matched unconstrained vs constrained direction (and prior-only).

Sources:
  models/validation/unconstrained_external_battery.json
  models/validation/complete_validation_report_v3.json
  models/validation/prior_only_direction.json

Outputs:
  paper/tables/table5_ablation_direction.{md,tex}
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "paper" / "tables"
UNCONST = PROJECT_ROOT / "models" / "validation" / "unconstrained_external_battery.json"
V3 = PROJECT_ROOT / "models" / "validation" / "complete_validation_report_v3.json"
PRIOR = PROJECT_ROOT / "models" / "validation" / "prior_only_direction.json"


def main() -> int:
    u = json.loads(UNCONST.read_text(encoding="utf-8"))
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    pr = json.loads(PRIOR.read_text(encoding="utf-8"))

    constrained = {
        "TUH": v3["layer5"]["cross_dataset"],
        "OSF": v3["layer5b_ad_labeled_external"]["cross_dataset"],
        "P-ADIC": v3["layer5d_padic_external"]["cross_dataset"],
    }
    un_map = {
        "TUH": u["cohorts"]["tuh"]["direction"],
        "OSF": u["cohorts"]["osf"]["direction"],
        "P-ADIC": u["cohorts"]["padic"]["direction"],
    }

    rows = []
    for name in ("TUH", "OSF", "P-ADIC"):
        c = constrained[name]
        un = un_map[name]
        n_key = "n_tuh_used" if name == "TUH" else "n_used"
        rows.append(
            {
                "dataset": name,
                "n": str(int(c.get(n_key) or un.get("n_used"))),
                "c_agree": c["direction_agreement_total"],
                "c_r": f"{float(c['effect_magnitude_correlation']):.3f}",
                "u_agree": un["direction_agreement_total"],
                "u_r": f"{float(un['effect_magnitude_correlation']):.3f}",
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Table 5. Matched unconstrained vs constrained direction agreement",
        "",
        "| Dataset | n | Constrained agree | Constrained r | Unconstrained agree | Unconstrained r |",
        "|---|---:|---|---:|---|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['dataset']} | {r['n']} | {r['c_agree']} | {r['c_r']} | {r['u_agree']} | {r['u_r']} |"
        )
    md_lines += [
        "",
        f"Prior-only vs constrained training signature: "
        f"{pr['direction_agreement_total']}; mean effect-vector r = "
        f"{float(pr['effect_magnitude_correlation_mean']):.3f} "
        f"(Donepezil r = {float(pr['donepezil']['effect_corr']):.3f}; "
        f"Memantine r = {float(pr['memantine']['effect_corr']):.3f}).",
        "",
        "Sources: `unconstrained_external_battery.json`, `complete_validation_report_v3.json`, "
        "`prior_only_direction.json`.",
        "",
    ]
    (OUT_DIR / "table5_ablation_direction.md").write_text("\n".join(md_lines), encoding="utf-8")

    tex = r"""\begin{table}[!t]
\centering
\caption{Matched unconstrained versus constrained direction agreement against the constrained training signature. Prior-only signed effects also reach 10/10 on signs with lower mean effect-vector correlation ($r{=}0.636$).}
\label{tab:ablation-direction}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}
\toprule
Dataset & $n$ & Constr.\ agree & Constr.\ $r$ & Unconstr.\ agree & Unconstr.\ $r$ \\
\midrule
"""
    for r in rows:
        tex += (
            f"{r['dataset']} & {r['n']} & {r['c_agree']} & {r['c_r']} & "
            f"{r['u_agree']} & {r['u_r']} \\\\\n"
        )
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    (OUT_DIR / "table5_ablation_direction.tex").write_text(tex, encoding="utf-8")
    print(f"[wrote] {OUT_DIR / 'table5_ablation_direction.md'}")
    print(f"[wrote] {OUT_DIR / 'table5_ablation_direction.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
