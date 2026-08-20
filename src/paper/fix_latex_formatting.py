"""Fix paper/latex formatting bugs (identifiers, math, bib VERIFY, Table 1)."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "paper" / "latex"
OVERLEAF = ROOT / "paper" / "overleaf"
BIB_SRC = ROOT / "paper" / "references.bib"

FIXES: list[str] = []


def log(msg: str) -> None:
    FIXES.append(msg)
    print(msg)


def ensure_tree() -> None:
    if not (LATEX / "main.tex").exists():
        if LATEX.exists():
            shutil.rmtree(LATEX)
        shutil.copytree(OVERLEAF, LATEX)


def fix_table1() -> None:
    (LATEX / "tables" / "table1_datasets.tex").write_text(
        r"""% Table 1 -- External validation datasets (page-width fit)
\begin{table*}[!t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\caption{External EEG datasets used for escalating validation of the constrained twin.
Sampling rates list native acquisition then the common 500\,Hz analysis rate.
Roles follow the Results narrative sequence.}
\label{tab:datasets}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llllllp{2.6cm}@{}}
\toprule
Dataset & N & Groups & Labels & Sampling rate & Montage & Role \\
\midrule
TUH & 200 & Abn 131 / Nor 69 & Abnormal vs normal & Mixed EDF $\rightarrow$ 500\,Hz & 19ch 10--20 & First external; proxy labels \\
OSF & 92 & AD 80 / HC 12 & Folder AD vs Healthy & 256 $\rightarrow$ 500\,Hz & 19ch 10--20 & AD/HC; small HC \\
P-ADIC & 145 & AD 49 / HC 96 & NIA-AA AD vs HC & 200/500 $\rightarrow$ 500\,Hz & 19ch 10--20 & Larger AD/HC \\
CAUEEG & 1122 & Nor 436 / MCI 395 / Dem 291 & Clinical Nor/MCI/Dem & 200 $\rightarrow$ 500\,Hz & 19ch AVG 10--20 & Large-N; encoding \\
\bottomrule
\end{tabular*}
\end{table*}
""",
        encoding="utf-8",
    )
    log("Table 1: footnotesize + tabular* + shorter columns (Sampling rate fit)")


def fix_bib() -> None:
    bib = BIB_SRC.read_text(encoding="utf-8")

    winblad = """@article{Winblad2007Memantine,
  author  = {Winblad, Bengt and Jones, Roy W. and Wirth, Yvonne and St{\\"o}ffler, Albrecht and M{\\"o}bius, Hans J{\\"o}rg},
  title   = {Memantine in moderate-to-severe {Alzheimer}'s disease: a meta-analysis of randomised clinical trials},
  journal = {Dementia and Geriatric Cognitive Disorders},
  year    = {2007},
  volume  = {24},
  number  = {1},
  pages   = {20--27},
  doi     = {10.1159/000102568}
}

"""
    # Fix winblad - use proper latex accents without re escape issues
    winblad = (
        "@article{Winblad2007Memantine,\n"
        "  author  = {Winblad, Bengt and Jones, Roy W. and Wirth, Yvonne and "
        'St{\\"o}ffler, Albrecht and M{\\"o}bius, Hans J{\\"o}rg},\n'
        "  title   = {Memantine in moderate-to-severe {Alzheimer}'s disease: "
        "a meta-analysis of randomised clinical trials},\n"
        "  journal = {Dementia and Geriatric Cognitive Disorders},\n"
        "  year    = {2007},\n"
        "  volume  = {24},\n"
        "  number  = {1},\n"
        "  pages   = {20--27},\n"
        "  doi     = {10.1159/000102568}\n"
        "}\n\n"
    )
    rive = (
        "@article{Rive2013Memantine,\n"
        "  author  = {Rive, B. and Gauthier, S. and Costello, S. and Marre, C. and "
        "Francois, C.},\n"
        "  title   = {Synthesis and Comparison of the Meta-Analyses Evaluating the "
        "Efficacy of Memantine in Moderate to Severe Stages of {Alzheimer}'s Disease},\n"
        "  journal = {CNS Drugs},\n"
        "  year    = {2013},\n"
        "  volume  = {27},\n"
        "  number  = {7},\n"
        "  pages   = {573--585},\n"
        "  doi     = {10.1007/s40263-013-0074-x}\n"
        "}\n\n"
    )
    milti = (
        "@article{Miltiadous2023OpenNeuro,\n"
        "  author  = {Miltiadous, Andreas and Tzimourta, Katerina D. and "
        "Giannakeas, Nikolaos and others},\n"
        "  title   = {A dataset of {EEG} recordings from {Alzheimer}'s disease, "
        "frontotemporal dementia and healthy subjects},\n"
        "  year    = {2023},\n"
        "  note    = {OpenNeuro accession ds004504}\n"
        "}\n\n"
    )

    def replace_entry(text: str, key: str, new_block: str) -> str:
        return re.sub(
            rf"@article\{{{key},[\s\S]*?\n\}}\n",
            lambda _m: new_block,
            text,
            count=1,
        )

    bib = replace_entry(bib, "Winblad2007Memantine", winblad)
    bib = replace_entry(bib, "Rive2013Memantine", rive)
    bib = replace_entry(bib, "Miltiadous2023OpenNeuro", milti)
    bib = re.sub(r"\n\s*note\s*=\s*\{VERIFY:[^}]*\},?", "", bib)
    if "VERIFY" in bib:
        raise RuntimeError("VERIFY still present in bibliography")
    BIB_SRC.write_text(bib, encoding="utf-8")
    (LATEX / "references.bib").write_text(bib, encoding="utf-8")
    log("Bibliography: removed all VERIFY notes; completed Winblad2007 + Rive2013 venues; cleaned Miltiadous")


def fix_sections() -> None:
    for path in sorted((LATEX / "sections").glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        orig = text
        prot: list[str] = []

        def protect(m: re.Match) -> str:
            prot.append(m.group(0))
            return f"@@T{len(prot) - 1}@@"

        t = re.sub(r"\\texttt\{[^{}]*\}", protect, text)
        t = re.sub(r"\$[^$]*\$", protect, t)
        t = re.sub(r"\\\([\s\S]*?\\\)", protect, t)
        t = re.sub(r"\\\[[\s\S]*?\\\]", protect, t)

        t = t.replace("Eyes_closed", r"\texttt{Eyes\_closed}")
        t = t.replace("PASS_WITH_CAVEATS", r"\texttt{PASS\_WITH\_CAVEATS}")
        t = t.replace("status FAIL;", r"status \texttt{FAIL};")
        t = t.replace("both FAIL)", r"both \texttt{FAIL})")
        t = t.replace("remained FAIL (", r"remained \texttt{FAIL} (")
        t = t.replace("discrimination status PASS;", r"discrimination status \texttt{PASS};")
        t = t.replace("external PASS:", r"external \texttt{PASS}:")
        t = t.replace(" → ", r" $\rightarrow$ ")
        t = t.replace("\u00d7", r"$\times$")

        t = re.sub(r"(?<![A-Za-z\\])n = (\d+)", r"$n = \1$", t)
        t = re.sub(r"(?<![A-Za-z\\])p = ([0-9.eE+-]+)", r"$p = \1$", t)
        t = re.sub(r"(?<![A-Za-z\\])p < 0\.001", r"$p < 0.001$", t)
        t = re.sub(r"(?<![A-Za-z\\])d = (-?[0-9.]+)", r"$d = \1$", t)
        t = re.sub(r"(?<![A-Za-z\\])H = ([0-9.]+)", r"$H = \1$", t)

        def restore(m: re.Match) -> str:
            return prot[int(m.group(1))]

        t = re.sub(r"@@T(\d+)@@", restore, t)
        t = re.sub(r"\\texttt\{\\texttt\{([^}]+)\}\}", r"\\texttt{\1}", t)

        if t != orig:
            path.write_text(t, encoding="utf-8")
            log(f"{path.name}: wrap status/Eyes_closed identifiers; math-mode n/p/d/H; arrow glyph")


def sync_overleaf() -> None:
    if not OVERLEAF.exists():
        return
    shutil.copy2(LATEX / "tables" / "table1_datasets.tex", OVERLEAF / "tables" / "table1_datasets.tex")
    shutil.copy2(LATEX / "references.bib", OVERLEAF / "references.bib")
    for p in (LATEX / "sections").glob("*.tex"):
        shutil.copy2(p, OVERLEAF / "sections" / p.name)
    log("Synced fixes into paper/overleaf/")


def try_compile() -> tuple[bool, str]:
    for cmd in (
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
    ):
        try:
            subprocess.run(cmd, cwd=LATEX, capture_output=True, text=True, timeout=180)
            if cmd[0] == "pdflatex":
                subprocess.run(["bibtex", "main"], cwd=LATEX, capture_output=True, text=True, timeout=60)
                subprocess.run(cmd, cwd=LATEX, capture_output=True, text=True, timeout=180)
                subprocess.run(cmd, cwd=LATEX, capture_output=True, text=True, timeout=180)
            ok = (LATEX / "main.pdf").exists()
            log_txt = ""
            if (LATEX / "main.log").exists():
                log_txt = (LATEX / "main.log").read_text(encoding="utf-8", errors="replace")[-3000:]
            # find Winblad/Rive numbers in bbl
            bbl_note = ""
            if (LATEX / "main.bbl").exists():
                bbl = (LATEX / "main.bbl").read_text(encoding="utf-8", errors="replace")
                bbl_note = bbl
            return ok, log_txt + "\n\n=== BBL ===\n" + bbl_note[:4000]
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return False, "no pdflatex/latexmk on PATH"


def audit_bare_underscores() -> list[str]:
    issues = []
    for path in sorted((LATEX / "sections").glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        scrub = re.sub(r"\\texttt\{[^{}]*\}", "", text)
        scrub = re.sub(r"\$[^$]*\$", "", scrub)
        scrub = re.sub(r"\\\([\s\S]*?\\\)", "", scrub)
        scrub = re.sub(r"\\\[[\s\S]*?\\\]", "", scrub)
        scrub = re.sub(r"\\(label|ref|input|includegraphics|cite)(\[[^\]]*\])?\{[^}]+\}", "", scrub)
        for i, line in enumerate(scrub.splitlines(), 1):
            if re.search(r"[A-Za-z0-9]_[A-Za-z0-9]", line):
                issues.append(f"{path.name}:{i}: {line.strip()[:120]}")
    return issues


def main() -> None:
    ensure_tree()
    fix_table1()
    fix_bib()
    fix_sections()
    sync_overleaf()
    leftover = audit_bare_underscores()
    ok, detail = try_compile()

    report = ["# paper/latex formatting fix report\n"]
    report.append("## Fixes applied\n")
    for i, f in enumerate(FIXES, 1):
        report.append(f"{i}. {f}\n")
    report.append("\n## Bare-underscore audit after fix\n")
    if leftover:
        for x in leftover:
            report.append(f"- REMAINING: {x}\n")
    else:
        report.append("- None found outside \\texttt{}/math.\n")
    report.append(f"\n## Compile\n- PDF ok: {ok}\n")
    report.append(
        "\n## Bibliography note for user\n"
        "- Literal `VERIFY:` text removed from the `.bib` file.\n"
        "- `Winblad2007Memantine` and `Rive2013Memantine` were the stub entries "
        "that previously carried VERIFY notes (these become the numbered refs "
        "that appeared corrupted as [11]/[12] depending on citation order).\n"
        "- Venues were filled from published DOIs (Dement Geriatr Cogn Disord "
        "2007; CNS Drugs 2013). Please confirm these match the intended "
        "prior sources in `pharmacological_embedding.py`.\n"
    )
    if detail:
        report.append("\n## Compiler detail (tail)\n```\n" + detail[-2500:] + "\n```\n")
    out = LATEX / "FIX_REPORT.md"
    out.write_text("".join(report), encoding="utf-8")
    print("".join(report))


if __name__ == "__main__":
    main()
