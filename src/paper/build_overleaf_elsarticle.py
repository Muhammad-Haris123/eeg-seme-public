"""
Build paper/overleaf/ Elsevier elsarticle package from paper/sections/*.md
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
OUT = PAPER / "overleaf"
SECTIONS = PAPER / "sections"
FIGS = PAPER / "figures"
TABLES = PAPER / "tables"

FIG_MAP = {
    r"Fig\.\s*1": r"Fig.~\\ref{fig:architecture}",
    r"Fig\.\s*2": r"Fig.~\\ref{fig:direction}",
    r"Fig\.\s*3": r"Fig.~\\ref{fig:diagnostic}",
    r"Fig\.\s*4": r"Fig.~\\ref{fig:pca}",
    r"Fig\.\s*5": r"Fig.~\\ref{fig:encoding}",
    r"Fig\.\s*6": r"Fig.~\\ref{fig:alpha}",
    r"Fig\.\s*A\.1": r"Fig.~\\ref{fig:traincurves}",
    r"Algorithm\s*1": r"Algorithm~\\ref{alg:train}",
    r"Table\s*1": r"Table~\\ref{tab:datasets}",
    r"Table\s*2": r"Table~\\ref{tab:ablation-direction}",
    r"Table\s*3": r"Table~\\ref{tab:direction-agreement}",
    r"Table\s*4": r"Table~\\ref{tab:diagnostic-results}",
    r"Table\s*5": r"Table~\\ref{tab:encoding-results}",
    r"Table\s*6": r"Table~\\ref{tab:multiseed-direction}",
    r"Table\s*A\.1": r"Table~\\ref{tab:architecture}",
    r"Table\s*A\.2": r"Table~\\ref{tab:compute}",
    r"Table\s*A\.3": r"Table~\\ref{tab:reviewer-defense}",
}


def _code_to_tex(inner_raw: str) -> str:
    """Render a markdown backtick span as caption-safe breakable \\texttt."""
    # Avoid \\path/\\url here: they error in captions and \\textbf moving args.
    inner = (
        inner_raw.replace("\\", "\\textbackslash{}")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )
    if len(inner_raw) >= 12 and any(c in inner_raw for c in "/_."):
        inner = (
            inner.replace("_", "\\_\\allowbreak{}")
            .replace("/", "/\\allowbreak{}")
            .replace(".", ".\\allowbreak{}")
        )
    else:
        inner = inner.replace("_", "\\_")
    return f"\\texttt{{{inner}}}"


def escape_text(s: str) -> str:
    # Protect math first
    parts = re.split(r"(\$\$[\s\S]*?\$\$|\$[^$]+\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\))", s)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # convert \( \) already ok; $$ to \[ \]
            if part.startswith("$$"):
                out.append("\\[" + part[2:-2] + "\\]")
            else:
                out.append(part)
            continue
        t = part
        # citations [Key] or [Key1, Key2]
        def cite_repl(m):
            keys = (m.group(1) + (m.group(2) or "")).replace(" ", "")
            return f"\\citep{{{keys}}}"

        t = re.sub(
            r"\[([A-Za-z][A-Za-z0-9]+)((?:\s*,\s*[A-Za-z][A-Za-z0-9]+)*)\]",
            cite_repl,
            t,
        )
        # Protect backtick spans, then escape bare _ (critical: Eyes_closed etc.)
        code_spans: list[str] = []

        def save_code(m: re.Match) -> str:
            code_spans.append(m.group(1))
            return f"@@CODE{len(code_spans) - 1}@@"

        t = re.sub(r"`([^`]+)`", save_code, t)
        # bold / italic before underscore escape so markers stay intact
        t = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", t)
        t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\textit{\1}", t)
        # special chars outside commands (rough)
        t = t.replace("&", "\\&")
        t = t.replace("%", "\\%")
        t = t.replace("#", "\\#")
        t = t.replace("\u00d7", "$\\times$")  # ×
        t = t.replace("\u2013", "--")  # en dash
        t = t.replace("\u2014", ", ")  # em dash -> comma (CBM rule)
        t = t.replace("\u03b8", "$\\theta$")
        t = t.replace("\u03b1", "$\\alpha$")
        t = t.replace("\u03bc", "$\\mu$")
        # Bare underscores -> \\_ (prevents runaway math mode in PDF)
        t = t.replace("_", "\\_")
        for j, raw in enumerate(code_spans):
            t = t.replace(f"@@CODE{j}@@", _code_to_tex(raw))
        # Fig/Table refs
        for pat, repl in FIG_MAP.items():
            t = re.sub(pat, repl, t)
        # Section references stay as words
        out.append(t)
    return "".join(out)


def md_body_to_tex(md: str, drop_top_h1: bool = True) -> str:
    lines = md.strip().splitlines()
    out: list[str] = []
    i = 0
    in_eq = False
    eq_buf: list[str] = []
    while i < len(lines):
        line = lines[i]
        if line.strip() == "$$" or line.strip() == "\\[":
            if not in_eq:
                in_eq = True
                eq_buf = []
                i += 1
                continue
        if in_eq:
            if line.strip() in ("$$", "\\]"):
                out.append("\\begin{equation}")
                out.append("\n".join(eq_buf))
                out.append("\\end{equation}")
                in_eq = False
                i += 1
                continue
            eq_buf.append(line)
            i += 1
            continue

        if line.startswith("# ") and drop_top_h1:
            # skip top-level title (handled by \section in main)
            i += 1
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            # strip leading numbering like 2.1
            title = re.sub(r"^\d+(\.\d+)*\s+", "", title)
            if title.lower().startswith("credit"):
                out.append("\\section*{CRediT authorship contribution statement}")
            elif title.lower() == "data availability":
                out.append("\\section*{Data availability}")
            elif title.lower().startswith("declaration"):
                out.append("\\section*{Declaration of competing interests}")
            elif title.lower() == "acknowledgments":
                out.append("\\section*{Acknowledgments}")
            elif title.lower() == "glossary":
                out.append("\\section*{Glossary}")
            else:
                out.append(f"\\subsection{{{escape_text(title)}}}")
            i += 1
            continue
        if line.strip() == "---":
            i += 1
            continue
        if line.startswith("- "):
            # collect list
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(escape_text(lines[i][2:].strip()))
                i += 1
            out.append("\\begin{itemize}")
            for it in items:
                out.append(f"  \\item {it}")
            out.append("\\end{itemize}")
            continue
        if not line.strip():
            out.append("")
            i += 1
            continue
        # paragraph
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("#") and not lines[i].startswith("- ") and lines[i].strip() != "---" and lines[i].strip() != "$$":
            para.append(lines[i])
            i += 1
        text = " ".join(p.strip() for p in para)
        out.append(escape_text(text))
        out.append("")
    return "\n".join(out).strip() + "\n"


def write_main() -> None:
    # Prefer locked Option A title + claim-aligned abstract (no empirical PD-transfer reading).
    abstract = (
        "Chemistry-conditioned EEG digital twins must show that constrained signatures "
        "remain concordant under external domain shift and must separately justify any "
        "diagnostic use of twin outputs. We trained a pharmacodynamically constrained "
        "CVAE ($N{=}66$) on 2185-D resting-state EEG features with fixed ChemBERTa "
        "embeddings for Donepezil and Memantine; that label names the literature-prior "
        "training objective, not empirical post-dose validation. External cohorts lack "
        "paired pre/post-drug EEG, so the primary endpoint is concordance of simulated "
        "band/connectivity signs with the training constrained signature. At the locked "
        "fold-0 seed-42 checkpoint, direction agreed at 10/10 on TUH EEG ($n{=}200$; "
        "$r{=}0.908$), OSF ($n{=}92$; $r{=}0.851$), and P-ADIC ($n{=}145$; $r{=}0.918$). "
        "These $r$ values are checkpoint-specific: five-fold means fell to "
        "$0.385$/$0.371$/$0.372$, and non-42 training seeds typically gave $r$ of 0.08 "
        "to 0.43, while signs remained mostly 9/10 to 10/10. Unconstrained twins also "
        "showed unstable continuous $r$; constrained signs exceeded unconstrained signs "
        "without establishing continuous-$r$ stabilization. A prior-only signed effect "
        "already reached 10/10 signs at mean $r{=}0.636$, so the CVAE does not uniquely "
        "explain the sign pattern. Latent magnitude did not diagnose reliably. Encoding "
        "probes placed latent ROC-AUC $0.579$ below theta/alpha $0.675$ and a raw 2185-D "
        "logistic feature-space diagnostic reference of $0.699$ (best nested head OOF "
        "AUC $0.593$; $0.699$ is not a CVAE score, ceiling, or upper bound). The "
        "strongest finding in this evaluation battery is directional/sign concordance "
        "under constraint, not seed-stable continuous fidelity, post-dose validation, "
        "or external diagnostic utility."
    )

    main = r"""%% Computer Methods and Programs in Biomedicine -- Elsevier elsarticle package
%% Created from paper/sections/ for Overleaf upload.
%% First submission: preprint layout is acceptable per Elsevier guidance.
\documentclass[preprint,12pt,3p,times]{elsarticle}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{float}
\newfloat{algorithm}{htbp}{loa}
\floatname{algorithm}{Algorithm}
\usepackage[expansion=false,protrusion=true]{microtype}
\usepackage{xurl}
\usepackage{hyperref}
\hypersetup{hidelinks,breaklinks=true,pdfauthor={},pdfcreator={}}
\urlstyle{tt}
% Allow slightly stretchy paragraphs instead of margin overflow.
\emergencystretch=3em
\tolerance=2000
\hbadness=10000
\hyphenpenalty=400
\exhyphenpenalty=100
% Soft hyphenation hints for dense technical prose
\hyphenation{phar-ma-co-dy-nam-i-cal-ly Chem-BER-Ta Done-pezil Mem-an-tine
CAU-EEG Alz-hei-mer con-nec-tiv-i-ty}

\journal{Computer Methods and Programs in Biomedicine}

\bibliographystyle{elsarticle-num}

\begin{document}

\begin{frontmatter}

\title{Constraint-Consistent Direction and Diagnostic Encoding Loss
in an Externally Assessed EEG-Drug CVAE}

\author{Authors to be completed at submission}

\begin{abstract}
""" + abstract + r"""
\end{abstract}

\begin{keyword}
pharmacodynamically constrained CVAE \sep EEG digital twin \sep drug-conditioned simulation \sep Alzheimer's disease \sep external signature assessment \sep encoding-level attenuation
\end{keyword}

\begin{highlights}
\item Locked seed-42 fold-0: 10/10 signs; high r checkpoint-specific (0.908/0.851/0.918).
\item Continuous r is fold- and seed-sensitive; signs are more stable in this battery.
\item Prior-only 10/10 at mean r=0.636; ensemble does not clearly exceed the prior.
\item No paired post-dose EEG; endpoint is signature agreement, not empirical PD transfer.
\item Latent probe AUC=0.579 vs theta/alpha=0.675 and 2185-D=0.699; head OOF=0.593.
\end{highlights}

\begin{graphicalabstract}
\centering
\includegraphics[width=\textwidth]{figures/graphical_abstract.png}
\\[0.4em]
{\footnotesize Direction scoring uses the 2185-D feature-space drug-minus-baseline effect vector, not CVAE latent means.}
\end{graphicalabstract}

\end{frontmatter}

\section{Introduction}
\label{sec:introduction}
\input{sections/01_introduction}

\section{Methods}
\label{sec:methods}
\input{sections/02_methods}

\section{Results}
\label{sec:results}
\input{sections/03_results_mechanism}
\input{sections/04_results_diagnosis}
\input{sections/05_results_encoding_analysis}
\input{sections/05b_results_constraint_strength}

\section{Discussion}
\label{sec:discussion}
\input{sections/06_discussion}

\section{Conclusion}
\label{sec:conclusion}
\input{sections/07_conclusion}

\input{sections/07_backmatter_extra}

\appendix
\renewcommand{\thetable}{A.\arabic{table}}
\renewcommand{\thefigure}{A.\arabic{figure}}
\renewcommand{\theHtable}{A.\arabic{table}}
\renewcommand{\theHfigure}{A.\arabic{figure}}
\setcounter{table}{0}
\setcounter{figure}{0}
\input{sections/S1_reproducibility}

\bibliography{references}

\end{document}
"""
    (OUT / "main.tex").write_text(main, encoding="utf-8")


def sanitize_table(tex: str) -> str:
    # ASCII-safe TeX: unicode punctuation/Greek break pdfLaTeX
    tex = tex.replace("\u2014", "--").replace("\u2013", "--")
    tex = tex.replace("\u00d7", "$\\times$")
    tex = tex.replace("\u03b8", "$\\theta$").replace("\u0398", "$\\Theta$")
    tex = tex.replace("\u03b1", "$\\alpha$").replace("\u0391", "$\\Alpha$")
    tex = tex.replace("\u03bc", "$\\mu$")
    tex = tex.replace("\u2192", "$\\rightarrow$")
    return tex


def main() -> None:
    global OUT
    staging = PAPER / "overleaf_staging"
    if staging.exists():
        shutil.rmtree(staging)
    OUT = staging
    (OUT / "sections").mkdir(parents=True)
    (OUT / "figures").mkdir(parents=True)
    (OUT / "tables").mkdir(parents=True)

    # Copy figures
    fig_copies = {
        "fig1_architecture.png": "fig1_architecture.png",
        "fig2_direction_agreement.png": "fig2_direction_agreement.png",
        "fig3_diagnostic_null.png": "fig3_diagnostic_null.png",
        "fig4_pca.png": "fig4_pca.png",
        "fig5_encoding_analysis.png": "fig5_encoding_analysis.png",
        "fig6_constraint_strength.png": "fig6_constraint_strength.png",
        "fig_s1_training_curves.png": "fig_s1_training_curves.png",
    }
    for src_name, dst_name in fig_copies.items():
        src = FIGS / src_name
        if src.exists():
            shutil.copy2(src, OUT / "figures" / dst_name)
    # graphical abstract (handle double .png.png typo)
    ga_candidates = [
        FIGS / "graphical_abstract.png",
        FIGS / "graphical_abstract.png.png",
    ]
    for ga in ga_candidates:
        if ga.exists():
            shutil.copy2(ga, OUT / "figures" / "graphical_abstract.png")
            break

    # Copy tables (sanitized)
    for tname in [
        "table1_datasets.tex",
        "table2_direction_agreement.tex",
        "table3_diagnostic_results.tex",
        "table4_encoding_results.tex",
        "table5_ablation_direction.tex",
        "table6_multiseed_direction.tex",
        "table7_architecture.tex",
        "table8_compute.tex",
        "algorithm1_training.tex",
        "table_s_reviewer_defense.tex",
    ]:
        text = sanitize_table((TABLES / tname).read_text(encoding="utf-8"))
        # Align Fig. 5 label with main.tex \label{fig:encoding}
        text = text.replace("fig:encoding-analysis", "fig:encoding")
        (OUT / "tables" / tname).write_text(text, encoding="utf-8")

    shutil.copy2(PAPER / "references.bib", OUT / "references.bib")

    # Convert sections
    mapping = [
        ("01_introduction.md", "01_introduction.tex", True),
        ("02_methods.md", "02_methods.tex", True),
        ("03_results_mechanism.md", "03_results_mechanism.tex", True),
        ("04_results_diagnosis.md", "04_results_diagnosis.tex", True),
        ("05_results_encoding_analysis.md", "05_results_encoding_analysis.tex", True),
        ("05b_results_constraint_strength.md", "05b_results_constraint_strength.tex", True),
        ("06_discussion.md", "06_discussion.tex", True),
    ]
    for md_name, tex_name, drop in mapping:
        md = (SECTIONS / md_name).read_text(encoding="utf-8")
        tex = md_body_to_tex(md, drop_top_h1=True)
        # Placeholder underscores are escaped by md_body_to_tex.
        tex = tex.replace(
            r"\_\_INPUT\_TABLE6\_MULTISEED\_\_",
            (
                "% Number as Table 6 (after Tables 4--5, which appear later in Results)\n"
                "\\setcounter{table}{5}\n"
                "\\input{tables/table6_multiseed_direction.tex}\n"
                "\\setcounter{table}{3}\n"
            ),
        )
        # Keep markdown ## headings as \subsection for all Results parts.
        (OUT / "sections" / tex_name).write_text(tex, encoding="utf-8")

    # Conclusion only (first part of 07)
    back = (SECTIONS / "07_backmatter.md").read_text(encoding="utf-8")
    # Split at first ---
    parts = re.split(r"\n---\n", back, maxsplit=1)
    conclusion_md = parts[0]
    extra_md = parts[1] if len(parts) > 1 else ""
    conc = md_body_to_tex(conclusion_md, drop_top_h1=True)
    (OUT / "sections" / "07_conclusion.tex").write_text(conc, encoding="utf-8")
    extra = md_body_to_tex("# Back\n" + extra_md, drop_top_h1=True)
    (OUT / "sections" / "07_backmatter_extra.tex").write_text(extra, encoding="utf-8")

    # Supplementary reproducibility appendix (keep code paths here, not in main Methods)
    s1_md = (SECTIONS / "S1_reproducibility.md").read_text(encoding="utf-8")
    s1_tex = md_body_to_tex(s1_md, drop_top_h1=True)
    # Promote top-level appendix heading
    if not s1_tex.lstrip().startswith("\\section"):
        s1_tex = (
            "\\section{Reproducibility appendix}\n"
            "\\label{sec:supp-repro}\n"
            + s1_tex
        )
    else:
        s1_tex = s1_tex.replace(
            "\\subsection{Reproducibility appendix}",
            "\\section{Reproducibility appendix}\\label{sec:supp-repro}",
            1,
        )
    # S1 md uses ## for S1.x -> already subsections after drop h1
    figs1 = r"""
\input{tables/table7_architecture.tex}
\input{tables/table8_compute.tex}
\input{tables/table_s_reviewer_defense.tex}
\begin{figure*}[!t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig_s1_training_curves.png}
\caption{Fold-0 constrained training curves from \texttt{constrained\_evaluation\_report.json} (\texttt{constraint\_curves.fold\_0}). (A) Train and validation total loss. (B) Reconstruction MSE. (C) Unweighted KL. (D) Unweighted $L_{\mathrm{band}}$ and $L_{\mathrm{conn}}$ with the implemented $\alpha_c(t)$ schedule. Validation total loss remains below training total loss on this fold (minimum validation total $0.0241$ at epoch 99). Logged $\alpha_c$ is 0 through epoch 6 and $0.1$ by epoch 11. Alternate-seed curves were not stored. Logged values only; not an external-generalization claim.}
\label{fig:traincurves}
\end{figure*}
"""
    s1_anchor = "Fig.~\\ref{fig:traincurves} plots those logged values."
    if s1_anchor in s1_tex:
        s1_tex = s1_tex.replace(s1_anchor, s1_anchor + "\n" + figs1, 1)
    else:
        raise RuntimeError("Failed to inject appendix architecture/compute/curve floats")
    (OUT / "sections" / "S1_reproducibility.tex").write_text(s1_tex, encoding="utf-8")

    # Patch methods: insert Fig 1 and Table 1 floats
    meth = (OUT / "sections" / "02_methods.tex").read_text(encoding="utf-8")
    fig1 = r"""
\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_architecture.png}
\caption{Architecture of the pharmacodynamically constrained conditional variational autoencoder (CVAE) digital twin. Resting-state EEG features (2185-D) and ChemBERTa drug embeddings are encoded and fused with a disease label, then mapped by a CVAE whose training includes pharmacodynamic band and connectivity constraints. Training cohort: $N = 66$ (AD 37 / HC 29).}
\label{fig:architecture}
\end{figure*}
"""
    tab1 = "\n\\input{tables/table1_datasets.tex}\n"
    # after Fig 1 sentence
    meth = meth.replace(
        "is illustrated in Fig.~\\ref{fig:architecture}.",
        "is illustrated in Fig.~\\ref{fig:architecture}.\n" + fig1,
        1,
    )
    meth = meth.replace(
        "summarized in Algorithm~\\ref{alg:train}.",
        "summarized in Algorithm~\\ref{alg:train}.\n\\input{tables/algorithm1_training.tex}\n",
        1,
    )
    meth = meth.replace(
        "are summarized in Table~\\ref{tab:datasets}.",
        "are summarized in Table~\\ref{tab:datasets}.\n" + tab1,
        1,
    )
    if "algorithm1_training.tex" not in meth:
        raise RuntimeError("Failed to inject Algorithm 1 into methods")
    (OUT / "sections" / "02_methods.tex").write_text(meth, encoding="utf-8")

    # Results direction: table5 early (ablation), then table2 + fig2
    r1 = (OUT / "sections" / "03_results_mechanism.tex").read_text(encoding="utf-8")
    fig2 = r"""
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig2_direction_agreement.png}
\caption{Locked fold-0 seed-42 checkpoint only: external signature concordance scored against the training constrained signature. Bars show full-sample effect-vector correlation $r$ for TUH ($r{=}0.908$, $n{=}200$), OSF ($r{=}0.851$, $n{=}92$), and P-ADIC ($r{=}0.918$, $n{=}145$). These bars are not representative of five-fold or multi-seed performance; five-fold means and fixed-fold seed sensitivity are reported in Section 3.2 and Table~6. Subject-bootstrap 95\% percentile intervals (B${=}2000$, seed 42) are 0.874 to 0.900 (TUH), 0.790 to 0.837 (OSF), and 0.871 to 0.904 (P-ADIC); those percentile intervals can sit slightly below the full-sample point. Secondary band-level signed agreement was 10/10 on each cohort. These values are checkpoint-specific signature concordance, not architecture-wide performance estimates and not independent pharmacodynamic validation.}
\label{fig:direction}
\end{figure}
"""
    tab5_anchor = (
        "Table~\\ref{tab:ablation-direction} summarizes constrained versus "
        "unconstrained direction on TUH, OSF, and P-ADIC so the locked-checkpoint "
        "control contrast is visible before the external constrained replications."
    )
    if tab5_anchor in r1:
        r1 = r1.replace(
            tab5_anchor,
            tab5_anchor + "\n\\input{tables/table5_ablation_direction.tex}\n",
            1,
        )
    else:
        # Fallback for shorter legacy phrasing
        r1 = r1.replace(
            "Table~\\ref{tab:ablation-direction} summarizes constrained versus unconstrained direction on TUH, OSF, and P-ADIC.",
            "Table~\\ref{tab:ablation-direction} summarizes constrained versus unconstrained direction on TUH, OSF, and P-ADIC.\n"
            + "\\input{tables/table5_ablation_direction.tex}\n",
            1,
        )
    fig2_anchor = (
        "Table~\\ref{tab:direction-agreement} and Fig.~\\ref{fig:direction} place "
        "the three locked seed-42 concordance values side by side so continuous \\(r\\) "
        "and 10/10 sign agreement can be compared across cohorts before the "
        "sensitivity analyses."
    )
    if fig2_anchor in r1:
        r1 = r1.replace(
            fig2_anchor,
            fig2_anchor
            + "\n\\input{tables/table2_direction_agreement.tex}\n"
            + fig2,
            1,
        )
    else:
        r1 = r1.replace(
            "illustrated in Fig.~\\ref{fig:direction}.",
            "illustrated in Fig.~\\ref{fig:direction}.\n"
            + "\\input{tables/table2_direction_agreement.tex}\n"
            + fig2,
            1,
        )
    if "\\includegraphics" not in r1 or "fig2_direction_agreement" not in r1:
        raise RuntimeError("Failed to inject Fig. 2 / Table 3 into results_mechanism")
    if "table5_ablation_direction" not in r1:
        raise RuntimeError("Failed to inject Table 2 into results_mechanism")
    (OUT / "sections" / "03_results_mechanism.tex").write_text(r1, encoding="utf-8")

    # Diagnosis: fig3, table3, fig4
    r2 = (OUT / "sections" / "04_results_diagnosis.tex").read_text(encoding="utf-8")
    fig3 = r"""
\begin{figure*}[!t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig3_diagnostic_null.png}
\caption{Forest plot of Cohen's $d$ for primary twin magnitude contrasts on TUH, OSF, P-ADIC, and CAUEEG. Approximate 95\% confidence intervals use the Hedges--Olkin standard-error formula from reported $d$ and group sizes (not bootstrap intervals stored in validation artifacts). Exact $n$, $p$, and $d$ values are listed in Table~\ref{tab:diagnostic-results}.}
\label{fig:diagnostic}
\end{figure*}
"""
    fig4 = r"""
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig4_pca.png}
\caption{PCA of 2185-D features: training cohort versus CAUEEG. Dashed circle marks the training 95\% radius; overlap fraction within that radius is 0.611.}
\label{fig:pca}
\end{figure}
"""
    diag_anchor = (
        "Fig.~\\ref{fig:diagnostic} places the corresponding effect sizes with "
        "approximate 95\\% intervals on a common axis."
    )
    if diag_anchor in r2:
        r2 = r2.replace(
            diag_anchor,
            diag_anchor
            + "\n\\input{tables/table3_diagnostic_results.tex}\n"
            + fig3,
            1,
        )
    else:
        r2 = r2.replace(
            "illustrated in Fig.~\\ref{fig:diagnostic}.",
            "illustrated in Fig.~\\ref{fig:diagnostic}.\n"
            + "\\input{tables/table3_diagnostic_results.tex}\n"
            + fig3,
            1,
        )
    r2 = r2.replace(
        "Fig.~\\ref{fig:pca}).",
        "Fig.~\\ref{fig:pca}).\n" + fig4,
        1,
    )
    if "fig3_diagnostic_null" not in r2 or "table3_diagnostic_results" not in r2:
        raise RuntimeError("Failed to inject Fig. 3 / Table 4 into results_diagnosis")
    (OUT / "sections" / "04_results_diagnosis.tex").write_text(r2, encoding="utf-8")

    # Encoding: fig5 + table4
    r3 = (OUT / "sections" / "05_results_encoding_analysis.tex").read_text(encoding="utf-8")
    fig5 = r"""
\begin{figure*}[!t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig5_encoding_analysis.png}
\caption{Encoding analysis on CAUEEG Dementia versus Normal ($n{=}727$). Amber: theta/alpha feature probe (mean-fold ROC-AUC $= 0.675$; pooled OOF $= 0.673$). Teal/navy: frozen-twin $\mu_{\mathrm{base}}$ readouts. Gray band: latent-probe permutation null 5th--95th percentiles (0.449--0.559). Best nested head OOF AUC $= 0.593$ remains below the spectral reference. A separate class-balanced logistic regression on packed 2185-D features (raw feature-space diagnostic reference, not a CVAE score) reaches mean-fold AUC $= 0.699$.}
\label{fig:encoding}
\end{figure*}
"""
    enc_anchor = (
        "chance (AUC = 0.5), so feature-level versus latent readout performance "
        "can be compared visually."
    )
    if enc_anchor in r3:
        r3 = r3.replace(
            enc_anchor,
            enc_anchor + "\n" + fig5 + "\\input{tables/table4_encoding_results.tex}\n",
            1,
        )
    else:
        r3 = r3.replace(
            "chance (AUC = 0.5).",
            "chance (AUC = 0.5).\n" + fig5 + "\\input{tables/table4_encoding_results.tex}\n",
            1,
        )
    if "fig5_encoding_analysis" not in r3 or "table4_encoding_results" not in r3:
        raise RuntimeError("Failed to inject Fig. 5 / Table 5 into encoding section")
    (OUT / "sections" / "05_results_encoding_analysis.tex").write_text(r3, encoding="utf-8")

    # Constraint-strength sweep: fig6
    r4 = (OUT / "sections" / "05b_results_constraint_strength.tex").read_text(encoding="utf-8")
    fig6 = r"""
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig6_constraint_strength.png}
\caption{Exploratory fold-0-only constraint-weight sweep on matched fold-0 twins ($\alpha_c$-sweep re-simulation; not the locked unconstrained battery in Section 3.1). This sweep is not nested cross-validation of $\alpha_c$; the paper weight $\alpha_c{=}0.1$ was not selected by nested CV. Coincidence of $\alpha_c{=}0.1$ with the continuous-$r$ peak on this curve is not evidence of principled hyperparameter tuning. Left axis: mean TUH/P-ADIC direction agreement (\%; teal) and $100\times$ mean effect-vector $r$ (navy). Right axis: CAUEEG Dementia versus Normal latent-probe ROC-AUC (amber). Direction signs reach 10/10 by $\alpha_c{=}0.05$; effect-vector $r$ peaks at $\alpha_c{=}0.1$ on this sweep; latent AUC is non-monotone. Sweep uses fold-0 twins and TUH/P-ADIC direction scores (OSF omitted).}
\label{fig:alpha}
\end{figure}
"""
    r4 = r4.replace(
        "illustrated in Fig.~\\ref{fig:alpha}.",
        "illustrated in Fig.~\\ref{fig:alpha}.\n" + fig6,
        1,
    )
    # Fallback if converter left "Fig. 6" literal
    if "\\includegraphics" not in r4 and "Fig. 6" in r4:
        r4 = r4.replace(
            "Results are illustrated in Fig. 6.",
            "Results are illustrated in Fig.~\\ref{fig:alpha}.\n" + fig6,
            1,
        )
    (OUT / "sections" / "05b_results_constraint_strength.tex").write_text(r4, encoding="utf-8")

    write_main()

    readme = """# Overleaf / Elsevier `elsarticle` package

Upload this **entire folder** to Overleaf:

1. Zip the `overleaf` directory (contents: `main.tex`, `sections/`, `figures/`, `tables/`, `references.bib`).
2. Overleaf → **New Project** → **Upload Project** → select the zip.
3. Set compiler to **pdfLaTeX** (default).
4. Click **Recompile**.

## Notes

- Document class: `elsarticle` (preprint, 12pt, 3p, times), bibliography `elsarticle-num`.
- Target journal: Computer Methods and Programs in Biomedicine (CMPB; ISSN 0169-2607).
- Graphical abstract: `figures/graphical_abstract.png` (Elsevier front matter; not Fig. N).
- Source of truth for prose remains `paper/sections/*.md`; re-run `src/paper/build_overleaf_elsarticle.py` after major edits.

## If `highlights` / `graphicalabstract` errors

Some older TeX Live installs need:
```latex
\\usepackage{framed} % rarely needed
```
Or comment out the `highlights` / `graphicalabstract` environments and paste those assets in the Elsevier submission form separately.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    final = PAPER / "overleaf"
    try:
        if final.exists():
            shutil.rmtree(final)
        OUT.rename(final)
        OUT = final
    except OSError:
        # Destination locked (Windows/IDE): copy over files instead of rename.
        final.mkdir(parents=True, exist_ok=True)
        for src in OUT.rglob("*"):
            if src.is_file():
                dst = final / src.relative_to(OUT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        print(f"Note: could not replace {final}; copied files from staging.")
        OUT = final
    print(f"Wrote Overleaf package to {OUT}")
    print("Files:", sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())[:40])


if __name__ == "__main__":
    main()
