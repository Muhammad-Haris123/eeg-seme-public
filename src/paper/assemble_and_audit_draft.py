"""
Assemble paper/manuscript_draft.md and run consistency audit (report only).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = ROOT / "paper" / "sections"
OUT = ROOT / "paper" / "manuscript_draft.md"
LEDGER = ROOT / "paper" / "citation_ledger.md"
ABBREV = ROOT / "paper" / "abbreviations.md"

ORDER = [
    "00_frontmatter.md",
    "01_introduction.md",
    "02_methods.md",
    "03_results_mechanism.md",
    "04_results_diagnosis.md",
    "05_results_encoding_analysis.md",
    "05b_results_constraint_strength.md",
    "06_discussion.md",
    "07_backmatter.md",
]

BANNED_PHRASES = [
    "novel",
    "cutting-edge",
    "unprecedented",
    "powerful",
    "shed light on",
    "pave the way",
    "in recent years",
    "it is worth noting that",
    "delve into",
    "we believe",
]


def assemble() -> str:
    parts = []
    parts.append("# Manuscript draft (Computers in Biology and Medicine)\n")
    parts.append(
        "_Assembled from `paper/sections/` for review. Figure/table assets live under "
        "`paper/figures/` and `paper/tables/`. Do not treat this file as camera-ready LaTeX._\n"
    )
    for name in ORDER:
        text = (SECTIONS / name).read_text(encoding="utf-8").strip()
        if name == "00_frontmatter.md":
            text = text.replace("# Front matter (Computers in Biology and Medicine)\n\n", "")
            text = (
                "# Pharmacodynamic Direction Transfers, Diagnostic Magnitude Does Not: "
                "Quantifying an Encoding Trade-off in a Constrained EEG-Drug CVAE Twin "
                "for Alzheimer's Disease\n\n"
                + re.sub(
                    r"## Title options.*?(?=## Highlights)",
                    "",
                    text,
                    count=1,
                    flags=re.S,
                )
            )
            text = re.sub(
                r"### Longer draft \(unused; kept for revision only\).*?(?=## Keywords)",
                "",
                text,
                count=1,
                flags=re.S,
            )
        parts.append(text)
        parts.append("\n\n---\n\n")
    return "\n".join(parts).rstrip() + "\n"


def first_cite_order(text: str) -> list[tuple[str, int]]:
    hits = []
    for m in re.finditer(r"\b(?:Fig\.|Figure|Table)\s*(\d+)\b", text):
        kind = "Fig" if m.group(0).lower().startswith("fig") else "Table"
        hits.append((f"{kind}. {m.group(1)}", m.start()))
    seen: dict[str, int] = {}
    for label, pos in hits:
        if label not in seen:
            seen[label] = pos
    return sorted(seen.items(), key=lambda x: x[1])


def extract_stat_numbers(text: str) -> set[str]:
    """Heuristic: decimals, p=/d=/r=/AUC patterns, 10/10, n= forms."""
    nums: set[str] = set()
    for m in re.finditer(
        r"(?:p\s*=\s*|d\s*=\s*|r\s*=\s*|AUC\s*=\s*|ROC-AUC\s*=\s*)(-?\d+\.\d+)",
        text,
        flags=re.I,
    ):
        nums.add(m.group(1))
    for m in re.finditer(r"\b(-?\d+\.\d{2,})\b", text):
        nums.add(m.group(1))
    for m in re.finditer(r"\b(\d+/\d+)\b", text):
        nums.add(m.group(1))
    for m in re.finditer(r"\bn\s*=\s*(\d+)\b", text, flags=re.I):
        nums.add(f"n={m.group(1)}")
    return nums


def main() -> None:
    draft = assemble()
    OUT.write_text(draft, encoding="utf-8")
    print(f"Wrote {OUT} ({len(draft)} chars, {len(draft.splitlines())} lines)")

    order = first_cite_order(draft)
    print("\n=== First-citation order (Task 2) ===")
    for i, (label, _) in enumerate(order, 1):
        print(f"  {i}. {label}")

    # Expected sequential: Fig 1..Fmax and Table 1..Tmax interleaved by first cite
    fig_seq = [int(x.split()[1]) for x, _ in order if x.startswith("Fig")]
    tab_seq = [int(x.split()[1]) for x, _ in order if x.startswith("Table")]
    print(f"  Figure sequence by first cite: {fig_seq}")
    print(f"  Table sequence by first cite: {tab_seq}")

    issues: list[str] = []

    # --- Em dashes / double hyphens ---
    for i, line in enumerate(draft.splitlines(), 1):
        if "\u2014" in line:
            issues.append(f"[EM_DASH] line {i}: {line.strip()[:140]}")
        if "\u2013" in line:
            # allow numeric ranges like 0.851–0.918 if en dash used
            issues.append(f"[EN_DASH] line {i}: {line.strip()[:140]}")
        # prose double-hyphen (not markdown HR ---)
        if re.search(r"[A-Za-z0-9)]\s*--\s*[A-Za-z(]", line):
            issues.append(f"[DOUBLE_HYPHEN_AS_EM] line {i}: {line.strip()[:140]}")

    # --- Numbering gaps ---
    if fig_seq and fig_seq != list(range(1, max(fig_seq) + 1)):
        issues.append(f"[FIG_NUMBERING] first-cite order {fig_seq} is not 1..N sequential")
    if tab_seq and tab_seq != list(range(1, max(tab_seq) + 1)):
        issues.append(f"[TABLE_NUMBERING] first-cite order {tab_seq} is not 1..N sequential")

    if "FIGURE:" in draft or "[FIGURE" in draft:
        issues.append("[PLACEHOLDER] unresolved FIGURE tag in draft")

    # --- Figure/table appears without prior citation ---
    # In this markdown draft, assets are not embedded; check for image embeds / caption blocks
    embed_re = re.compile(
        r"!\[.*?\]\((.*?fig|.*?table).*?\)|"
        r"^#{1,3}\s*(Fig\.|Figure|Table)\s*(\d+)",
        re.I | re.M,
    )
    for m in embed_re.finditer(draft):
        pos = m.start()
        # find which number
        num_m = re.search(r"(Fig\.|Figure|Table)\s*(\d+)", m.group(0), re.I)
        if not num_m:
            continue
        kind = "Fig" if num_m.group(1).lower().startswith("fig") else "Table"
        label = f"{kind}. {num_m.group(2)}"
        # prior textual citation of same label
        prior = draft[:pos]
        if not re.search(rf"\b{re.escape(kind)}\.?\s*{num_m.group(2)}\b", prior, re.I):
            issues.append(
                f"[UNCITED_BEFORE_APPEARANCE] {label} appears/embeds at char {pos} "
                f"without a prior textual citation"
            )
    # Also: figure files exist but if first prose cite is missing entirely
    for n in range(1, 6):
        if f"Fig. {n}" not in draft and f"Figure {n}" not in draft:
            issues.append(f"[MISSING_CITE] Fig. {n} never cited in prose")
    for n in range(1, 5):
        if f"Table {n}" not in draft:
            issues.append(f"[MISSING_CITE] Table {n} never cited in prose")

    # --- Banned language ---
    for phrase in BANNED_PHRASES:
        for i, line in enumerate(draft.splitlines(), 1):
            if phrase in line.lower():
                issues.append(f"[BANNED_LANGUAGE] '{phrase}' line {i}: {line.strip()[:140]}")
    for i, line in enumerate(draft.splitlines(), 1):
        # 'robust' filler adjective (not 'robustly' in locked storyline sense if used loosely)
        if re.search(r"\brobust(?:ly)?\b", line, re.I):
            issues.append(f"[BANNED_OR_FILLER] 'robust/robustly' line {i}: {line.strip()[:140]}")

    # --- Abbreviations: first use before definition ---
    # Manual high-value checks (automated expansion matching is noisy for short tokens)
    abbrev_checks = [
        (
            "TUH",
            r"Temple University Hospital\s*\(TUH\)",
            "TUH used before 'Temple University Hospital (TUH)' definition.",
        ),
        (
            "OSF",
            r"Open Science Framework\s*\(OSF\)",
            "OSF used before 'Open Science Framework (OSF)' definition.",
        ),
        (
            "P-ADIC",
            r"p-adic quantum potential EEG\s*\(P-ADIC\)",
            "P-ADIC used before 'p-adic quantum potential EEG (P-ADIC)' definition.",
        ),
        (
            "OOF",
            r"out-of-fold\s*\(OOF\)",
            "OOF used before 'out-of-fold (OOF)' definition.",
        ),
        (
            "ChemBERTa",
            r"chemical Bidirectional Encoder Representations from Transformers\s*\(ChemBERTa\)",
            "ChemBERTa used before full expansion with (ChemBERTa).",
        ),
    ]
    # Assembled draft strips unused title options; start checks at Highlights
    hl = draft.find("## Highlights")
    body = draft[hl:] if hl >= 0 else draft
    for ab, def_pat, msg in abbrev_checks:
        first_ab = re.search(rf"\b{re.escape(ab)}\b", body)
        first_def = re.search(def_pat, body, re.I)
        if first_ab and (not first_def or first_def.start() > first_ab.start()):
            # Allow Highlights CAUEEG-only path; report position
            snippet = body[max(0, first_ab.start() - 30) : first_ab.start() + len(ab) + 30].replace("\n", " ")
            issues.append(f"[ABBREV_BEFORE_DEF] {ab}: {msg} Near: ...{snippet}...")

    # --- Section open/close transitions ---
    transition_issues = [
        (
            "# 4. Discussion",
            r"Section 3\.3|encoding analysis|previous section|Results",
            "[TRANSITION_OPEN] Discussion opens by restating the Introduction question "
            "rather than an explicit one-sentence link to what §3.3 (encoding) established.",
        ),
        (
            "# 5. Conclusion",
            r"Discussion|previous section|Section 4",
            "[TRANSITION_OPEN] Conclusion opens with 'This study establishes' and does not "
            "explicitly link back to the Discussion's resolved tension in the opening sentence.",
        ),
        (
            "# 2. Methods",
            r"Introduction|framed|previous",
            None,  # expect OK
        ),
    ]
    # Check Discussion opening paragraph specifically
    disc = re.search(r"# 4\. Discussion\n\n(.+?)(?:\n\n|\Z)", draft, re.S)
    if disc:
        opening = disc.group(1).split("\n")[0]
        if not re.search(r"Section 3\.3|§3\.3|encoding", opening, re.I):
            issues.append(
                "[TRANSITION_OPEN] Discussion first sentence does not link to §3.3 encoding. "
                f"Opening: {opening[:160]}..."
            )
    conc = re.search(r"# 5\. Conclusion\n\n(.+?)(?:\n\n|\Z)", draft, re.S)
    if conc:
        opening = conc.group(1).split("\n")[0]
        if not re.search(r"Discussion|Section 4", opening, re.I):
            issues.append(
                "[TRANSITION_OPEN] Conclusion first sentence does not reference Discussion/"
                f"Section 4. Opening: {opening[:160]}..."
            )

    # Methods subsections open/close (spot-check weak handoffs)
    for header in ["## 2.2", "## 2.3", "## 2.4", "## 2.5", "## 2.6"]:
        m = re.search(rf"{header}[^\n]*\n\n(.+)", draft)
        if m:
            first = m.group(1).split("\n")[0]
            # weak heuristic: should reference prior subsection topic
            if header == "## 2.2" and "architecture" not in first.lower() and "Building" not in first:
                if "constraint" in first.lower() or "pharmac" in first.lower():
                    pass  # soft
            if header == "## 2.6" and "2.5" not in first and "Endpoint" not in first and "map" not in first.lower():
                issues.append(
                    f"[TRANSITION_OPEN] {header} opening may not explicitly link to §2.5: "
                    f"{first[:140]}..."
                )

    # Intro close -> Methods: present
    if "Methods section next" not in draft and "Section 2 describes" not in draft:
        issues.append("[TRANSITION_CLOSE] Introduction may lack handoff to Methods")

    # --- Citation ledger coverage ---
    ledger = LEDGER.read_text(encoding="utf-8")
    draft_nums = extract_stat_numbers(draft)
    # Filter to scientific-looking values; skip years and DOIs handled separately
    skip_prefixes = ()
    missing_ledger: list[str] = []
    for num in sorted(draft_nums, key=lambda s: (len(s), s)):
        if num.startswith("n="):
            # n= values are numerous; only flag if neither n=N nor bare N appears in ledger context
            bare = num.split("=")[1]
            if bare not in ledger and num not in ledger:
                # only flag distinctive large/clinical n's
                if bare in {"200", "92", "145", "727", "131", "69", "80", "12", "49", "96", "436", "395", "291", "214", "1122"}:
                    if bare not in ledger:
                        missing_ledger.append(num)
            continue
        # require exact string in ledger
        if num not in ledger:
            # allow slight variants
            if num.replace("-", "") in ledger.replace("-", ""):
                continue
            missing_ledger.append(num)

    # Deduplicate and report notable missing
    for num in missing_ledger:
        issues.append(
            f"[LEDGER_GAP] number '{num}' appears in assembled draft but has no exact "
            f"string match in citation_ledger.md"
        )

    # Spot-check known secondary stats often omitted from ledger detail
    secondary_samples = [
        ("0.877", "CAUEEG Dementia vs MCI p"),
        ("0.363", "CAUEEG MCI vs Normal p"),
        ("0.624", "CAUEEG AD-tagged Dementia vs Normal p"),
        ("0.07", "secondary Cohen's d values"),
        ("0.411", "CAUEEG disease=0 ablation p"),
        ("0.776", "CAUEEG disease=1 ablation p"),
        ("0.527", "P-ADIC disease=0 ablation p"),
        ("0.702", "P-ADIC disease=1 ablation p"),
        ("0.64", "OSF disease0 d"),
        ("0.34", "OSF disease1 d"),
        ("0.005", "best head perm p"),
        ("0.543", "MLP OOF AUC"),
        ("0.556", "HGB OOF AUC"),
        ("0.587", "mu+logvar OOF AUC"),
        ("0.449", "null band p5"),
        ("0.559", "null band p95"),
        ("0.096", "latent vs theta/alpha gap"),
    ]
    for num, label in secondary_samples:
        if num in draft and num not in ledger:
            # already added by extract loop possibly; ensure labeled once
            pass

    # VERIFY flags
    for i, line in enumerate(draft.splitlines(), 1):
        if "[VERIFY" in line:
            issues.append(f"[VERIFY_FLAG] line {i}: {line.strip()[:160]}")

    # Em dash in citation_ledger itself (bookkeeping, note separately)
    if "\u2014" in ledger:
        issues.append(
            "[NOTE] citation_ledger.md itself contains an em dash (bookkeeping file, "
            "not manuscript body)"
        )

    print("\n=== CONSISTENCY ISSUES (report only; not auto-fixed) ===\n")
    if not issues:
        print("  (none flagged)")
    else:
        for j, iss in enumerate(issues, 1):
            print(f"  {j}. {iss}")
    print(f"\nTotal issues flagged: {len(issues)}")
    print("\nStopped after report (per Task 3). No auto-fixes applied beyond Tasks 1–2.")


if __name__ == "__main__":
    main()
