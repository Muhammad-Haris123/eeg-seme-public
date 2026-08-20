"""Fix LaTeX abstract + discussion without PowerShell $-expansion."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

abstract = (
    "Chemistry-conditioned digital twins for drug-response simulation must show that a "
    "constrained mechanism transfers externally and must separately justify any diagnostic "
    "use of twin outputs. Most EEG twin and generative models emphasize one check, not both. "
    "We trained a pharmacodynamically constrained conditional variational autoencoder (CVAE) "
    "on 2185-dimensional resting-state EEG features with chemical Bidirectional Encoder "
    "Representations from Transformers (ChemBERTa) embeddings for Donepezil and Memantine. "
    "Band-level drug-response direction agreed with the training signature at 10/10 on three "
    "external cohorts: Temple University Hospital (TUH) EEG ($n = 200$; effect-magnitude "
    "correlation $r = 0.908$), an Open Science Framework (OSF) AD/healthy control (HC) set "
    "($n = 92$; $r = 0.851$), and p-adic quantum potential EEG (P-ADIC) AD/HC recordings "
    "($n = 145$; $r = 0.918$). Latent drug-response magnitude did not diagnose reliably "
    "across four escalating external tests. TUH abnormal versus normal was null "
    "($p = 0.272$, $d = -0.13$). OSF AD versus HC passed with caveats "
    "($p = 0.000278$, $d = 0.44$, HC $n = 12$). P-ADIC AD versus HC was null "
    "($p = 0.968$, $d = -0.07$). CAUEEG Dementia versus Normal ($n = 727$) was null "
    "($p = 0.516$, $d = 0.00$) despite calibrated scale and good PCA overlap. On CAUEEG, "
    "a logistic probe on baseline latent means reached ROC-AUC $= 0.579$ (permutation "
    "$p = 0.010$), below a theta/alpha feature probe (ROC-AUC $= 0.675$), and nested heads "
    "peaked at out-of-fold ROC-AUC $= 0.593$. The constrained twin transfers mechanistic "
    "direction while compressing diagnostic information at encoding, quantifying that "
    "measurement trade-off for this architecture."
)

main_path = ROOT / "paper" / "latex" / "main.tex"
main = main_path.read_text(encoding="utf-8")
main = re.sub(
    r"(\\begin\{abstract\}\n).*?(\\end\{abstract\})",
    r"\1" + abstract + "\n" + r"\2",
    main,
    count=1,
    flags=re.S,
)
# clean any broken \1 prefix if previous regex failed oddly
main = main.replace(r"\1Chemistry-conditioned", "Chemistry-conditioned")
main_path.write_text(main, encoding="utf-8")
print("abstract fixed")

# rebuild discussion from md with safe python
md = (ROOT / "paper" / "sections" / "06_discussion.md").read_text(encoding="utf-8")
lines = md.strip().splitlines()
if lines and lines[0].startswith("#"):
    lines = lines[1:]
text = "\n".join(lines).strip() + "\n"


def cite(m: re.Match) -> str:
    keys = [k.strip() for k in m.group(1).split(",")]
    return "\\citep{" + ",".join(keys) + "}"


text = re.sub(r"\[([A-Za-z0-9_, ]+)\]", cite, text)
text = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", text)
text = text.replace("%", r"\%")
text = text.replace(
    "(p = 0.516, d = -0.00)",
    "($p = 0.516$, $d = -0.00$)",
)
(ROOT / "paper" / "latex" / "sections" / "06_discussion.tex").write_text(text, encoding="utf-8")
print("discussion fixed")
