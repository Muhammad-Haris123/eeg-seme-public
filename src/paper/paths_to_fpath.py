"""Convert path-like \\texttt{...} to breakable \\fpath{...}."""
from pathlib import Path
import re

files = [
    Path("paper/latex/sections/02_methods.tex"),
    Path("paper/latex/sections/07_backmatter_extra.tex"),
]
pat = re.compile(r"\\texttt\{([^{}]+)\}")


def convert(inner: str) -> str | None:
    raw = inner.replace(r"\_", "_").replace(r"\%", "%")
    if any(c in inner for c in "'[]*=:()"):
        return None
    if "/" in raw and raw.startswith(("src/", "models/", "data/", "paper/")):
        return r"\fpath{" + raw + "}"
    return None


for p in files:
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8")

    def repl(m: re.Match) -> str:
        out = convert(m.group(1))
        return out if out is not None else m.group(0)

    new = pat.sub(repl, text)
    p.write_text(new, encoding="utf-8")
    print(p, "fpath=", new.count(r"\fpath{"))
