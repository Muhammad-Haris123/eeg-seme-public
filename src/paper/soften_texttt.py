"""Insert \\allowbreak after \\_ . / inside long \\texttt{...} spans."""
from pathlib import Path
import re

files = list(Path("paper/latex/sections").glob("*.tex")) + list(
    Path("paper/latex/tables").glob("*.tex")
)
pat = re.compile(r"\\texttt\{([^{}]+)\}")


def soften(inner: str) -> str:
    raw = inner.replace(r"\_", "_")
    if len(raw) < 22:
        return inner
    if r"\allowbreak" in inner:
        return inner
    fixed = inner.replace(r"\_", r"\_\allowbreak{}")
    fixed = fixed.replace(".", r".\allowbreak{}")
    fixed = fixed.replace("/", r"/\allowbreak{}")
    return fixed


for p in files:
    text = p.read_text(encoding="utf-8")

    def repl(m: re.Match) -> str:
        return r"\texttt{" + soften(m.group(1)) + "}"

    new = pat.sub(repl, text)
    if new != text:
        p.write_text(new, encoding="utf-8")
        print("updated", p)
    else:
        print("unchanged", p.name)
