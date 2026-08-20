"""Convert \\code{raw_id} back to \\texttt{escaped\\_id}."""
from pathlib import Path
import re

files = list(Path("paper/latex/sections").glob("*.tex")) + list(
    Path("paper/latex/tables").glob("*.tex")
)

pat = re.compile(r"\\code\{([^{}]+)\}")


def escape_tt(raw: str) -> str:
    return raw.replace("\\", "\\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


for p in files:
    text = p.read_text(encoding="utf-8")

    def repl(m: re.Match) -> str:
        return r"\texttt{" + escape_tt(m.group(1)) + "}"

    new = pat.sub(repl, text)
    if new != text:
        p.write_text(new, encoding="utf-8")
        print("updated", p)
    else:
        print("unchanged", p.name)
