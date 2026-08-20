"""Convert long \\texttt{snake_case} identifiers to breakable \\code{...}."""
from pathlib import Path
import re

files = list(Path("paper/latex/sections").glob("*.tex")) + list(
    Path("paper/latex/tables").glob("*.tex")
)

KEEP = {
    ".set",
    "cuda",
    "FAIL",
    "PASS",
    "PASS_WITH_CAVEATS",
    "Eyes_closed",
    "eeg_twin",
}


def to_raw(inner: str) -> str:
    return inner.replace(r"\_", "_").replace(r"\%", "%").replace(r"\&", "&")


def should_code(inner: str) -> bool:
    if any(c in inner for c in "{}"):
        return False
    raw = to_raw(inner)
    if raw in KEEP:
        return False
    # Keep config/kwargs literals with quotes, brackets, equals, etc.
    if any(c in inner for c in "'[]*=:()"):
        return False
    if "_" in raw or "/" in raw:
        return True
    if raw.endswith((".py", ".json", ".md", ".pt", ".npy")):
        return True
    return False


pat = re.compile(r"\\texttt\{([^{}]+)\}")

for p in files:
    text = p.read_text(encoding="utf-8")
    text = text.replace(r"\code{.set}", r"\texttt{.set}")

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        if should_code(inner):
            return r"\code{" + to_raw(inner) + "}"
        return m.group(0)

    new = pat.sub(repl, text)
    if new != text:
        p.write_text(new, encoding="utf-8")
        print("updated", p)
    else:
        print("unchanged", p.name)

print("--- remaining long texttt with underscore ---")
for p in files:
    t = p.read_text(encoding="utf-8")
    for m in pat.finditer(t):
        inner = m.group(1)
        if r"\_" in inner and len(inner) > 12:
            print(p.name, ":", inner[:90])
