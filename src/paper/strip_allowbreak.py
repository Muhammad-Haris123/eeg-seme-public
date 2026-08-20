"""Remove \\allowbreak{} insertions from latex section/table sources."""
from pathlib import Path

files = list(Path("paper/latex/sections").glob("*.tex")) + list(
    Path("paper/latex/tables").glob("*.tex")
)
for p in files:
    text = p.read_text(encoding="utf-8")
    new = text.replace(r"\allowbreak{}", "")
    # also bare \allowbreak with space variants
    new = new.replace(r"\allowbreak ", "")
    new = new.replace(r"\allowbreak", "")
    if new != text:
        p.write_text(new, encoding="utf-8")
        print("cleaned", p)
    else:
        print("ok", p.name)
