import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "NLP" / "NLP03" / "NLP03_project_chatbot.ipynb"

with NB_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        fixed = textwrap.dedent(source).strip()
        cell["source"] = [line + "\n" for line in fixed.splitlines()]

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"fixed code indentation: {NB_PATH}")
