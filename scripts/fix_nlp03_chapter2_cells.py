import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "NLP" / "NLP03" / "NLP03_project_chatbot.ipynb"


PREPROCESS_CODE = """import re

def preprocess_sentence(sentence):
    sentence = str(sentence).lower()
    sentence = re.sub(r"[^가-힣a-z0-9?.!,]+", " ", sentence)
    sentence = re.sub(r"\\s+", " ", sentence)
    sentence = sentence.strip()
    return sentence

print("슝=3")
"""


CHECK_CODE = """for i in range(5):
    print(f"[{i}] 질문 원본:", questions[i])
    print(f"[{i}] 질문 정제:", preprocess_sentence(questions[i]))
    print(f"[{i}] 답변 원본:", answers[i])
    print(f"[{i}] 답변 정제:", preprocess_sentence(answers[i]))
    print("-" * 60)
"""


with NB_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    if "def preprocess_sentence(sentence)" in source:
        cell["source"] = [line + "\n" for line in PREPROCESS_CODE.strip().splitlines()]
    elif "질문 원본" in source and "답변 정제" in source:
        cell["source"] = [line + "\n" for line in CHECK_CODE.strip().splitlines()]

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"fixed chapter 2 cells: {NB_PATH}")
