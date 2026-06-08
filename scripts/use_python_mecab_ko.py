import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "NLP" / "NLP03" / "NLP03_project_chatbot.ipynb"


TOKENIZER_CODE = """try:
    # Windows 로컬에서는 KoNLPy Mecab보다 python-mecab-ko가 설치와 실행이 안정적입니다.
    from mecab import MeCab
    mecab = MeCab()
    tokenize = mecab.morphs
    tokenizer_name = "python-mecab-ko"
except Exception as e:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()

    def tokenize(sentence):
        return [token.form for token in kiwi.tokenize(sentence)]

    tokenizer_name = "Kiwi"
    print("MeCab을 사용할 수 없어 Kiwi로 대체합니다.")
    print("MeCab 오류:", e)

sample = "오늘 너무 피곤하다."
print("사용 토크나이저:", tokenizer_name)
print("원본:", sample)
print("토큰화:", tokenize(preprocess_sentence(sample)))
"""


with NB_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    source = "".join(cell.get("source", []))

    if cell.get("cell_type") == "markdown" and "Windows 로컬 환경에서는" in source and "Kiwi" in source:
        source = source.replace(
            "다만 Windows 로컬 환경에서는 MeCab 설치가 자주 막히므로, 이 노트북은 `Mecab`을 먼저 시도하고 실패하면 `Kiwi` 토크나이저로 대체합니다.",
            "다만 Windows 로컬 환경에서는 KoNLPy의 `Mecab`이 자주 막히므로, 이 노트북은 `python-mecab-ko`의 `MeCab`을 사용합니다. 이것도 MeCab 기반 형태소 분석기입니다."
        )
        cell["source"] = [line + "\n" for line in source.splitlines()]

    if cell.get("cell_type") == "code" and "tokenizer_name" in source and "sample =" in source:
        cell["source"] = [line + "\n" for line in TOKENIZER_CODE.strip().splitlines()]

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"updated to python-mecab-ko: {NB_PATH}")
