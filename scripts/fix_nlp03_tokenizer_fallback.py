import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "NLP" / "NLP03" / "NLP03_project_chatbot.ipynb"


TOKENIZER_CODE = """try:
    from konlpy.tag import Mecab
    mecab = Mecab()
    tokenize = mecab.morphs
    tokenizer_name = "Mecab"
except Exception as e:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()

    def tokenize(sentence):
        return [token.form for token in kiwi.tokenize(sentence)]

    tokenizer_name = "Kiwi"
    print("Mecab을 사용할 수 없어 Kiwi로 대체합니다.")
    print("Mecab 오류:", e)

sample = "오늘 너무 피곤하다."
print("사용 토크나이저:", tokenizer_name)
print("원본:", sample)
print("토큰화:", tokenize(preprocess_sentence(sample)))
"""


BUILD_CALL_CODE = """que_corpus, ans_corpus = build_corpus(
    questions,
    answers,
    tokenize,
    max_len=MAX_TOKEN_LEN
)

print("질문 말뭉치 개수:", len(que_corpus))
print("답변 말뭉치 개수:", len(ans_corpus))
print("질문 토큰 예시:", que_corpus[0])
print("답변 토큰 예시:", ans_corpus[0])
"""


with NB_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and "여기서는 LMS 조건에 맞춰 KoNLPy의 `Mecab`" in source:
        source = source.replace(
            "여기서는 LMS 조건에 맞춰 KoNLPy의 `Mecab`을 사용합니다.  \n"
            "`mecab.morphs(\"문장\")`을 실행하면 문장이 형태소 단위의 리스트로 나뉩니다.",
            "원래 LMS 조건은 KoNLPy의 `Mecab`을 사용하는 것입니다.  \n"
            "다만 Windows 로컬 환경에서는 MeCab 설치가 자주 막히므로, 이 노트북은 `Mecab`을 먼저 시도하고 실패하면 `Kiwi` 토크나이저로 대체합니다."
        )
        cell["source"] = [line + "\n" for line in source.splitlines()]

    if cell.get("cell_type") != "code":
        continue

    if "from konlpy.tag import Mecab" in source and "sample =" in source:
        cell["source"] = [line + "\n" for line in TOKENIZER_CODE.strip().splitlines()]
    elif "mecab.morphs" in source and "build_corpus" in source:
        cell["source"] = [line + "\n" for line in BUILD_CALL_CODE.strip().splitlines()]

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"updated tokenizer fallback: {NB_PATH}")
