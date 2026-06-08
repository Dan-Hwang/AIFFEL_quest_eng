import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "NLP" / "NLP03" / "NLP03_project_chatbot.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


with NB_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

all_text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
if "## 5. 데이터 토큰화" in all_text:
    print("chapter 3 already exists")
    raise SystemExit

chapter_cells = [
    md(
        """
        ## 5. 데이터 토큰화

        이제 문장을 모델이 다룰 수 있는 작은 단위로 나눕니다.

        여기서는 LMS 조건에 맞춰 KoNLPy의 `Mecab`을 사용합니다.  
        `mecab.morphs("문장")`을 실행하면 문장이 형태소 단위의 리스트로 나뉩니다.

        예를 들어 `"오늘 너무 피곤하다"`는 대략 `["오늘", "너무", "피곤", "하다"]`처럼 쪼개집니다.
        """
    ),
    code(
        """from konlpy.tag import Mecab

mecab = Mecab()

sample = "오늘 너무 피곤하다."
print("원본:", sample)
print("토큰화:", mecab.morphs(preprocess_sentence(sample)))
"""
    ),
    md(
        """
        ### `build_corpus()` 함수 만들기

        이 함수는 질문 문장과 답변 문장을 받아서 각각 토큰화된 말뭉치로 바꿉니다.

        흐름은 이렇게 잡았습니다.

        1. 질문과 답변을 하나씩 짝지어 꺼냅니다.
        2. 앞에서 만든 `preprocess_sentence()`로 정제합니다.
        3. 전달받은 토크나이저 함수로 토큰화합니다.
        4. 토큰 수가 너무 긴 문장은 제외합니다.
        5. 중복 문장은 제외합니다.

        여기서 중복 검사는 LMS 안내에 맞춰 질문은 질문대로, 답변은 답변대로 따로 검사합니다.
        """
    ),
    code(
        """from tqdm.notebook import tqdm

MAX_TOKEN_LEN = 40

def build_corpus(src_sentences, tgt_sentences, tokenizer, max_len=MAX_TOKEN_LEN):
    src_corpus = []
    tgt_corpus = []
    src_seen = set()
    tgt_seen = set()

    for src, tgt in tqdm(zip(src_sentences, tgt_sentences), total=len(src_sentences)):
        src = preprocess_sentence(src)
        tgt = preprocess_sentence(tgt)

        src_tokens = tokenizer(src)
        tgt_tokens = tokenizer(tgt)

        if len(src_tokens) == 0 or len(tgt_tokens) == 0:
            continue

        if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
            continue

        src_key = " ".join(src_tokens)
        tgt_key = " ".join(tgt_tokens)

        if src_key in src_seen or tgt_key in tgt_seen:
            continue

        src_seen.add(src_key)
        tgt_seen.add(tgt_key)

        src_corpus.append(src_tokens)
        tgt_corpus.append(tgt_tokens)

    return src_corpus, tgt_corpus

print("슝=3")
"""
    ),
    md(
        """
        ### 질문/답변 데이터 토큰화

        이제 전체 `questions`, `answers` 데이터에 `build_corpus()`를 적용합니다.

        결과인 `que_corpus`, `ans_corpus`는 문자열 문장 리스트가 아니라, **토큰 리스트들의 리스트**입니다.
        """
    ),
    code(
        """que_corpus, ans_corpus = build_corpus(
    questions,
    answers,
    mecab.morphs,
    max_len=MAX_TOKEN_LEN
)

print("질문 말뭉치 개수:", len(que_corpus))
print("답변 말뭉치 개수:", len(ans_corpus))
print("질문 토큰 예시:", que_corpus[0])
print("답변 토큰 예시:", ans_corpus[0])
"""
    ),
    md(
        """
        ### 3장 확인

        여기까지 확인할 내용입니다.

        - `Mecab()`이 정상 실행되는지
        - `mecab.morphs()`가 문장을 토큰 리스트로 바꾸는지
        - `que_corpus`, `ans_corpus`의 길이가 같은지
        - 예시 출력이 토큰 리스트 형태인지

        다음 장에서는 이 토큰 리스트에 Lexical Substitution을 적용해서 데이터를 늘립니다.
        """
    ),
]

nb["cells"].extend(chapter_cells)

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"added chapter 3: {NB_PATH}")
