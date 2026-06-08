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

for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        source = source.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")
        cell["source"] = [line + "\n" for line in source.splitlines()]

all_text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
if "## 6. Augmentation" in all_text:
    print("chapter 4 already exists; only tqdm import was updated")
    with NB_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    raise SystemExit

chapter_cells = [
    md(
        """
        ## 6. Augmentation

        이번 단계에서는 데이터 수를 늘립니다.

        방법은 **Lexical Substitution**입니다.  
        문장 안의 단어 하나를 고르고, Word2Vec 임베딩에서 의미가 비슷한 단어로 바꿔서 새로운 문장을 만드는 방식입니다.

        LMS에서는 `ko.bin` 사전학습 임베딩을 사용하라고 안내합니다.  
        이 노트북은 `data/ko.bin` 파일이 있으면 그 파일을 사용하고, 없으면 현재 챗봇 말뭉치로 작은 Word2Vec을 임시 학습해서 진행합니다.

        제출용으로 더 좋게 만들고 싶다면 나중에 `data/ko.bin`을 받아서 같은 셀을 다시 실행하면 됩니다.
        """
    ),
    code(
        """import random
from gensim.models import Word2Vec, KeyedVectors

random.seed(42)

KO_BIN_PATH = os.path.join(DATA_DIR, "ko.bin")

if os.path.exists(KO_BIN_PATH):
    word2vec = KeyedVectors.load_word2vec_format(KO_BIN_PATH, binary=True)
    embedding_source = "data/ko.bin"
else:
    word2vec_model = Word2Vec(
        sentences=que_corpus + ans_corpus,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=10,
        seed=42
    )
    word2vec = word2vec_model.wv
    embedding_source = "temporary Word2Vec trained from chatbot corpus"

print("Embedding source:", embedding_source)
print("Vocabulary size:", len(word2vec.key_to_index))
"""
    ),
    md(
        """
        ### `lexical_sub()` 함수

        아래 함수는 토큰 리스트 하나를 받아서, 바꿀 수 있는 단어 하나를 비슷한 단어로 교체합니다.

        단어를 반드시 바꿀 수 있는 것은 아닙니다.  
        Word2Vec 사전에 없는 단어이거나, 비슷한 단어 후보가 없으면 원문을 그대로 반환합니다.
        """
    ),
    code(
        """def lexical_sub(tokens, word2vec, topn=10):
    if len(tokens) == 0:
        return tokens

    candidates = [token for token in tokens if token in word2vec.key_to_index]
    if len(candidates) == 0:
        return tokens[:]

    selected = random.choice(candidates)

    try:
        similar_words = word2vec.similar_by_word(selected, topn=topn)
    except KeyError:
        return tokens[:]

    replacements = [
        word for word, score in similar_words
        if word != selected and len(word.strip()) > 0
    ]

    if len(replacements) == 0:
        return tokens[:]

    replacement = random.choice(replacements)
    new_tokens = tokens[:]
    replace_idx = new_tokens.index(selected)
    new_tokens[replace_idx] = replacement

    return new_tokens

print("원본:", que_corpus[0])
print("증강:", lexical_sub(que_corpus[0], word2vec))
"""
    ),
    md(
        """
        ### 전체 데이터 증강

        LMS 조건에 맞춰 전체 데이터를 약 3배로 늘립니다.

        1. 원본 질문 + 원본 답변
        2. 증강 질문 + 원본 답변
        3. 원본 질문 + 증강 답변

        이렇게 만들면 질문과 답변의 병렬 관계를 크게 깨지 않으면서 데이터가 늘어납니다.
        """
    ),
    code(
        """aug_que_corpus = [
    lexical_sub(sentence, word2vec)
    for sentence in tqdm(que_corpus, desc="augment questions")
]

aug_ans_corpus = [
    lexical_sub(sentence, word2vec)
    for sentence in tqdm(ans_corpus, desc="augment answers")
]

original_count = len(que_corpus)

que_corpus = que_corpus + aug_que_corpus + que_corpus
ans_corpus = ans_corpus + ans_corpus + aug_ans_corpus

print("원본 데이터 수:", original_count)
print("증강 후 질문 데이터 수:", len(que_corpus))
print("증강 후 답변 데이터 수:", len(ans_corpus))
print("증가 배율:", len(que_corpus) / original_count)
print()
print("원본 질문 예시:", que_corpus[0])
print("증강 질문 예시:", que_corpus[original_count])
print("증강 답변 예시:", ans_corpus[original_count * 2])
"""
    ),
    md(
        """
        ### 4장 확인

        여기까지 확인할 내용입니다.

        - `Embedding source`가 출력되는지
        - 데이터 개수가 원래의 약 3배가 되었는지
        - `que_corpus`와 `ans_corpus`의 길이가 같은지

        다음 장에서는 답변 데이터에 `<start>`, `<end>` 토큰을 붙이고, 전체 말뭉치를 숫자 벡터로 바꿉니다.
        """
    ),
]

nb["cells"].extend(chapter_cells)

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"added chapter 4: {NB_PATH}")
