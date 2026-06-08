import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "NLP" / "NLP03" / "NLP03_project_chatbot.ipynb"


LOADER_CODE = """import random
from gensim import utils
from gensim.models import Word2Vec, KeyedVectors

random.seed(42)

KO_BIN_PATH = os.path.join(DATA_DIR, "ko.bin")

def load_ko_word2vec(path):
    \"\"\"Kyubyong ko.bin처럼 오래된 gensim 형식도 최신 KeyedVectors로 변환해 읽습니다.\"\"\"
    try:
        return KeyedVectors.load_word2vec_format(path, binary=True)
    except Exception:
        pass

    try:
        model = Word2Vec.load(path)
        if hasattr(model, "wv"):
            return model.wv
    except Exception:
        pass

    old_model = utils.unpickle(path)
    vectors = old_model.syn0
    words = old_model.index2word

    keyed_vectors = KeyedVectors(vector_size=vectors.shape[1])
    keyed_vectors.add_vectors(words, vectors)
    return keyed_vectors

if os.path.exists(KO_BIN_PATH):
    word2vec = load_ko_word2vec(KO_BIN_PATH)
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
print("Vector size:", word2vec.vector_size)
"""


with NB_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

updated = False
for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "code" and "KO_BIN_PATH" in source and "Embedding source" in source:
        cell["source"] = [line + "\n" for line in LOADER_CODE.strip().splitlines()]
        updated = True

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"updated ko.bin loader: {NB_PATH}, updated={updated}")
