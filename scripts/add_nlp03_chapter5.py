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
if "## 7. 데이터 벡터화" in all_text:
    print("chapter 5 already exists")
    raise SystemExit

chapter_cells = [
    md(
        """
        ## 7. 데이터 벡터화

        지금까지의 `que_corpus`, `ans_corpus`는 토큰 리스트입니다.

        하지만 Transformer는 문자열 토큰을 직접 받을 수 없습니다.  
        그래서 각 토큰을 숫자 ID로 바꿔야 합니다.

        이 단계에서 하는 일은 세 가지입니다.

        1. 답변 문장 앞뒤에 `<start>`, `<end>` 토큰을 붙입니다.
        2. 질문과 답변 전체를 이용해 단어 사전을 만듭니다.
        3. 토큰 리스트를 숫자 ID 리스트로 바꾼 뒤, 같은 길이로 padding합니다.
        """
    ),
    code(
        """START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

ans_corpus = [
    [START_TOKEN] + sentence + [END_TOKEN]
    for sentence in ans_corpus
]

print("답변 예시:", ans_corpus[0])
"""
    ),
    md(
        """
        ### 단어 사전 만들기

        챗봇 데이터는 질문과 답변이 모두 한국어입니다.  
        그래서 질문용 사전과 답변용 사전을 따로 만들지 않고, 하나의 단어 사전을 공유합니다.

        `word2idx`는 토큰을 숫자로 바꾸는 사전이고, `idx2word`는 숫자를 다시 토큰으로 되돌리는 사전입니다.
        """
    ),
    code(
        """from collections import Counter

MIN_FREQ = 1

counter = Counter()
for sentence in que_corpus + ans_corpus:
    counter.update(sentence)

idx2word = [PAD_TOKEN, UNK_TOKEN]
idx2word += [
    word for word, count in counter.most_common()
    if count >= MIN_FREQ and word not in {PAD_TOKEN, UNK_TOKEN}
]

word2idx = {word: idx for idx, word in enumerate(idx2word)}

VOCAB_SIZE = len(idx2word)

print("단어 사전 크기:", VOCAB_SIZE)
print("앞쪽 단어 예시:", idx2word[:20])
"""
    ),
    md(
        """
        ### 토큰을 숫자로 바꾸기

        이제 토큰 리스트를 숫자 ID 리스트로 바꿉니다.

        사전에 없는 토큰은 `<unk>`의 번호로 바꿉니다.  
        길이가 짧은 문장은 `<pad>` 번호인 0으로 채우고, 너무 긴 문장은 `MAX_LEN` 길이로 자릅니다.
        """
    ),
    code(
        """import numpy as np

MAX_LEN = 40 + 2

def tokens_to_ids(tokens):
    return [word2idx.get(token, word2idx[UNK_TOKEN]) for token in tokens]

def pad_sequences(sequences, max_len=MAX_LEN, pad_value=0):
    padded = np.full((len(sequences), max_len), pad_value, dtype=np.int64)

    for i, sequence in enumerate(sequences):
        sequence = sequence[:max_len]
        padded[i, :len(sequence)] = sequence

    return padded

enc_train = pad_sequences([tokens_to_ids(sentence) for sentence in que_corpus])
dec_train = pad_sequences([tokens_to_ids(sentence) for sentence in ans_corpus])

print("enc_train shape:", enc_train.shape)
print("dec_train shape:", dec_train.shape)
print("질문 숫자 예시:", enc_train[0][:20])
print("답변 숫자 예시:", dec_train[0][:20])
"""
    ),
    md(
        """
        ### 학습/검증 데이터 나누기

        모델이 훈련 데이터만 외우는지 확인하려면, 일부 데이터를 검증용으로 남겨두는 것이 좋습니다.

        여기서는 전체 데이터의 10%를 검증 데이터로 사용합니다.
        """
    ),
    code(
        """from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 64

enc_train_np, enc_valid_np, dec_train_np, dec_valid_np = train_test_split(
    enc_train,
    dec_train,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

train_dataset = TensorDataset(
    torch.LongTensor(enc_train_np),
    torch.LongTensor(dec_train_np)
)
valid_dataset = TensorDataset(
    torch.LongTensor(enc_valid_np),
    torch.LongTensor(dec_valid_np)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("train size:", len(train_dataset))
print("valid size:", len(valid_dataset))
print("batch size:", BATCH_SIZE)
"""
    ),
    md(
        """
        ### 5장 확인

        여기까지 확인할 내용입니다.

        - `ans_corpus[0]` 앞뒤에 `<start>`, `<end>`가 붙었는지
        - `VOCAB_SIZE`가 출력되는지
        - `enc_train`, `dec_train`의 shape가 같은 첫 번째 차원을 가지는지
        - `train_loader`, `valid_loader`가 만들어졌는지

        다음 장에서는 이 데이터를 Transformer 모델에 넣어 훈련합니다.
        """
    ),
]

nb["cells"].extend(chapter_cells)

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"added chapter 5: {NB_PATH}")
