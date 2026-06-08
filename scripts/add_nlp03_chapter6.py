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
if "## 8. 훈련하기" in all_text:
    print("chapter 6 already exists")
    raise SystemExit

chapter_cells = [
    md(
        """
        ## 8. 훈련하기

        이제 벡터화된 데이터를 Transformer 모델에 넣어 학습합니다.

        앞 실습에서 직접 구현했던 Transformer와 구조는 같습니다.

        - 토큰 ID를 embedding 벡터로 바꿉니다.
        - 위치 정보를 positional encoding으로 더합니다.
        - encoder가 질문 문장을 읽습니다.
        - decoder가 `<start>`부터 시작해서 답변을 예측합니다.
        - 마지막 Linear 층이 각 위치마다 다음 토큰 확률을 만듭니다.

        여기서는 코드 길이를 줄이고 안정성을 높이기 위해 PyTorch의 `nn.Transformer`를 사용합니다.  
        직접 구현한 Multi-Head Attention, FFN, Encoder/Decoder Layer가 내부에 들어 있는 모듈이라고 보면 됩니다.
        """
    ),
    code(
        """import math
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
"""
    ),
    md(
        """
        ### Positional Encoding

        Transformer는 RNN처럼 순서대로 읽는 구조가 아닙니다.  
        그래서 토큰 embedding에 위치 정보를 따로 더해줘야 합니다.

        아래 `PositionalEncoding`은 앞 실습에서 본 sin/cos 위치 인코딩을 PyTorch 모듈로 만든 것입니다.
        """
    ),
    code(
        """class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

print("슝=3")
"""
    ),
    md(
        """
        ### 챗봇 Transformer 모델

        이 모델은 질문 토큰 `src`와 답변 입력 토큰 `tgt`를 받습니다.

        학습할 때 답변은 한 칸 밀어서 사용합니다.

        - decoder 입력: `<start> 오늘 은 ...`
        - 정답: `오늘 은 ... <end>`

        이렇게 해야 모델이 현재까지 본 답변을 바탕으로 다음 토큰을 맞히는 연습을 합니다.
        """
    ),
    code(
        """class ChatbotTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=8,
        d_ff=512,
        n_layers=2,
        dropout=0.2,
        max_len=MAX_LEN
    ):
        super().__init__()

        self.d_model = d_model
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_key_padding_mask = src.eq(0)
        tgt_key_padding_mask = tgt.eq(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1),
            device=tgt.device
        )

        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        return self.output_layer(out)

print("슝=3")
"""
    ),
    md(
        """
        ### Loss와 학습 함수

        `CrossEntropyLoss`는 모델의 예측과 실제 정답이 얼마나 다른지 계산합니다.

        `ignore_index=0`은 padding 토큰 `<pad>`는 loss 계산에서 제외하겠다는 뜻입니다.  
        padding은 문장 길이를 맞추기 위한 가짜 토큰이므로, 모델이 맞혀야 할 정답으로 보면 안 됩니다.
        """
    ),
    code(
        """criterion = nn.CrossEntropyLoss(ignore_index=0)

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for src, tgt in tqdm(loader, desc="train", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_input)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt_output.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total_loss = 0.0

    for src, tgt in tqdm(loader, desc="valid", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt_output.reshape(-1)
        )

        total_loss += loss.item()

    return total_loss / len(loader)

print("슝=3")
"""
    ),
    md(
        """
        ### 모델 생성 및 훈련

        데이터가 크지 않고 과제 시간이 제한되어 있으므로, 모델 크기는 너무 크게 잡지 않습니다.

        만약 시간이 부족하면 `EPOCHS = 1`로 줄여서 전체 흐름을 먼저 확인해도 됩니다.
        """
    ),
    code(
        """N_LAYERS = 2
D_MODEL = 256
N_HEADS = 8
D_FF = 512
DROPOUT = 0.2
EPOCHS = 3
LEARNING_RATE = 1e-4

model = ChatbotTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_ff=D_FF,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    max_len=MAX_LEN
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

history = []
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    valid_loss = evaluate_loss(model, valid_loader)
    history.append((train_loss, valid_loss))
    print(f"Epoch {epoch:02d} | train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f}")
"""
    ),
    md(
        """
        ### 6장 확인

        여기까지 확인할 내용입니다.

        - `device`가 `cuda`인지 확인합니다.
        - epoch마다 train loss와 valid loss가 출력되는지 확인합니다.
        - loss가 너무 크게 튀거나 `nan`이 되지 않는지 확인합니다.

        다음 장에서는 학습된 모델로 실제 질문에 답변을 생성하고, BLEU 점수도 계산합니다.
        """
    ),
]

nb["cells"].extend(chapter_cells)

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"added chapter 6: {NB_PATH}")
