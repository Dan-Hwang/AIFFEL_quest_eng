import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "NLP" / "NLP03" / "NLP03_project_chatbot.ipynb"


TRAIN_FUNCTION_CODE = """criterion = nn.CrossEntropyLoss(ignore_index=0)

class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=1000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

def train_one_epoch(model, loader, optimizer, lr_scheduler=None):
    model.train()
    total_loss = 0.0
    last_lr = optimizer.param_groups[0]["lr"]

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

        if lr_scheduler is not None:
            last_lr = lr_scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader), last_lr

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


GENERATION_CODE = """def ids_to_sentence(ids):
    words = []
    for idx in ids:
        word = idx2word[int(idx)]
        if word == END_TOKEN:
            break
        if word not in [PAD_TOKEN, START_TOKEN]:
            words.append(word)

    sentence = " ".join(words)

    for mark in [".", "?", "!", ","]:
        sentence = sentence.replace(" " + mark, mark)

    for particle in ["은", "는", "이", "가", "을", "를", "도", "에", "에서", "에게", "와", "과", "로", "으로", "요"]:
        sentence = sentence.replace(" " + particle, particle)

    sentence = sentence.replace("하 다", "하다")
    sentence = sentence.replace("되 다", "되다")
    sentence = sentence.replace("있 어요", "있어요")
    sentence = sentence.replace("없 어요", "없어요")
    sentence = sentence.replace("해 줄게요", "해줄게요")
    return sentence.strip()

def encode_question(sentence):
    tokens = tokenize(preprocess_sentence(sentence))
    ids = tokens_to_ids(tokens)
    padded = pad_sequences([ids], max_len=MAX_LEN)
    return torch.LongTensor(padded).to(device)

@torch.no_grad()
def generate_answer(model, question, max_len=MAX_LEN, min_len=3):
    model.eval()

    src = encode_question(question)
    generated = [word2idx[START_TOKEN]]

    for _ in range(max_len - 1):
        tgt = torch.LongTensor([generated]).to(device)
        logits = model(src, tgt)

        next_id = logits[0, -1].argmax(dim=-1).item()
        generated.append(next_id)

        if next_id == word2idx[END_TOKEN] and len(generated) > min_len:
            break

    return ids_to_sentence(generated)

print("슝=3")
"""


TRAINING_CODE = """import copy

N_LAYERS = 2
D_MODEL = 384
N_HEADS = 8
D_FF = 1024
DROPOUT = 0.2
EPOCHS = 10
WARMUP_STEPS = 1000

model = ChatbotTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_ff=D_FF,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    max_len=MAX_LEN
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0,
    betas=(0.9, 0.98),
    eps=1e-9
)

lr_scheduler = TransformerLRScheduler(
    optimizer,
    d_model=D_MODEL,
    warmup_steps=WARMUP_STEPS
)

sample_questions = [
    "나 피곤해.",
    "너무 힘들어.",
    "오늘 기분이 안 좋아.",
    "심심해.",
    "헤어졌어.",
    "나랑 놀자.",
    "좋아하는 사람이 생겼어.",
    "집에 가고 싶어."
]

best_valid_loss = float("inf")
best_epoch = 0
best_model_state = None
history = []

for epoch in range(1, EPOCHS + 1):
    train_loss, current_lr = train_one_epoch(model, train_loader, optimizer, lr_scheduler)
    valid_loss = evaluate_loss(model, valid_loader)
    history.append((train_loss, valid_loss, current_lr))

    print(f"Epoch {epoch:02d} | train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f} | lr: {current_lr:.6f}")

    for question in sample_questions:
        print("Q:", question)
        print("A:", generate_answer(model, question))
    print("-" * 60)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model_state)

print("Best Epoch:", best_epoch)
print("Best Valid Loss:", f"{best_valid_loss:.4f}")
print("Best model sample answers")
for question in sample_questions:
    print("Q:", question)
    print("A:", generate_answer(model, question))
"""


with NB_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue

    source = "".join(cell.get("source", []))
    if "def train_one_epoch" in source and "evaluate_loss" in source:
        cell["source"] = [line + "\n" for line in TRAIN_FUNCTION_CODE.strip().splitlines()]
    elif "def ids_to_sentence" in source and "def generate_answer" in source:
        cell["source"] = [line + "\n" for line in GENERATION_CODE.strip().splitlines()]
    elif "model = ChatbotTransformer" in source and "EPOCHS" in source and "optimizer" in source:
        cell["source"] = [line + "\n" for line in TRAINING_CODE.strip().splitlines()]

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"updated training cells: {NB_PATH}")
