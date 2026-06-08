import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "NLP" / "NLP03" / "NLP03_project_chatbot.ipynb"


SPLIT_CODE = """import torch
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 64
VALID_RATIO = 0.1

indices = np.arange(len(enc_train))
np.random.seed(42)
np.random.shuffle(indices)

valid_size = int(len(indices) * VALID_RATIO)
valid_indices = indices[:valid_size]
train_indices = indices[valid_size:]

enc_train_np = enc_train[train_indices]
dec_train_np = dec_train[train_indices]
enc_valid_np = enc_train[valid_indices]
dec_valid_np = dec_train[valid_indices]

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


with NB_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

updated = False
for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "code" and "train_test_split" in source:
        cell["source"] = [line + "\n" for line in SPLIT_CODE.strip().splitlines()]
        updated = True

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"updated split cell: {NB_PATH}, updated={updated}")
