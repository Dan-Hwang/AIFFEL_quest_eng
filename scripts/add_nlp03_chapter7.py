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
if "## 9. 성능 측정하기" in all_text:
    print("chapter 7 already exists")
    raise SystemExit

chapter_cells = [
    md(
        """
        ## 9. 성능 측정하기

        챗봇은 번역 모델처럼 정답 문장이 하나로 고정되는 문제는 아닙니다.

        예를 들어 `"나 피곤해"`라는 질문에는 `"좀 쉬어도 돼요"`, `"무리하지 마세요"`, `"오늘은 일찍 자요"`처럼 여러 답변이 모두 자연스러울 수 있습니다.

        그래서 여기서는 두 가지 방식으로 확인합니다.

        1. 실제 질문을 넣어 답변을 직접 확인합니다.
        2. 참고 지표로 BLEU score를 계산합니다.
        """
    ),
    code(
        """final_questions = [
    "나 피곤해.",
    "너무 힘들어.",
    "좋아하는 사람이 생겼어.",
    "집에 가고 싶어."
]

print("Translations")
for i, question in enumerate(final_questions, start=1):
    answer = generate_answer(model, question)
    print(f"> {i}. Q: {question}")
    print(f">    A: {answer}")
"""
    ),
    md(
        """
        ### BLEU Score 계산

        BLEU는 생성 문장과 기준 문장이 얼마나 겹치는지 보는 지표입니다.

        다만 챗봇 답변은 정답이 하나가 아니기 때문에 BLEU 점수가 낮아도 무조건 나쁜 답변이라고 보기는 어렵습니다.  
        여기서는 모델 답변이 기준 답변과 어느 정도 겹치는지 확인하는 참고 지표로만 사용합니다.
        """
    ),
    code(
        """from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference, candidate):
    reference_tokens = tokenize(preprocess_sentence(reference))
    candidate_tokens = tokenize(preprocess_sentence(candidate))

    return sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        smoothing_function=SmoothingFunction().method1
    )

bleu_examples = [
    ("나 피곤해.", "무리하지 말고 조금 쉬어도 돼요."),
    ("너무 힘들어.", "힘들었겠어요. 조금 쉬어도 괜찮아요."),
    ("좋아하는 사람이 생겼어.", "좋아하는 마음이 생겼군요."),
    ("집에 가고 싶어.", "집에 가서 편히 쉬면 좋겠어요.")
]

total_bleu = 0.0

for question, reference in bleu_examples:
    candidate = generate_answer(model, question)
    score = calculate_bleu(reference, candidate)
    total_bleu += score

    print("Q:", question)
    print("Reference:", reference)
    print("Candidate:", candidate)
    print("BLEU:", f"{score:.4f}")
    print("-" * 40)

print("Average BLEU:", f"{total_bleu / len(bleu_examples):.4f}")
"""
    ),
    md(
        """
        ## 10. 제출 정리

        아래 값은 학습 결과를 바탕으로 제출 설명에 적기 위한 요약입니다.

        실제 출력된 `Best Epoch`, `Best Valid Loss`, 답변 예시는 실행 결과에 맞춰 확인하면 됩니다.
        """
    ),
    code(
        """print("Hyperparameters")
print("> n_layers:", N_LAYERS)
print("> d_model:", D_MODEL)
print("> n_heads:", N_HEADS)
print("> d_ff:", D_FF)
print("> dropout:", DROPOUT)

print("\\nTraining Parameters")
print("> warmup_steps:", WARMUP_STEPS)
print("> batch_size:", BATCH_SIZE)
print("> max_epochs:", EPOCHS)
print("> best_epoch:", best_epoch)
print("> best_valid_loss:", f"{best_valid_loss:.4f}")

print("\\nData")
print("> vocab_size:", VOCAB_SIZE)
print("> max_len:", MAX_LEN)
print("> train_size:", len(train_dataset))
print("> valid_size:", len(valid_dataset))
"""
    ),
    md(
        """
        ### 프로젝트 회고

        이번 프로젝트에서는 한국어 챗봇 데이터를 정제하고, 형태소 토큰화와 Lexical Substitution 기반 augmentation을 적용했습니다.

        이후 질문과 답변이 같은 한국어라는 점을 활용해 하나의 단어 사전을 공유했고, Transformer 기반 encoder-decoder 모델을 학습했습니다.

        챗봇은 정답이 하나로 고정되는 번역 문제와 달라 BLEU만으로 품질을 판단하기 어렵습니다. 따라서 BLEU는 참고 지표로 사용하고, 실제 질문에 대한 답변 예시를 함께 확인했습니다.
        """
    ),
]

nb["cells"].extend(chapter_cells)

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"added chapter 7: {NB_PATH}")
