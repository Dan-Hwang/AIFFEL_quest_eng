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
if "## 4. 데이터 정제" in all_text:
    print("chapter 2 already exists")
    raise SystemExit

chapter_cells = [
    md(
        """
        ## 4. 데이터 정제

        이번 장에서는 원본 문장을 모델이 다루기 쉬운 형태로 정리합니다.

        LMS 조건은 두 가지입니다.

        1. 영어 문자는 모두 소문자로 바꿉니다.
        2. 한글, 영어, 숫자, 주요 문장부호만 남기고 나머지는 제거합니다.

        여기서 너무 많은 전처리를 하지는 않겠습니다. 뒤에서 `mecab.morphs`가 한국어를 토큰화해 줄 것이기 때문에, 문장부호 양옆에 공백을 넣는 작업은 생략합니다.
        """
    ),
    code(
        """
        import re

        def preprocess_sentence(sentence):
            sentence = str(sentence).lower()
            sentence = re.sub(r"[^가-힣a-z0-9?.!,]+", " ", sentence)
            sentence = re.sub(r"\\s+", " ", sentence)
            sentence = sentence.strip()
            return sentence

        print("슝=3")
        """
    ),
    md(
        """
        ### 전처리 결과 확인

        함수가 제대로 작동하는지 바로 확인해 봅니다.

        아래 셀은 원본 질문/답변 5개와 전처리된 결과를 나란히 보여줍니다.  
        여기서는 아직 데이터를 바꾸는 단계가 아니라, **정제 함수가 어떻게 작동하는지 확인하는 단계**입니다.
        """
    ),
    code(
        """
        for i in range(5):
            print(f"[{i}] 질문 원본:", questions[i])
            print(f"[{i}] 질문 정제:", preprocess_sentence(questions[i]))
            print(f"[{i}] 답변 원본:", answers[i])
            print(f"[{i}] 답변 정제:", preprocess_sentence(answers[i]))
            print("-" * 60)
        """
    ),
    md(
        """
        ### 2장 확인

        여기까지 확인할 내용은 간단합니다.

        - `preprocess_sentence()` 함수가 실행되는지
        - 영어가 소문자로 바뀌는지
        - 이상한 특수문자가 제거되는지
        - 한글 문장이 너무 망가지지 않는지

        다음 장에서는 이 함수를 실제 데이터 전체에 적용하면서, `mecab.morphs`로 질문과 답변을 토큰화하는 `build_corpus()`를 만들겠습니다.
        """
    ),
]

nb["cells"].extend(chapter_cells)

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"added chapter 2: {NB_PATH}")
