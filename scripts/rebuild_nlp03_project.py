# -*- coding: utf-8 -*-
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "NLP" / "NLP03" / "NLP03_project_chatbot.ipynb"


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


cells = [
    md(
        """
# NLP03 Project: 멋진 챗봇 만들기

이번 프로젝트에서는 번역 모델 실습에서 학습한 전처리, 토큰화, Transformer 모델 학습, BLEU 평가 흐름을 한국어 챗봇 데이터에 적용한다.

프로젝트 진행은 다음 순서로 나눈다.

1. 라이브러리 및 데이터 준비
2. 데이터 정제
3. 데이터 토큰화
4. 데이터 Augmentation
5. 데이터 벡터화
6. Transformer 훈련
7. 성능 측정 및 회고

우선 한 단계씩 작성하고 실행 결과를 확인하면서 진행한다.
"""
    ),
    md(
        """
## 1. 라이브러리 버전 확인

프로젝트에 사용할 주요 라이브러리를 불러오고 버전을 확인한다.
현재 프로젝트는 `AIFFEL py312 GPU` 커널 사용을 기준으로 한다.

`torch.cuda.is_available()`이 `True`이면 GPU를 사용할 수 있는 상태다.
"""
    ),
    code(
        """
import numpy
import pandas
import torch
import nltk
import gensim

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("torch:", torch.__version__)
print("nltk:", nltk.__version__)
print("gensim:", gensim.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
"""
    ),
    md(
        """
## 2. 데이터 다운로드

프로젝트에서 사용할 데이터는 `songys/Chatbot_data`의 `ChatbotData.csv`이다.
CSV 파일을 내려받은 뒤 `pandas`로 읽고, 질문 데이터와 답변 데이터를 각각 `questions`, `answers` 변수로 분리한다.

- `Q`: 질문 문장
- `A`: 답변 문장
- `label`: 감정 라벨이며, 이번 챗봇 생성 모델에서는 사용하지 않는다.
"""
    ),
    code(
        """
import os
import urllib.request

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv"
DATA_PATH = os.path.join(DATA_DIR, "ChatbotData.csv")

if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("Downloaded:", DATA_PATH)
else:
    print("Already exists:", DATA_PATH)
"""
    ),
    md(
        """
## 3. 데이터 불러오기 및 기본 확인

다운로드한 CSV 파일을 읽어 데이터의 크기와 컬럼을 확인한다.
이후 질문과 답변 컬럼을 리스트로 분리해 다음 단계의 전처리 입력으로 사용한다.
"""
    ),
    code(
        """
df = pandas.read_csv(DATA_PATH)

print("데이터 크기:", df.shape)
print("컬럼:", df.columns.tolist())
display(df.head())

questions = df["Q"].astype(str).tolist()
answers = df["A"].astype(str).tolist()

print("질문 개수:", len(questions))
print("답변 개수:", len(answers))
print("질문 예시:", questions[0])
print("답변 예시:", answers[0])
"""
    ),
    md(
        """
### 1장 확인

여기까지 완료되면 다음을 확인한다.

- 라이브러리 import가 모두 성공했는가?
- `cuda available`이 `True`로 나오는가?
- `ChatbotData.csv`가 정상적으로 다운로드되었는가?
- `questions`, `answers`의 길이가 같은가?

다음 장에서는 `preprocess_sentence()`를 구현해 문장을 정제한다.
"""
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "AIFFEL py312 GPU",
            "language": "python",
            "name": "aiffel_py312_gpu",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"rebuilt chapter 1: {NB_PATH}")
