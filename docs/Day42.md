# Day42 - KFold 교차 검증 & cross_val_score (iris 분류 · california 회귀) · RNGD furiosa-llm 서빙

**학습 기간:** 2026-11-02

> 지금까지는 `train_test_split`으로 train / test를 **한 번만** 나눠 평가했다. 이 날은 `KFold`로 전체 데이터를 5조각(fold)으로 나누고, `cross_val_score`로 **조각마다 돌아가며 검증하는 교차 검증(Cross Validation)** 을 했다. iris 분류(`DecisionTreeClassifier`)는 fold별 accuracy, california 회귀(`DecisionTreeRegressor`)는 fold별 R2를 받아 평균으로 모델 성능을 확인했다.
>
> 또 WSL 에 설치한 RNGD 에서 `furiosa-llm serve` 로 모델을 띄우고, OpenAI 클라이언트의 `base_url` 을 로컬 서버(`http://127.0.0.1:8000/v1`)로 바꿔 질문을 보내 봤다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 교차 검증 | 데이터를 n조각으로 나눠 1조각 검증 + 나머지 훈련을 n번 반복 → 모든 데이터가 한 번씩 검증에 쓰인다 |
| KFold | `KFold(n_splits=5, shuffle=True, random_state=333)` : 5조각, 섞어서 나눔 |
| cross_val_score | `cross_val_score(model, x, y, cv=kfold)` : fold마다 fit + score 자동 반복 → 점수 5개 배열 |
| 분류 / 회귀 점수 | Classifier → accuracy, Regressor → R2 (model.score와 같은 기준) |
| 대표 성능 | `np.mean(scores)` : fold 점수의 평균 |
| furiosa-llm serve | RNGD 에 모델을 올려 OpenAI 호환 서버(`127.0.0.1:8000/v1`)로 띄운다 |
| OpenAI 클라이언트 | `base_url` 만 로컬 서버로 바꾸면 `models.list` · `chat.completions.create` 를 그대로 쓴다 |

---

## 📖 핵심 학습 내용

### 1. KFold + cross_val_score 분류 (m02_kfold_00_iris)

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

#1. 데이터
x, y = load_iris(return_X_y=True)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)     # 5 조각, 섞어서 나눔

#2. 모델 구성
model = DecisionTreeClassifier()

#3. 훈련, 평가
scores = cross_val_score(model, x, y, cv=kfold)     # Classifier 기준 : fold 별 accuracy 5 개 반환

print("acc : ", scores, "\n평균 acc : ", round(np.mean(scores), 2))
```

- `KFold(n_splits=5)` : iris 150개 → 훈련 120개 / 검증 30개를 5번 (검증 조각이 매번 바뀐다)
- `shuffle=True` : iris는 클래스 순서대로 정렬되어 있어서, 섞지 않으면 fold마다 클래스가 치우친다
- `random_state=333` : 섞는 순서를 고정 → 실행할 때마다 같은 fold로 나뉜다
- `cross_val_score` : `fit` · `score`를 직접 부르지 않아도 fold마다 자동으로 훈련 · 평가
- `train_test_split` 없이 전체 x, y를 그대로 넣는다

```
acc :  [0.93333333 0.9        1.         0.96666667 0.9       ]
평균 acc :  0.94
```

- 같은 모델이라도 어느 조각이 검증이 되느냐에 따라 0.9 ~ 1.0으로 차이가 난다 → 한 번 나눈 점수보다 평균이 더 믿을 만하다
- 모델에 `random_state`를 주지 않아서 실행할 때마다 조금씩 달라진다 (다시 실행 : 마지막 fold 0.9333, 평균 0.95). KFold의 `random_state`는 데이터를 나누는 순서만 고정한다

### 2. KFold + cross_val_score 회귀 (m02_kfold_01_california)

m02_00을 베이스로 california 회귀 데이터에 `DecisionTreeRegressor`를 적용했다.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor

x, y = fetch_california_housing(return_X_y=True)

model = DecisionTreeRegressor()

scores = cross_val_score(model, x, y, cv=kfold) # Regressor 기준 : R2(결정계수) 반환

print("r2 : ", scores, "\n평균 r2 : ", round(np.mean(scores), 2))
```

```
r2 :  [0.6036485  0.60703136 0.57476646 0.61100558 0.63768908]
평균 r2 :  0.61
```

- 회귀 모델은 `cross_val_score`가 fold별 **R2**를 돌려준다
- fold별 점수는 실행할 때마다 조금씩 달라지지만 평균은 0.61 근처

### 3. RNGD furiosa-llm 서빙 + OpenAI 클라이언트 호출 (f01_furiosa_test)

터미널에서 모델을 서버로 띄운다. 파일 맨 위 주석에 현재 환경에서 실행해 본 모델을 정리했다 (`(X)` = 실행 안 됨).

```
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct                  (X)
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
furiosa-llm serve furiosa-ai/Qwen3.5-0.88                           (X)
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8
```

```python
from openai import OpenAI

client = OpenAI(                            # Request, Response 역할
    base_url = "http://127.0.0.1:8000/v1",
    api_key = "EMPTY"
)

# 서버에 로드된 모델 확인
models = client.models.list()
model_id = models.data[0].id

print("모델 :", model_id)

# 추론
response = client.chat.completions.create(
    model = model_id,
    messages=[
        {
            "role" : "user",
            "content" : "NPU와 GPU의 차이점을 설명해줘"
        }
    ],
    max_tokens=3000
)

print(response.choices[0].message.content)
```

- `furiosa-llm serve` 로 띄운 서버는 OpenAI API 와 같은 형식이라 `openai` 패키지를 그대로 쓴다
- `base_url` 을 로컬 서버 주소로 바꾸고, 로컬 서버라 `api_key` 는 `"EMPTY"` 로 둔다
- `client.models.list()` 로 서버에 올라간 모델을 확인하고, 그 id 를 `model` 에 넣어 질문한다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `ml/m02_kfold_00_iris.py` | iris 분류 `KFold(n_splits=5, shuffle=True)` + `cross_val_score` → fold별 accuracy · 평균 |
| `ml/m02_kfold_01_california.py` | california 회귀 `DecisionTreeRegressor` + `cross_val_score` → fold별 R2 · 평균 |
| `rngd/f01_furiosa_test.py` | RNGD 에 `furiosa-llm serve` 로 띄운 모델을 OpenAI 클라이언트(`base_url` 로컬 서버)로 호출 |
| `docs/Day42.md` | Day42 학습 기록 신규 작성 |
| `docs/Day41.md` | 하단 nav에 Day42 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(ml/ · rngd/ Day42) / 진행도(42일, 53%) 갱신 (수정) |

---

## 📊 실행 결과

| 파일 | fold별 점수 | 평균 |
|---|---|:---:|
| m02_00 (iris, accuracy) | `[0.9333 0.9 1.0 0.9667 0.9]` | 0.94 |
| m02_01 (california, R2) | `[0.6036 0.6070 0.5748 0.6110 0.6377]` | 0.61 |

---

## 💻 핵심 개념

### train_test_split vs KFold

```
train_test_split (한 번 나눔)               KFold(n_splits=5) (5 번 나눔)
[ train 80% | test 20% ]                     1 회 : [검증][훈련][훈련][훈련][훈련]
                                             2 회 : [훈련][검증][훈련][훈련][훈련]
test 로 뽑힌 데이터는 훈련에 안 쓰인다          ...
점수 1 개 (어떻게 나눴느냐에 따라 달라짐)       5 회 : [훈련][훈련][훈련][훈련][검증]
                                             점수 5 개 → 평균 = 대표 성능
```

### cross_val_score 흐름

```
kfold  = KFold(n_splits=5, shuffle=True, random_state=333)
scores = cross_val_score(model, x, y, cv=kfold)     # fold 마다 fit → score 를 5 번
np.mean(scores)                                     # Classifier → accuracy 평균 / Regressor → R2 평균
```

---

## 성과

- `KFold`로 데이터를 5조각으로 나눠, 모든 데이터가 한 번씩 검증에 쓰이는 교차 검증을 했다
- `cross_val_score` 한 줄로 fold별 훈련 · 평가를 반복하고, 분류(accuracy)와 회귀(R2)에 모두 적용했다
- fold마다 점수가 다르다는 것을 확인하고, 평균을 모델의 대표 성능으로 봤다
- RNGD 에 `furiosa-llm serve` 로 모델을 띄우고 OpenAI 클라이언트로 질문을 보냈다

---

## 💡 주요 학습 포인트

1. **교차 검증 = 검증 조각을 바꿔 가며 n번 평가**: 한 번 나눈 점수보다 평균이 더 믿을 만하다
2. **`KFold(n_splits=n)`**: 1조각 검증 + (n-1)조각 훈련을 n번 반복
3. **`shuffle=True`**: 정렬된 데이터(iris)는 섞어서 나눠야 fold마다 클래스가 고르게 들어간다
4. **KFold의 `random_state`는 나누는 순서만 고정**: 모델에 random_state가 없으면 점수는 실행마다 조금씩 달라진다
5. **`cross_val_score(model, x, y, cv=kfold)`**: fit · score를 직접 부르지 않고 fold별 점수 배열을 받는다
6. **점수 기준은 model.score와 같다**: Classifier → accuracy, Regressor → R2
7. **`furiosa-llm serve` = OpenAI 호환 서버**: `base_url` 만 `http://127.0.0.1:8000/v1` 로 바꾸면 OpenAI 클라이언트 코드를 그대로 쓴다

---

[⬅️ Day41](Day41.md) · [🏠 전체 목차](../README.md) · [Day43 ➡️](Day43.md)
