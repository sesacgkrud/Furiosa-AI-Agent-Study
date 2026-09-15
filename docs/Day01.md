# Day01 - Keras 기초 및 Deep Learning 구조

**학습 기간:** 2026-08-31

> Sequential 모델로 첫 신경망을 만들고 **데이터 -> 모델 -> 컴파일/훈련 -> 평가/예측** 4단계 흐름을 익혔다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 4단계 흐름 | `#1 데이터` -> `#2 모델 구성` -> `#3 컴파일, 훈련` -> `#4 평가, 예측` |
| Sequential | 층을 순서대로 쌓는 모델. `model.add(Dense(...))` 로 추가 |
| Dense | `Dense(출력 개수, input_dim=입력 개수)` - 완전연결층 |
| 손실함수 / 옵티마이저 | `loss='mse'` + `optimizer='adam'` (회귀의 기본 조합) |
| 성능 올리는 법 | 층을 깊게, epochs를 늘리기, 노드 수 조절 |
| batch_size | 데이터를 몇 개씩 잘라서 가중치를 갱신할지 (기본 32) |

---

## 📖 핵심 학습 내용

### 모델 만드는 4단계
- `#1 데이터` -> `#2 모델 구성` -> `#3 컴파일, 훈련` -> `#4 평가, 예측`
- 앞으로 모든 파일이 이 순서를 그대로 따른다

### Sequential + Dense
- `Sequential()` - 층을 순서대로 쌓는 모델 (클래스라서 `()` 까지 써야 한다)
- `Dense(1, input_dim=1)` - 입력 1개를 받아 출력 1개를 내는 완전연결층
- 두 번째 층부터는 `input_dim` 을 생략한다 (앞 층의 출력 개수가 자동으로 입력이 된다)

### 컴파일과 훈련
- `loss='mse'` - 오차를 제곱해서 평균 낸 값. 회귀의 기본 손실함수
- `optimizer='adam'` - 대부분의 상황에서 무난한 최적화 알고리즘
- `epochs` - 같은 데이터를 몇 번 반복해서 훈련할지
- `batch_size` - 데이터를 몇 개씩 잘라서 가중치를 갱신할지 (작게 자를수록 1 epoch 안의 갱신 횟수가 늘어난다)

### 성능 개선 방법
- 층을 더 깊게 쌓는다 (`500 -> 300 -> 1`)
- `epochs` 를 늘린다 (300 -> 900)
- 노드 수를 조절한다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras01.py` | 기본 신경망 구성 (1-1-1 구조) |
| `keras02.py` | 선형 회귀 모델 + `evaluate` 추가 |
| `keras03.py` | epochs 조정을 통한 손실값 개선 (900 epochs) |
| `keras04_deep1.py` | 4층 Deep Learning (500-300-1 구조) |
| `keras05_deep2.py` | 3층 최적화 네트워크 (1000-300-1 구조), `input_dim` 생략 |
| `keras06_batch.py` | Batch Processing 개념 이해 |

---

## 💻 핵심 개념

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))    # Dense(출력 개수, 입력 개수)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300)

#4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict(np.array([4]))
```

---

## 성과
- 기본 신경망 구성 완료
- 손실값 최적화 (0.3238)
- Deep Learning 개념 이해

---

## 💡 주요 학습 포인트
1. **모델 구성**: `Sequential()` -> `add(Dense())` -> `compile()` -> `fit()`
2. **손실함수**: MSE (Mean Squared Error)
3. **최적화**: Adam optimizer
4. **성능 개선**: 레이어 깊이 증가, epochs 조정, 뉴런 수 증가

---

[🏠 전체 목차](../README.md) · [Day02 ➡️](Day02.md)
