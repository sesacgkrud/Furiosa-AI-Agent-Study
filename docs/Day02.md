# Day02 - 다중 입력/출력 신경망 및 데이터 분할

**학습 기간:** 2026-09-01

> 입력·출력이 여러 개인 MLP를 만들고, numpy shape를 읽는 법과 Train/Test 분할 개념을 익혔다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| shape 읽기 | 대괄호 `[` 가 열린 깊이 = 차원 수, 각 자리 숫자 = 그 차원의 원소 개수 |
| 데이터 형태 | 케라스는 `(행 = 데이터 개수, 열 = 특성 개수)` 로 읽는다 |
| `input_dim` | **열(특성)의 개수**를 넣는다 |
| 전치(Transpose) | `x.T` / `x.transpose()` - 행과 열을 바꿔 형태를 맞춘다 |
| 다중 출력 | 마지막 `Dense` 의 숫자 = y의 열 개수 |
| Train/Test 분할 | 훈련은 `x_train` 으로, 평가는 한 번도 안 본 `x_test` 로 |

---

## 📖 핵심 학습 내용

### numpy 다차원 배열과 shape
- 대괄호 `[` 가 열린 깊이 = 차원 수, 각 자리의 숫자 = 그 차원의 원소 개수
- `(3,)` = 1차원 3개 / `(1, 3)` = 2차원 / `(1, 3, 2)` = 3차원
- 모델의 `input_dim` 을 맞추려면 데이터 shape를 먼저 정확히 읽을 수 있어야 한다

### 다중 입력(Multi-Input)
- 케라스는 데이터를 `(행 = 데이터 개수, 열 = 특성 개수)` 로 읽는다
- 따라서 `input_dim` 에는 **열의 개수**를 넣는다
- `(2, 5)` 처럼 잘못된 형태면 전치해서 `(5, 2)` 로 바꾼다

### 데이터 전치(Transpose)
- `x.T` 또는 `x.transpose()` - 둘 다 결과는 같다
- 데이터를 행 단위로 쓰고 마지막에 `.T` 로 뒤집는 방식은 앞으로 계속 쓴다

### 다중 출력(Multi-Output)
- y의 열 개수만큼 마지막 `Dense` 의 숫자를 맞춘다
- 입력 개수와 출력 개수는 서로 달라도 된다 (`(10,)` 입력 -> `(10, 3)` 출력)

### Train/Test 분할 입문
- 훈련은 `x_train` 으로만 하고, 평가는 한 번도 보지 않은 `x_test` 로 한다
- 훈련에 쓴 데이터로 평가하면 외운 답을 다시 맞히는 셈이라 성능을 알 수 없다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras07_matrix.py` | Numpy 다차원 배열 기초 (1D, 2D, 3D, 5D 데이터 구조) |
| `keras08_mlp1_1.py` | 다중 입력 신경망 (2입력-1출력, shape: (5,2)) |
| `keras08_mlp1_2.py` | 데이터 전치 적용 (2입력-1출력, transpose 사용) |
| `keras08_mlp2_1.py` | 3입력-1출력 MLP (10->5->2->1 구조, loss: 5.54e-05) |
| `keras08_mlp2_2.py` | 3입력-1출력 MLP + `range()` 로 데이터 만들기 |
| `keras08_mlp3_1.py` | 3입력-2출력 MLP (20->10->5->2 구조) |
| `keras08_mlp4.py` | 1입력-3출력 MLP (10->5->3 구조) |
| `keras09_train_test1.py` | Train/Test 분할 (검증 방식 학습) |

---

## 💻 핵심 개념

### 데이터 전치

```python
x.T            # numpy array 전치
x.transpose()  # 메서드를 이용한 전치
```

### 다중 입력/출력

```python
Dense(n_output, input_dim=n_input)  # n_input 개의 입력, n_output 개의 출력
```

### Train/Test 분할

```python
x_train, x_test = x[:7], x[7:]
y_train, y_test = y[:7], y[7:]
model.fit(x_train, y_train, ...)  # 훈련
model.evaluate(x_test, y_test)    # 검증
```

---

## 성과
- 다중 입력/출력 신경망 완성
- 다차원 데이터 처리 능력 확보
- Train/Test 분할 개념 이해
- MLP 고급 구조 마스터

---

## 💡 주요 학습 포인트
1. **데이터 형태**: (sample, features) 형태 유지
2. **Transpose**: 데이터를 올바른 형태로 변환
3. **다중 출력**: 최종 Dense layer 출력값 = 원하는 출력 개수
4. **Train/Test**: 전체 데이터를 훈련/검증 데이터로 분할

---

[⬅️ Day01](Day01.md) · [🏠 전체 목차](../README.md) · [Day03 ➡️](Day03.md)
