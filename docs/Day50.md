# Day50 - 활성화 함수 직접 구현 · 시각화 (sigmoid · tanh · ReLU · SiLU · Mish · ELU · SELU · Leaky ReLU) & R-CNN · Fast · Faster · Mask R-CNN 논문 발표

**학습 기간:** 2026-11-12

> 신경망에서 쓰는 활성화 함수 8 개를 numpy 식으로 직접 만들고(def → lambda), x = -5 ~ 4.9 (0.1 간격, 100 개) 에 대해 matplotlib 으로 그래프를 그려 모양을 비교했다 (keras78_01 ~ 08).
>
> 객체 탐지(Object Detection) 논문 R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN 발표가 있었다.

---

## 🎯 한눈에 보기

| 함수 | 식 | 출력 범위 · 특징 |
|---|---|---|
| sigmoid | 1 / (1 + e^-x) | 0 ~ 1, x = 0 이면 0.5 |
| tanh | np.tanh(x) | -1 ~ 1, 0 중심 대칭 |
| ReLU | max(0, x) | 음수는 0, 양수는 그대로 |
| SiLU | x · sigmoid(x) | 음수 쪽 최소 약 -0.28, 부드러운 곡선 |
| Mish | x · tanh(ln(1 + e^x)) | 음수 쪽 최소 약 -0.31, SiLU 와 비슷 |
| ELU | x (x > 0), α(e^x - 1) (x ≤ 0) | 음수 쪽이 -α 로 수렴 (α = 1) |
| SELU | λ · ELU (α ≈ 1.6733, λ ≈ 1.0507) | 음수 쪽이 약 -1.76 으로 수렴, 고정 상수 |
| Leaky ReLU | x (x > 0), αx (x ≤ 0) | 음수에 작은 기울기 α = 0.01 |

---

## 📖 핵심 학습 내용

### 1. sigmoid · tanh (keras78_01 · 02)

```python
def sigmoid (x):
    return 1 / (1 + np.exp(-x))    # 1 / (1 + e^-x)

sigmoid = lambda x : 1 / (1 + np.exp(-x))
x = np.arange(-5, 5, 0.1)
print(len(x))   # 100

y = sigmoid(x)
plt.plot(x, y)
plt.grid()
plt.show()
```

```python
y = np.tanh(x)      # numpy 내장 함수를 바로 사용
```

- `np.arange(-5, 5, 0.1)` : -5 부터 5 직전(4.9) 까지 0.1 간격 → 100 개
- numpy 배열을 함수에 넣으면 원소마다 계산된 배열이 나온다 → 그대로 `plt.plot(x, y)`
- 같은 함수를 `def` 와 `lambda` 두 방식으로 써 봤다 (뒤에 정의한 lambda 가 덮어쓴다)
- sigmoid : 0 ~ 1, 이진 분류 출력층 / tanh : -1 ~ 1, 0 중심 대칭, LSTM · GRU 내부 기본 활성화

### 2. ReLU · SiLU · Mish (keras78_03 · 04 · 05)

```python
relu  = lambda x : np.maximum(0, x)
silu  = lambda x : x * (1 / (1 + np.exp(-x)))
mish  = lambda x : x * np.tanh(np.log(1 + np.exp(x)))
```

- ReLU : `np.maximum(0, x)` 로 0 과 원소마다 비교 → 음수는 0. 음수 입력은 기울기도 0 이라 dying ReLU 가 생길 수 있다
- SiLU : x 에 sigmoid(x) 를 곱한 값 → 음수 쪽을 0 으로 자르지 않고 부드럽게 (최소 약 -0.28)
- Mish : `ln(1 + e^x)` (softplus) 에 tanh 를 씌워 x 에 곱한 값 → SiLU 와 비슷한 모양 (최소 약 -0.31)

### 3. ELU · SELU (keras78_06 · 07)

```python
alpha = 1
elu = lambda x, alpha : (x > 0) * x + (x < 0) * (alpha * (np.exp(x) - 1))

# SELU 고정 상수 (논문 제안값)
alpha = 1.6732632423543772
lmbda = 1.0507009873554805
selu = lambda x, alpha, lmbda : lmbda * ((x > 0) * x + (x <= 0) * (alpha * (np.exp(x) -1)))
```

- `(x > 0)` 은 True / False 배열 → 곱하면 1 / 0 으로 계산되어 조건에 맞는 쪽 식만 남는다 (구간별 함수를 if 없이)
- lambda 에 인자를 여러 개 (`x, alpha, lmbda`) 받을 수 있다
- `lambda` 는 파이썬 예약어라 SELU 상수 변수 이름을 `lmbda` 로 썼다
- ELU 음수 쪽은 -alpha (= -1) 로, SELU 음수 쪽은 -λα ≈ -1.76 으로 가까워진다

### 4. Leaky ReLU (keras78_08)

```python
alpha = 0.01

# def leaky_relu(x):
#     # return np.maximum(alpha * x, x)
#     return np.where(x > 0, x, alpha * x)

leaky_relu = lambda x : np.where(x > 0, x, alpha * x)
```

- `np.where(조건, 참일 때 값, 거짓일 때 값)` : 원소마다 조건으로 값을 고른다
- `np.maximum(alpha * x, x)` 도 같은 결과 (alpha < 1 이면 양수는 x, 음수는 alpha * x 가 더 크다)
- 음수에도 작은 기울기(0.01) 를 남겨 dying ReLU 를 줄인다

### 5. 논문 발표 : R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN

| 논문 | 핵심 |
|---|---|
| R-CNN | 후보 영역(Region Proposal) 을 뽑아 영역마다 CNN 으로 특징을 뽑고 분류하는 2 단계 객체 탐지 |
| Fast R-CNN | 이미지 전체를 CNN 에 한 번만 통과시키고, 특징 맵에서 후보 영역을 잘라(RoI Pooling) 분류 · 박스 조정을 함께 학습 |
| Faster R-CNN | 후보 영역 추출도 신경망(RPN, Region Proposal Network) 으로 바꿔 전체를 하나의 네트워크로 |
| Mask R-CNN | Faster R-CNN 에 객체별 마스크(픽셀 단위 영역) 를 예측하는 가지를 더해 인스턴스 분할까지 |

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras2/keras78_01_sigmoid.py` | sigmoid 를 def · lambda 로 만들고 -5 ~ 4.9 구간 그래프 |
| `keras2/keras78_02_tanh.py` | `np.tanh` 그래프 |
| `keras2/keras78_03_relu.py` | `np.maximum(0, x)` 로 ReLU 그래프 |
| `keras2/keras78_04_SiLu.py` | x · sigmoid(x) SiLU 그래프 |
| `keras2/keras78_05_Mish.py` | x · tanh(ln(1 + e^x)) Mish 그래프 |
| `keras2/keras78_06_elu.py` | 조건 배열 곱으로 구간별 ELU (alpha = 1) 그래프 |
| `keras2/keras78_07_selu.py` | SELU 고정 상수 alpha · lmbda 로 SELU 그래프 |
| `keras2/keras78_08_leaky_relu.py` | `np.where` 로 Leaky ReLU (alpha = 0.01) 그래프 |
| `docs/Day50.md` | Day50 학습 기록 신규 작성 |
| `docs/Day49.md` | 하단 nav에 Day50 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(keras2/ Day50) / 진행도(50일, 63%) 갱신 (수정) |

---

## 💻 핵심 개념

### def vs lambda

```
def relu(x):                        : 이름 · 여러 줄 본문 · return
    return np.maximum(0, x)
relu = lambda x : np.maximum(0, x)  : 식 하나짜리 함수를 한 줄로 (인자 여러 개 가능 : lambda x, alpha : ...)
```

### 구간별 함수를 numpy 로

```
(x > 0) * A + (x <= 0) * B   : True=1, False=0 을 곱해 조건에 맞는 쪽만 남김 (ELU · SELU)
np.where(x > 0, A, B)        : 조건이 참이면 A, 거짓이면 B (Leaky ReLU)
np.maximum(0, x)             : 원소마다 큰 값 (ReLU)
```

### 음수 구간 비교

```
ReLU        : 0 (기울기 0)
Leaky ReLU  : 0.01 * x
ELU         : -1 로 수렴 (alpha = 1)
SELU        : 약 -1.76 으로 수렴
SiLU · Mish : 살짝 음수(약 -0.28 · -0.31) 로 내려갔다가 0 으로
```

---

## 성과

- 활성화 함수 8 개를 numpy 식으로 직접 구현하고 그래프로 모양을 비교했다
- 같은 함수를 def 와 lambda 두 방식으로 작성했다
- 조건 배열 곱 · `np.where` · `np.maximum` 으로 구간별 함수를 if 없이 계산했다
- R-CNN 계열 객체 탐지 논문 4 편 발표를 들었다

---

## 💡 주요 학습 포인트

1. **sigmoid**: 0 ~ 1, 양 끝 기울기가 0 에 가까워 기울기 소실
2. **tanh**: -1 ~ 1, 0 중심 대칭
3. **ReLU**: 음수는 0 → 계산이 단순, dying ReLU 가능
4. **SiLU · Mish**: 음수 쪽을 부드럽게 살린 ReLU 계열
5. **ELU · SELU**: 음수 쪽이 지수 함수로 일정 값에 수렴, SELU 는 고정 상수 alpha · lambda
6. **Leaky ReLU**: 음수에 작은 기울기 0.01
7. **np.arange · np.where · np.maximum**: 구간 만들기와 원소별 조건 계산
8. **R-CNN → Fast → Faster → Mask R-CNN**: 영역별 CNN → 특징 맵 공유 → RPN 으로 후보 영역 → 마스크까지 예측

---

[⬅️ Day49](Day49.md) · [🏠 전체 목차](../README.md)
