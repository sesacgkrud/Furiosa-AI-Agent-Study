# Day18 - learning_rate 직접 지정, ReduceLROnPlateau, RNN · LSTM 입문

**학습 기간:** 2026-09-23

> Day17까지는 **데이터**(스케일링, 증폭)와 **모델 구조**(CNN, DNN, 함수형)를 바꿔 왔다. Day18은 처음으로 **훈련 방식 자체**를 건드린다. `optimizer='adam'` 문자열 뒤에 숨어 있던 **learning_rate**를 꺼내 직접 정하고, 그것도 고정값이 아니라 **ReduceLROnPlateau**로 훈련 도중 줄여 나갔다. 지금까지 쓴 15개 데이터셋 전부에 같은 실험을 반복했다. 오후에는 시계열을 다루는 **SimpleRNN**으로 넘어가 3차원 입력과 파라미터 계산식을 익히고, **LSTM**의 파라미터가 왜 4배인지까지 확인했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| learning_rate | 가중치를 한 번에 얼마나 크게 고칠지 정하는 **보폭** |
| `optimizer='adam'` | 문자열로 주면 learning_rate가 기본값 **0.001**로 고정된다 |
| `Adam(learning_rate=...)` | 객체로 만들어 넘겨야 보폭을 내가 정할 수 있다 |
| 크게 주면 | 최저점을 건너뛰고 주변에서 튕겨 다닌다 (0.05 / 0.01) |
| 작게 주면 | 한 걸음이 작아 epoch가 끝날 때까지 최저점에 못 닿는다 (0.00001) |
| batch_size와의 관계 | batch_size가 작을수록 learning_rate도 작게 가야 한다 |
| ReduceLROnPlateau | val_loss가 평평해지면 learning_rate에 factor를 곱해 줄인다 |
| es vs rlr | es는 **멈춘다**, rlr은 **보폭을 줄여 더 해 본다** |
| patience 순서 | `rlr.patience < es.patience` 여야 "줄여 보고 → 그래도 안 되면 멈춘다"가 된다 |
| rlr의 전제 | 시작 learning_rate를 넉넉히 줘야 줄일 여지가 생긴다 |
| RNN | 앞에서 뒤로 흐르는 **순서**를 보는 층 (시계열) |
| RNN 입력 | **3차원** `(N, timesteps, features)` |
| RNN 출력 | 마지막 결과 하나만 내보내 **2차원** → Dense와 바로 연결 |
| RNN 파라미터 | `(units × features) + (units × units) + units` — timesteps는 식에 없다 |
| `input_length` / `input_dim` | `input_shape=(3, 1)`을 나눠서 쓰는 표기법 |
| LSTM 파라미터 | SimpleRNN의 **정확히 4배** (게이트가 4덩어리) |

---

## 📖 핵심 학습 내용

### 1. learning_rate를 직접 지정 (keras52_optimizer01 ~ 15)

지금까지는 `model.compile(..., optimizer='adam')` 이라고 문자열만 썼다. 이러면 learning_rate가 **기본값 0.001**로 고정된다.

```python
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
```

- **learning_rate = 보폭**이다. loss가 줄어드는 방향으로 가중치를 고칠 때 한 번에 얼마나 크게 고칠지를 정한다
  - 크면 최저점을 건너뛰고 그 주변을 튕겨 다닌다
  - 작으면 한 걸음이 작아서 epoch가 끝날 때까지 최저점에 닿지 못한다
- 후보를 파일마다 주석으로 나열해 두고 **하나씩 열어 가며 직접 비교**했다
  - `0.05 / 0.01 / 0.009 / 0.005 / 0.001(기본) / 0.00001`
  - 데이터에 따라 `0.0018`, `0.0015`, `0.0005`처럼 기본값 근처를 손으로 좁혀 간 것도 있다
- **바뀌는 건 compile 한 줄뿐**이다. 데이터도 모델도 그대로 두고 보폭만 바꿔 비교했다

#### 데이터마다 정답이 달랐다

| 방향 | 데이터 | 무슨 일이 있었나 |
|---|---|---|
| 좋아짐 | boston, ddareung, wine | 기본값 0.001로는 덜 내려간 상태에서 es가 걸리고 있었다 |
| 나빠짐 | california, digits, fashion | 보폭이 커서 최저점 주변을 튕겨 다녔다 |
| 변화 없음 | santander, bike | 스케일링 때도 반응이 없던 데이터. 보폭을 만져도 한계가 같다 |

- **wine**: acc는 1개 차이인데 **소요 시간이 115초 → 3초**로 줄었다. 보폭을 키우니 훨씬 적은 epoch 만에 최저점에 닿아 es가 일찍 걸린 것이다
- **digits**: `batch_size=4`로 아주 작게 주는 파일이다. 가중치를 고치는 횟수가 원래 많은데 보폭까지 키우니 0.9374 → 0.9068로 무너졌다
  - → **batch_size가 작을수록 learning_rate는 작게** 가야 한다
- **fashion**: 0.01을 주니 0.879 → 0.6219, 그것도 **Epoch 26에 조기 종료**됐다. val_acc가 오르지도 못한 채 patience 25를 넘긴 것이다
  - 학습이 안 되는데 조기 종료까지 빨라지는 것이 보폭이 너무 클 때의 전형적인 모습이다

### 2. ReduceLROnPlateau로 훈련 도중 learning_rate 줄이기 (keras52_ReduceLR01 ~ 15)

고정 learning_rate는 **한 값으로 두 가지를 다 할 수 없다.**

- 초반에는 크게 움직여 빨리 내려가야 한다
- 최저점 근처에서는 작게 움직여야 지나치지 않는다

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,    # val_loss가 20 epoch 동안 안 좋아지면
    factor=0.5,     # learning_rate에 0.5를 곱한다 (절반)
    verbose=1,
)
model.fit(..., callbacks=[es, rlr])
```

- **plateau = 고원**. val_loss가 더 이상 줄지 않고 평평해지는 구간을 말한다
- `factor=0.5` → 0.01 → 0.005 → 0.0025 → ... 로 절반씩 떨어진다
- `verbose=1`이면 줄어드는 순간 `ReduceLROnPlateau reducing learning rate to ...` 가 찍힌다
- **es와 역할이 다르다**
  - es : 더 안 좋아지면 **멈춘다**
  - rlr : 더 안 좋아지면 **보폭을 줄여 더 해 본다**
  - 그래서 `callbacks=[es, rlr]` 로 같이 넣는다 (mcp까지 쓰면 `[es, mcp, rlr]`)

#### patience를 어떻게 주느냐가 결과를 갈랐다

| 관계 | 무슨 일이 일어나나 | 이번에 해당한 파일 |
|---|---|---|
| `rlr < es` | 줄여 보고 → 그래도 안 되면 멈춘다 **(의도한 동작)** | 01(20/40), 08(20/50), 10(20/100), 11~14(20/25) |
| `rlr == es` | lr이 줄어드는 바로 그 epoch에 es도 같이 걸린다 | 02~07, 09 (둘 다 20) |
| `rlr > es` | **lr을 줄여 보기도 전에 훈련이 끝난다** | 15 man_woman (rlr 20 / es 10) |

- **digits**: 0.9068 → **0.9485**. 튕겨 다니던 것이 lr이 절반씩 줄면서 잡혔다. es patience가 100이라 줄여 볼 기회가 여러 번 왔다
- **fashion**: 0.6219 → **0.7825**. 0.16이나 올랐지만 증폭 기록 0.879에는 못 미친다
  - → 큰 lr로 한 번 망가진 출발은 뒤에서 줄여도 완전히는 되돌리지 못한다
- **ddareung**: RMSE 41.94 → **47.75로 나빠졌다**. 시작값을 0.0008로 이미 작게 잡아서 거기서 또 절반씩 줄어드니 최저점까지 가지 못했다
  - → ReduceLROnPlateau는 **"시작값을 넉넉히 주고 줄여 나간다"가 전제**다
- **cifar100**: 시작 lr이 0.01로 **똑같은데** 0.3144 → 0.3270으로 올랐다. 줄여 나간 것 자체가 효과를 낸 경우다
- **man_woman**: 0.8995 → 0.9067로 올랐지만 es patience가 10이라 **Epoch 24에 es가 먼저 걸렸다**. rlr이 한 번도 작동하지 못한 상태의 기록이라, 이 차이는 rlr 효과가 아니다

### 3. RNN(SimpleRNN) 입문 (keras54_RNN1)

| | 무엇을 보는가 |
|---|---|
| DNN | 컬럼끼리 순서가 없다 (키·몸무게·나이 순서를 바꿔도 같은 데이터) |
| CNN | 가로 / 세로로 옆에 붙어 있는 픽셀끼리의 관계 |
| **RNN** | **앞에서 뒤로 흐르는 순서** (주가, 날씨, 문장 등 시계열) |

- RNN은 계산한 결과를 **다음 칸으로 넘기면서** 같은 층을 timesteps 번 반복한다
- **데이터를 직접 잘라서 만든다**: 원본은 `[1,2,3,...,10]` 하나뿐이다
  - x = 앞 3칸, y = 그 다음 1칸으로 한 칸씩 밀어 가며 자른다 → `(7, 3)`, `(7,)`
- **RNN은 3차원을 받는다**: `(N, timesteps, features)`
  - N = 데이터 개수 (7)
  - timesteps = 한 묶음에서 몇 칸을 보는가 (3)
  - features = 한 칸에 값이 몇 개인가 (1)
  - `x.reshape(x.shape[0], x.shape[1], 1)` → `(7, 3, 1)`
- `input_shape`에는 **N을 빼고** `(timesteps, features)`만 적는다 (Dense의 `input_dim`과 같은 규칙)
- **RNN 출력은 2차원**이다. timesteps를 다 돌고 마지막 결과 하나만 내보내기 때문에 `(N, units)`가 된다
  - → CNN의 `Flatten` 같은 것이 필요 없고 **Dense와 바로 연결**된다
- 예측도 훈련 때와 같은 3차원으로 넣는다: `np.array([8,9,10]).reshape(1, 3, 1)`
- 결과 : **[[10.846822]]** — 정답 11에 못 미친다
  - 훈련 데이터의 y가 4 ~ 10까지뿐이라 모델은 10보다 큰 값을 본 적이 없다
  - 규칙(+1)을 이해한 게 아니라 본 범위 안에서 흉내 낸 것에 가깝다

### 4. SimpleRNN 파라미터 계산 (keras54_RNN2_summary)

Dense를 10층 쌓은 RNN1은 summary가 길어서, 계산을 눈으로 따라가려고 `SimpleRNN(5) + Dense(7) + Dense(1)`로 줄였다.

```
Param = (units × features) + (units × units) + units
      = (5 × 1)            + (5 × 5)         + 5     = 35
```

| 항 | 뜻 | 값 |
|---|---|:---:|
| units × features | 입력값을 받는 가중치 | 5 |
| **units × units** | **앞 칸의 결과를 다시 받는 가중치** ← RNN에만 있는 항 | 25 |
| units (bias) | 편향 | 5 |

- 가운데 항이 "앞에서 계산한 것을 다음 칸으로 넘긴다"는 **RNN의 정체**다
- **timesteps는 식에 없다**. 같은 가중치를 timesteps 번 돌려 쓰기 때문에 3칸이든 100칸이든 35개다
- `Output Shape`이 `(None, 5)` **2차원**으로 찍힌다 → timesteps 3이 사라진 것을 눈으로 확인

### 5. input_length / input_dim (keras54_RNN3_input_length)

```python
model.add(SimpleRNN(10, input_shape=(3, 1)))                 # 묶어서
model.add(SimpleRNN(units=10, input_length=3, input_dim=1))  # 나눠서 (같은 뜻)
model.add(SimpleRNN(units=10, input_dim=1, input_length=3))  # 순서도 바꿀 수 있다
```

- `input_length` = timesteps (몇 칸을 보는가)
- `input_dim` = features (한 칸에 값이 몇 개인가)
- Dense에서 쓰던 `input_dim`과 이름이 같다. RNN은 거기에 `input_length`가 하나 더 붙는 것이다
- 키워드로 이름을 붙여 넘기므로 **순서를 바꿔 써도 된다**
- 모델 구조는 RNN1과 완전히 같다. 표기법만 바꿔 본 파일이다

### 6. LSTM 파라미터는 RNN의 4배 (keras55_LSTM1_summary)

SimpleRNN은 앞 칸의 결과를 계속 곱해 가며 뒤로 넘긴다. timesteps가 길어지면 앞쪽에서 온 값이 점점 희미해져 사라진다(장기 기억 문제). LSTM은 **계속 들고 갈 값(cell state)** 을 따로 두고, 게이트로 얼마나 버리고 / 받아들이고 / 내보낼지를 스스로 정한다.

```
SimpleRNN 한 덩어리 = (units × features) + (units × units) + units
units=10, features=1 →  (10 × 1) + (10 × 10) + 10 = 120

LSTM 은 그 덩어리가 4개 (cell state 후보 + forget / input / output 게이트)
120 × 4 = 480
```

| 층 | Param | 계산 |
|---|:---:|---|
| `LSTM(10)` | 480 | `((10×1) + (10×10) + 10) × 4` |
| `Dense(7)` | 77 | `10 × 7 + 7` |
| `Dense(1)` | 8 | `7 × 1 + 1` |
| **Total** | **565** | |

- **같은 units면 항상 LSTM이 SimpleRNN의 4배**다
- 파라미터가 4배라는 건 계산도 그만큼 더 한다는 뜻이다 → 기억을 오래 들고 가는 대신 느려지는 것이 LSTM의 값이다
- `SimpleRNN`을 `LSTM`으로 **바꿔 끼우기만** 하면 된다. 입출력 모양이 같아서 뒤 코드는 그대로다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras52_optimizer01_california.py` | `Adam(learning_rate=0.01)` 적용, RMSE 0.5023 → 0.5290 |
| `keras52_optimizer02_diabetes.py` | lr 0.005, loss 3232 → 3203 |
| `keras52_optimizer03_boston.py` | lr 0.01, loss 21.35 → 19.10 |
| `keras52_optimizer04_dacon_ddareung.py` | lr 0.0018, RMSE 43.25 → 41.94 |
| `keras52_optimizer05_kaggle_bike.py` | lr 0.01, RMSE 148.52 → 146.92 |
| `keras52_optimizer06_cancer.py` | lr 0.01, acc 0.9649 → 0.9766 |
| `keras52_optimizer07_santander.py` | lr 0.0015, acc 0.9116 → 0.9116 (변화 없음) |
| `keras52_optimizer08_wine.py` | lr 0.01, acc 0.963 → 0.981 / 115초 → 3초 |
| `keras52_optimizer09_fetch_covtype.py` | lr 0.0005, acc 0.9426 → 0.9453 |
| `keras52_optimizer10_digits.py` | lr 0.01, acc 0.9374 → 0.9068 (batch_size=4에 보폭이 너무 컸다) |
| `keras52_optimizer11_mnist.py` | lr 0.01, acc 0.879 → 0.7108 |
| `keras52_optimizer12_fashion.py` | lr 0.01, acc 0.879 → 0.6219 (Epoch 26 조기 종료) |
| `keras52_optimizer13_cifar10.py` | lr 0.005, acc 0.5433 → 0.5308 |
| `keras52_optimizer14_cifar100.py` | lr 0.01, acc 0.3326 → 0.3144 |
| `keras52_optimizer15_man_woman.py` | lr 0.0018, acc 0.9082 → 0.8995 |
| `keras52_ReduceLR01_california.py` | `ReduceLROnPlateau(patience=20, factor=0.5)` 추가, RMSE 0.5290 → 0.5148 |
| `keras52_ReduceLR02_diabetes.py` | 시작 lr 0.0001, loss 3203 → 3296 |
| `keras52_ReduceLR03_boston.py` | 시작 lr 0.005, loss 19.10 → 18.54 (boston 최저 기록) |
| `keras52_ReduceLR04_dacon_ddareung.py` | 시작 lr 0.0008, RMSE 41.94 → 47.75 (시작값이 너무 작았다) |
| `keras52_ReduceLR05_kaggle_bike.py` | 시작 lr 0.01, RMSE 146.92 → 146.03 |
| `keras52_ReduceLR06_cancer.py` | 시작 lr 0.0006, acc 0.9766 → 0.9708 |
| `keras52_ReduceLR07_santander.py` | 시작 lr 0.0015, acc 0.9116 → 0.9105 |
| `keras52_ReduceLR08_wine.py` | 시작 lr 0.0005, acc 0.981 → 0.981 / 3초 → 12초 |
| `keras52_ReduceLR09_fetch_covtype.py` | 시작 lr 0.0005, acc 0.9453 → 0.9479 (covtype 최고 기록) |
| `keras52_ReduceLR10_digits.py` | 시작 lr 0.005, acc 0.9068 → 0.9485 |
| `keras52_ReduceLR11_mnist.py` | 시작 lr 0.005, acc 0.7108 → 0.7942 |
| `keras52_ReduceLR12_fashion.py` | 시작 lr 0.005, acc 0.6219 → 0.7825 |
| `keras52_ReduceLR13_cifar10.py` | 시작 lr 0.0005, acc 0.5308 → 0.5347 |
| `keras52_ReduceLR14_cifar100.py` | 시작 lr 0.01(동일), acc 0.3144 → 0.3270 |
| `keras52_ReduceLR15_man_woman.py` | 시작 lr 0.0018, acc 0.8995 → 0.9067 (es patience=10이라 rlr 미작동) |
| `keras54_RNN1.py` | SimpleRNN 입문, `(7,3,1)` 3차원 입력, `[8,9,10]` → 10.846822 |
| `keras54_RNN2_summary.py` | SimpleRNN 파라미터 계산 (units 5 → 35), Output Shape 2차원 확인 |
| `keras54_RNN3_input_length.py` | `input_length` / `input_dim`으로 나눠 쓰는 표기법 |
| `keras55_LSTM1_summary.py` | LSTM 파라미터 480 = SimpleRNN 120의 4배 확인 |
| `keras1/dataset.txt` | 11 mnist ~ 15 man_woman 추가 (이미지 5종을 번호 체계에 합류) (수정) |
| `keras27_Scaler01_california.py` | 현재 코드로 재실행해 기록 갱신 (RMSE 0.5023) (수정) |
| `docs/Day18.md` | Day18 학습 기록 신규 작성 |
| `docs/Day17.md` | 하단 nav에 Day18 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조 / 진행도(18일, 23%) 갱신 (수정) |

---

## 📊 실행 결과

### learning_rate 고정 → ReduceLROnPlateau (15개 데이터셋)

| # | 파일 | 지표 | 이전 기록 | lr 지정 | ReduceLROnPlateau |
|:---:|---|:---:|:---:|:---:|:---:|
| 01 | california | RMSE | 0.5023 (기본 0.001) | 0.5290 (0.01) | **0.5148** (0.01) |
| 02 | diabetes | loss | 3232.84 | 3203.10 (0.005) | 3296.06 (0.0001) |
| 03 | boston | loss | 21.35 | 19.10 (0.01) | **18.54** (0.005) |
| 04 | ddareung | RMSE | 43.25 | **41.94** (0.0018) | 47.75 (0.0008) |
| 05 | kaggle_bike | RMSE | 148.52 | 146.92 (0.01) | **146.03** (0.01) |
| 06 | cancer | acc | 0.9649 | **0.9766** (0.01) | 0.9708 (0.0006) |
| 07 | santander | acc | 0.9116 | 0.9116 (0.0015) | 0.9105 (0.0015) |
| 08 | wine | acc | 0.963 | **0.981** (0.01) | **0.981** (0.0005) |
| 09 | fetch_covtype | acc | 0.9426 | 0.9453 (0.0005) | **0.9479** (0.0005) |
| 10 | digits | acc | 0.9374 | 0.9068 (0.01) | **0.9485** (0.005) |
| 11 | 11_mnist | acc | 0.879 | 0.7108 (0.01) | **0.7942** (0.005) |
| 12 | fashion | acc | 0.879 | 0.6219 (0.01) | 0.7825 (0.005) |
| 13 | cifar10 | acc | 0.5433 | 0.5308 (0.005) | 0.5347 (0.0005) |
| 14 | cifar100 | acc | 0.3326 | 0.3144 (0.01) | 0.3270 (0.01) |
| 15 | man_woman | acc | 0.9082 | 0.8995 (0.0018) | 0.9067 (0.0018) |

괄호 안은 적용한 learning_rate(ReduceLR은 시작값). **이전 기록**은 정형 데이터 01~10은 Day10의 스케일러 최종 기록, 이미지 11~15는 Day17의 증폭 기록이다.

**읽는 법**
- lr 지정만으로 좋아진 것은 **boston / ddareung / cancer / wine** 4개뿐이고, **digits / fashion**은 오히려 크게 나빠졌다 → learning_rate는 키운다고 좋아지는 값이 아니다
- ReduceLROnPlateau를 얹으면 **15개 중 10개**가 lr 고정보다 좋아졌다. 특히 lr을 너무 크게 줘서 망가졌던 digits(+0.042) / fashion(+0.16) / 11번(+0.083)에서 회복 폭이 컸다
- 나빠진 **diabetes / ddareung**은 둘 다 시작 lr을 0.0001 / 0.0008로 작게 잡은 경우다

### 소요 시간이 함께 바뀐 경우

| 데이터 | 기록 | 시간 | 이유 |
|---|:---:|---:|---|
| wine | 0.963 → 0.981 | 115초 → **3초** | 보폭을 키우니 적은 epoch 만에 수렴해 es가 일찍 걸렸다 |
| fetch_covtype | 0.9426 → 0.9453 → 0.9479 | 664 → 1231 → **1828초** | 점수 0.005를 시간 1100초와 바꾼 셈 |
| man_woman | 0.8995 → 0.9067 | 653 → 538초 | es patience=10이라 Epoch 24에 조기 종료 |

### RNN / LSTM

| 파일 | 모델 | 결과 |
|---|---|---|
| `keras54_RNN1.py` | SimpleRNN(10) + Dense 10층 | loss 5.95e-05, `[8,9,10]` → **10.846822** (정답 11) |
| `keras54_RNN2_summary.py` | SimpleRNN(5) + Dense(7) + Dense(1) | Total params **85** (RNN 35 / Dense 42 / Dense 8) |
| `keras55_LSTM1_summary.py` | LSTM(10) + Dense(7) + Dense(1) | Total params **565** (LSTM 480 / Dense 77 / Dense 8) |

---

## 💻 핵심 개념

### learning_rate 직접 지정

```python
from tensorflow.keras.optimizers import Adam

learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
```

문자열 `'adam'`으로 주면 0.001로 고정된다. **객체로 만들어 넘겨야** 바꿀 수 있다.

### ReduceLROnPlateau + EarlyStopping

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=40,                # rlr 보다 길게
    restore_best_weights=True,
    verbose=1,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,                # es 보다 짧게 -> 줄여 보고 나서 멈춘다
    factor=0.5,                 # 0.01 -> 0.005 -> 0.0025 ...
    verbose=1,
)

model.fit(x_train, y_train,
          epochs=500, batch_size=32,
          validation_split=0.2,
          callbacks=[es, rlr],  # mcp 까지 쓰면 [es, mcp, rlr]
          verbose=1,)
```

### 시계열 데이터 만들기 + SimpleRNN

```python
datasets = np.array([1,2,3,4,5,6,7,8,9,10])     # 원본은 이어진 숫자 하나

x = np.array([[1,2,3],      # 한 칸씩 밀면서 3개 묶음
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])
y = np.array([4,5,6,7,8,9,10])                  # 각 묶음의 '다음 값'

print(x.shape, y.shape)                         # (7, 3) (7,)
x = x.reshape(x.shape[0], x.shape[1], 1)        # (7, 3, 1) = (N, timesteps, features)

model = Sequential()
model.add(SimpleRNN(10, input_shape=(3, 1)))    # input_shape 에는 N 을 빼고 적는다
model.add(Dense(512, activation='relu'))        # RNN 출력이 2차원이라 바로 연결
...
model.add(Dense(1))                             # 숫자 하나를 맞히는 회귀

x_predict = np.array([8,9,10]).reshape(1,3,1)   # 예측도 같은 3차원으로
y_predict = model.predict(x_predict)
```

### 입력 모양 표기 두 가지

```python
model.add(SimpleRNN(10, input_shape=(3, 1)))                 # 묶어서
model.add(SimpleRNN(units=10, input_length=3, input_dim=1))  # 나눠서 (같은 뜻)
model.add(SimpleRNN(units=10, input_dim=1, input_length=3))  # 순서 상관 없음
```

### 파라미터 계산식

```
SimpleRNN = (units × features) + (units × units) + units
            └ 입력 받기        └ 앞 칸 결과 받기  └ bias
            (timesteps 는 식에 없다 - 같은 가중치를 돌려 쓴다)

LSTM      = SimpleRNN × 4
            (cell state 후보 + forget / input / output 게이트)

units=10, features=1
  SimpleRNN : (10×1) + (10×10) + 10 = 120
  LSTM      : 120 × 4               = 480
```

---

## 성과

- `optimizer='adam'` 문자열 뒤에 숨어 있던 learning_rate를 꺼내 **15개 데이터셋 전부에 직접 지정**하고 기존 기록과 비교
- 데이터마다 맞는 learning_rate가 다르다는 것을 실제 수치로 확인 (boston·ddareung·wine은 개선, digits·fashion은 악화)
- `batch_size=4`인 digits가 큰 lr에서 무너지는 것을 보고 **batch_size와 learning_rate의 관계** 파악
- ReduceLROnPlateau를 15개 데이터셋에 적용해 **10개에서 lr 고정보다 개선**, digits 0.9068 → 0.9485 / fashion 0.6219 → 0.7825
- `es.patience`와 `rlr.patience`의 대소 관계에 따라 rlr이 작동조차 못 할 수 있다는 것을 man_woman에서 확인
- SimpleRNN으로 시계열 데이터를 직접 잘라 만들고 `(N, timesteps, features)` 3차원 입력을 익힘
- SimpleRNN / LSTM의 파라미터를 손으로 계산해 `model.summary()`와 대조 (35 / 480)
- `keras1/dataset.txt`에 11 mnist ~ 15 man_woman을 추가해 이미지 데이터셋까지 번호 체계에 합류

---

## 💡 주요 학습 포인트

1. **`optimizer='adam'`은 learning_rate 0.001 고정이다**: 바꾸려면 `Adam(learning_rate=...)` 객체로 넘겨야 한다
2. **learning_rate는 보폭이다**: 크면 최저점을 건너뛰고, 작으면 최저점까지 못 간다
3. **키운다고 좋아지지 않는다**: 15개 중 좋아진 건 4개뿐이었다
4. **batch_size가 작으면 learning_rate도 작게**: digits(`batch_size=4`)가 0.01에서 0.9374 → 0.9068로 무너졌다
5. **lr이 너무 크면 조기 종료도 빨라진다**: fashion은 val_acc가 오르지도 못한 채 Epoch 26에 멈췄다
6. **점수뿐 아니라 시간도 같이 본다**: wine은 acc 1개 차이에 시간이 115초 → 3초로 줄었다
7. **고정 lr로는 두 가지를 다 못 한다**: 초반엔 크게, 최저점 근처에선 작게 움직여야 한다
8. **ReduceLROnPlateau는 val_loss가 평평해지면 lr을 줄인다**: `factor=0.5`면 절반씩
9. **es와 rlr은 역할이 다르다**: es는 멈추고, rlr은 보폭을 줄여 더 해 본다
10. **`rlr.patience < es.patience`**: 같으면 동시에 걸리고, 크면 rlr이 작동도 못 한다 (man_woman)
11. **rlr은 시작값을 넉넉히 줘야 한다**: ddareung은 0.0008로 시작해 줄일 여지가 없어 RMSE가 41.94 → 47.75가 됐다
12. **한 번 망가진 출발은 완전히 되돌아오지 않는다**: fashion 0.6219 → 0.7825지만 증폭 기록 0.879에는 못 미친다
13. **안 변하는 데이터는 안 변한다**: santander는 스케일링·lr·rlr 무엇을 해도 0.91에서 멈춘다 → 모델/데이터 쪽 문제라는 신호
14. **RNN은 순서를 본다**: DNN은 컬럼 순서가 없고, CNN은 옆 픽셀을 보고, RNN은 앞에서 뒤로 흐르는 순서를 본다
15. **RNN 입력은 3차원 `(N, timesteps, features)`**: `input_shape`에는 N을 빼고 적는다
16. **RNN 출력은 2차원**: 마지막 결과 하나만 내보내므로 Flatten 없이 Dense와 바로 연결된다
17. **RNN 파라미터에 timesteps는 없다**: 같은 가중치를 timesteps 번 돌려 쓴다
18. **`units × units` 항이 RNN의 정체다**: 앞 칸의 결과를 다시 받는 가중치
19. **`input_length` + `input_dim`은 `input_shape`를 나눠 쓴 것**: 순서도 바꿀 수 있다
20. **LSTM은 SimpleRNN의 정확히 4배**: 게이트가 4덩어리라서. 오래 기억하는 대신 느려진다
21. **본 적 없는 범위는 못 맞힌다**: y가 4~10뿐이라 `[8,9,10]`의 답이 11이 아니라 10.85가 나왔다

---

[⬅️ Day17](Day17.md) · [🏠 전체 목차](../README.md) · [Day19 ➡️](Day19.md)
