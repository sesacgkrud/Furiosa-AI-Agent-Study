# Day20 - split 함수로 여러 값 예측, return_sequences · Flatten, Jena 기후 시계열 예측

**학습 기간:** 2026-09-29

> Day19에서는 split_x 함수로 작은 시계열(10개)을 잘라 다음 값 하나를 맞혔다. Day20은 1 ~ 100까지 **긴 시계열**을 잘라서 101 ~ 106 **여섯 개**를 한 번에 예측하고, 1차원 데이터를 `(N, 5, 2)`로 바꿔 **다음 시점의 값 2개**를 맞혔다. 그다음 RNN 층을 여러 개 쌓을 때 필요한 **return_sequences**와 RNN 출력을 **Flatten**으로 펴서 Dense로 넘기는 방법을 익혔다. 마지막으로 실제 기상 데이터인 **Kaggle Jena Climate**(42만 행)로 "하루 전 144개를 보고 다음 하루 144개를 맞히는" 시계열 예측을 했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 긴 시계열 split | 1 ~ 100을 `size=6`으로 자르면 `(95, 6)` → x `(95, 5, 1)` / y `(95,)` |
| 예측 데이터 split | 예측 데이터에는 y 칸이 없으니 `size - 1`(=5)로 자른다 → 96 ~ 105에서 6묶음 |
| 여러 값 한 번에 예측 | 예측 묶음이 6개면 `predict` 결과도 6개 → 101 ~ 106 |
| 1차원 → `(N, 5, 2)` | `a.reshape(-1, 2)` 로 `(50, 2)` 만든 뒤 split → `(45, 6, 2)` |
| 값 2개 맞히기 | `y = bbb[:, -1]` → `(45, 2)`, 출력층 `Dense(2)` |
| return_sequences | `False`(기본) = 마지막 시점만 2차원 / `True` = 모든 시점 3차원 |
| RNN 쌓기 | 뒤에 RNN이 또 오면 앞 RNN은 `return_sequences=True` (아니면 ndim 에러) |
| RNN + Flatten | `(None, 3, 10)` → Flatten → `(None, 30)` → 모든 시점 출력을 Dense로 |
| LSTM 파라미터 | `4 × units × (feature + units + 1)`, 앞 층 units가 다음 층 feature |
| Jena 구간 나누기 | 훈련 `[:-288]` / 예측용 x `[-288:-144]` / 정답 `[-144:]` |
| x, y 144칸 어긋나기 | `x_data = [:-288]`, `y_data = [144:-144]` → "하루 전 → 다음 하루" 짝 |
| 스케일링 위치 | split 전 **2차원**에서 컬럼별로, fit은 훈련 구간만 |
| 메모리 | `(420120, 144, 13)` float32 ≈ 3.1GB → npy를 float32로 저장 |
| 출력 144개 | `Dense(144)` 로 다음 하루 144개 시점을 한 번에 |
| y 컬럼 바꾸기 | y를 `wd (deg)` → `T (degC)`로 바꾸고, 기존 코드는 주석 + 이유 |

---

## 📖 핵심 학습 내용

### 1. 1 ~ 100 시계열로 101 ~ 106 예측 (keras56_split3)

```python
a = np.array(range(1, 101))            # 1 ~ 100
x_predict = np.array(range(96, 106))   # 96 ~ 105 → 이걸로 101 ~ 106 을 찾자
size = 6                               # 한 묶음 = x 5개 + y 1개

bbb = split_x(a, size)                 # (95, 6)   100 - 6 + 1 = 95 묶음
x = bbb[:, :-1]                        # (95, 5)
y = bbb[:, -1]                         # (95,)
x = x.reshape(x.shape[0], x.shape[1], 1)   # (95, 5, 1)
```

- 목표 : loss 0.1 이하, `[101, 102, 103, 104, 105, 106]`의 근사치
- 예측 데이터도 **훈련 x와 같은 모양**으로 잘라야 한다

```python
x_predict = split_x(x_predict, size - 1)   # (6, 5)  10 - 5 + 1 = 6 묶음
# [[ 96  97  98  99 100]  → 101
#  [ 97  98  99 100 101]  → 102
#  ...
#  [101 102 103 104 105]] → 106
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)   # (6, 5, 1)
```

- `size`가 아니라 **`size - 1`** 로 자르는 이유 : 예측할 데이터에는 y 칸이 없다
- 예측 묶음이 6개라 `predict` 결과도 **6개**가 한 번에 나온다
- Day19에서 배운 대로 `x / 100` 스케일링 + `activation='linear'` 적용 (예측할 101 ~ 106은 훈련 y 6 ~ 100 범위 밖)

```python
model.add(LSTM(64, input_shape=(5, 1), activation='linear'))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

es = EarlyStopping(monitor='loss', mode='min', patience=50, restore_best_weights=True)
model.fit(x, y, epochs=1000, batch_size=4, callbacks=[es])
```

- 데이터가 95개뿐이라 validation을 떼지 않고 **훈련 loss**로 EarlyStopping
  - 시계열에서 `validation_split`은 뒤쪽 20%(가장 큰 값들)를 떼어가서 훈련 범위가 더 좁아진다
- 결과 : loss 0.00307, **[100.97, 101.98, 102.98, 104.01, 105.01, 106.01]**

### 2. 1차원 데이터를 (N, 5, 2)로 바꿔 값 2개 예측 (keras56_split4)

```python
a = a.reshape(-1, 2)           # (100,) → (50, 2)   [[1 2] [3 4] ... [99 100]]
bbb = split_x(a, size)         # (45, 6, 2)          50 - 6 + 1 = 45 묶음

x = bbb[:, :-1]                # (45, 5, 2)  [[1 2] [3 4] [5 6] [7 8] [9 10]]
y = bbb[:, -1]                 # (45, 2)     [11 12]
```

- `(N, 10, 1)` → `(N, 5, 2)` : 한 시점에 값 1개씩 10번 보던 것을, 한 시점에 값 2개씩 5번 보도록 바꿨다
- `-1`은 "나머지는 알아서 계산" → `100 / 2 = 50`
- split_x는 **행 기준**으로 자르므로 2차원도 열 개수를 유지한 채 잘린다 → 이미 3차원이라 reshape 불필요
- y가 `(45, 2)` → 출력층도 **`Dense(2)`** (다음 시점의 값 2개를 한 번에)
  - 한 값만 맞히려면 `y = bbb[:, -1, -1]` → `(45,)`, `Dense(1)`

```python
x_predict = x_predict.reshape(-1, 2)          # (5, 2)
x_predict = split_x(x_predict, size - 1)      # (1, 5, 2)
# [[[ 96  97] [ 98  99] [100 101] [102 103] [104 105]]] → 정답 [106 107]

model.add(LSTM(64, input_shape=(5, 2), activation='linear'))
...
model.add(Dense(2))
```

- 결과 : loss 0.01038, **[[105.77, 106.86]]** (정답 [106, 107])

### 3. return_sequences로 RNN 여러 층 쌓기 (keras57_01_return_sequence)

| 옵션 | 내보내는 출력 | 모양 |
|---|---|---|
| `return_sequences=False` (기본값) | 마지막 시점의 출력 1개 | 2차원 `(batch, units)` |
| `return_sequences=True` | 모든 시점의 출력 | 3차원 `(batch, timesteps, units)` |

- RNN 층(SimpleRNN / LSTM / GRU)은 **3차원만** 입력으로 받는다
- 그래서 LSTM 뒤에 LSTM을 또 쌓으려면 앞 LSTM이 `return_sequences=True`로 3차원을 내보내야 한다
- 없이 쌓으면 에러

```
ValueError: Input 0 of layer "lstm_1" is incompatible with the layer:
            expected ndim=3, found ndim=2. Full shape received: (None, 10)
```

```python
model.add(LSTM(10, input_shape=(3, 1), return_sequences=True))  # (None, 3, 1)  → (None, 3, 10)
model.add(LSTM(8))                                              # (None, 3, 10) → (None, 8)
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
```

| 층 | Output Shape | Param |
|---|:---:|:---:|
| `LSTM(10, return_sequences=True)` | `(None, 3, 10)` | 480 |
| `LSTM(8)` | `(None, 8)` | 608 |
| `Dense(16)` | `(None, 16)` | 144 |
| `Dense(8)` | `(None, 8)` | 136 |
| `Dense(1)` | `(None, 1)` | 9 |
| **Total** | | **1,377** |

- LSTM 파라미터 = `4 × units × (feature + units + 1)`
  - `lstm` : `4 × 10 × (1 + 10 + 1)` = 480
  - `lstm_1` : `4 × 8 × (10 + 8 + 1)` = 608 ← **앞 LSTM의 units(10)가 이 층의 feature**
- `return_sequences=True`는 출력 **모양**만 바꾸고 파라미터 개수는 그대로다 (시점마다 같은 가중치를 반복해서 쓴다)
- LSTM을 여러 층 쌓는다고 항상 좋아지지는 않는다. 보통 RNN 1 ~ 2층 + Dense 조합
- 결과 : loss 0.00469, `[50, 60, 70]` → **[[72.31]]** (정답 80)
  - 80은 훈련 y 최댓값(70)보다 큰 **범위 밖** 값이라 못 미친다
  - 이 데이터는 앞 10개는 +1, 뒤 3개는 +10 규칙이라 전부 linear로 바꾸면 오히려 100 이상으로 튄다

### 4. RNN 출력을 Flatten으로 펴서 Dense로 (keras57_02_rnn_flatten)

```python
model.add(LSTM(10, input_shape=(3, 1), return_sequences=True))  # (None, 3, 1)  → (None, 3, 10)
model.add(Flatten())                                            # (None, 3, 10) → (None, 30)
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
```

| 층 | Output Shape | Param |
|---|:---:|:---:|
| `LSTM(10, return_sequences=True)` | `(None, 3, 10)` | 480 |
| `Flatten` | `(None, 30)` | 0 |
| `Dense(16)` | `(None, 16)` | 496 (`30 × 16 + 16`) |
| `Dense(8)` | `(None, 8)` | 136 |
| `Dense(1)` | `(None, 1)` | 9 |
| **Total** | | **1,121** |

- `return_sequences=True`의 3차원 출력을 CNN 때처럼 **Flatten**으로 펴서 Dense에 넘길 수 있다
- keras57_01과 비교
  - `return_sequences=False` : **마지막 시점** 출력만 Dense로
  - `return_sequences=True` + Flatten : **모든 시점** 출력을 Dense로 → 중간 시점 정보도 쓴다
- Flatten은 모양만 바꾸므로 파라미터 0. 3차원 이상일 때만 의미가 있다 (이미 2차원이면 변화 없음)
- 결과 : loss 0.2415, **[[71.75]]** (정답 80) → 구조의 차이일 뿐, 범위 밖 예측을 해결하는 방법은 아니다

### 5. Kaggle Jena Climate - 데이터 자르기와 npy 저장 (keras58_kaggle_jena1)

- 데이터 : 독일 예나 기상 관측, 2009.01.01 00:10 ~ 2017.01.01 00:00, **10분 간격**, 14개 컬럼
  - https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
- 실습 : y는 `wd (deg)`(풍향), 맞출 구간은 2016.12.31 00:10 ~ 2017.01.01 00:00 **144개** (10분 × 144 = 24시간), 이 144개는 훈련에 사용하지 않는다

```python
datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)   # (420551, 14)
# index_col=0 : 'Date Time' 을 인덱스로 → 나머지 14개 컬럼만 데이터
print(datasets.isna().sum().sum())   # 0 → 결측치 없음
```

전체를 뒤에서부터 세 구간으로 나눴다 (N = 420551).

```
[0 ................................ N-288]  [N-288 ~ N-144]  [N-144 ~ N]
└──────── 훈련에 쓰는 구간 ────────┘  └ 예측용 x 144개 ┘  └ 정답 y 144개 ┘
```

```python
y_cor     = datasets[-144:]['wd (deg)']                          # (144,)       정답 (채점용)
x_predict = datasets[-288:-144].drop(['wd (deg)'], axis=1)       # (144, 13)    예측용 x
x_data    = datasets[:-288].drop(['wd (deg)'], axis=1)           # (420263, 13)
y_data    = datasets[144:-144]['wd (deg)']                       # (420263,)
```

- "직전 144개(하루)를 보고 → 다음 144개(하루)를 맞힌다"로 설계
- `y_data`는 `x_data`보다 **144칸 뒤**에서 시작한다
  - x 묶음 i : `i ~ i+143` 행 / y 묶음 i : `i+144 ~ i+287` 행 → "하루 전 → 다음 하루"로 짝이 맞는다
  - `x_data`와 `y_data` 길이가 같아야 split 후 묶음 개수가 맞는다
- csv를 매번 읽지 않도록 **npy로 저장**, `float32`로 바꿔 용량을 절반으로 줄였다

```python
np.save(np_path + 'keras58_x_data.npy', arr=x_data.values.astype(np.float32))
```

**y를 기온(`T (degC)`)으로 바꾸기**

기존 코드는 지우지 않고 주석 처리한 뒤, 왜 주석 처리했는지 이유를 적고 바꿨다.

```python
# y_cor = datasets[-144:]['wd (deg)']
#   [주석 이유] 정답 컬럼이 풍향(wd) 에서 기온(T) 으로 바뀌었다 → 마지막 144개의 기온을 정답으로 잡는다
y_cor = datasets[-144:]['T (degC)']

# x_predict = datasets[-288:-144].drop(['wd (deg)'], axis=1)
#   [주석 이유] y 컬럼이 T 로 바뀌었으니 x 에서 빼야 하는 컬럼도 T 로 바꿔야 한다
#               wd 를 그대로 빼면 x 에 T(=y) 가 남고, 풍향 정보는 쓸데없이 버려진다
x_predict = datasets[-288:-144].drop(['T (degC)'], axis=1)
```

- 자르는 구간(`[:-288]`, `[144:-144]`)은 그대로라 shape도 그대로다. 이제 `wd`는 x의 13개 컬럼 중 하나가 된다

### 6. Jena Climate - split, 스케일링, 훈련, 예측 (keras58_kaggle_jena2)

```python
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)       # fit 은 훈련 구간으로만
x_predict = scaler.transform(x_predict)     # 예측용 x 는 transform 만

x = split_x(x_data, 144)    # (420120, 144, 13)
y = split_x(y_data, 144)    # (420120, 144)
x_predict = x_predict.reshape(1, 144, 13)
```

- 스케일링은 split **전 2차원** 상태에서 한다
  - 컬럼마다 단위가 다르다 (p ≈ 1000, rho ≈ 1300, wv ≈ 0 ~ 10)
  - 3차원에서 하려면 `reshape(-1, 13)` → 스케일링 → 다시 reshape 해야 하고 메모리도 훨씬 많이 쓴다
  - y는 스케일링하지 않는다 → 예측값이 바로 원래 단위로 나온다
- y도 144개씩 묶어 **다음 하루 144개를 한 번에** 맞힌다
- 메모리 : `420120 × 144 × 13 × 4byte` ≈ **3.1GB** (float64였으면 약 6.3GB)

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)
del x, y
# (336096, 144, 13) (84024, 144, 13) / (336096, 144) (84024, 144)
```

- `shuffle`은 **묶음 단위**로 섞는다. 묶음 안의 144개 시점 순서는 그대로라 시계열 순서가 깨지지 않는다

```python
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'   # tensorflow import 전에

model.add(LSTM(64, input_shape=(144, 13)))   # activation 기본값(tanh) → GPU cuDNN LSTM
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(144))                        # 출력 144개 = 다음 하루

es  = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True,
                      filepath='./_save/keras58/keras58_jena.hdf5')
model.fit(x_train, y_train, epochs=100, batch_size=1024, validation_split=0.2, callbacks=[es, mcp])
```

- 데이터가 33만 개라 `batch_size=1024`로 키웠다
- 훈련이 오래 걸려서 val_loss가 가장 좋았던 모델을 ModelCheckpoint로 남겼다
- 채점은 **RMSE** : mse는 단위가 제곱이라 루트를 씌워 원래 단위로 몇 정도 틀렸는지 본다

```python
y_predict = model.predict(x_predict).reshape(-1)          # (1, 144) → (144,)
rmse = np.sqrt(mean_squared_error(y_cor, y_predict))
```

- 결과 (y = 풍향, 64 epoch에서 EarlyStopping, 314초)
  - test loss 5648.32 (루트 약 75도)
  - 2016.12.31 00:10 ~ 2017.01.01 00:00 풍향 **RMSE 56.45도**
- 풍향은 0도와 360도가 같은 방향(북쪽)인 **원형 값**이라, mse로는 359도와 1도를 358도 차이로 계산한다 → 원래 맞히기 어려운 타깃

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras56_split3.py` | 1 ~ 100을 `size=6`으로 split → `(95, 5, 1)`, 96 ~ 105를 5씩 잘라 101 ~ 106 예측 → 100.97 ~ 106.01 |
| `keras56_split4.py` | `reshape(-1, 2)` 후 split → `(45, 5, 2)`, y `(45, 2)` + `Dense(2)`, [106, 107] 예측 → [105.77, 106.86] |
| `keras57_01_return_sequence.py` | `return_sequences=True`로 LSTM 2층 쌓기, 파라미터 480 / 608, 80 예측 → 72.31 |
| `keras57_02_rnn_flatten.py` | LSTM(`return_sequences=True`) + Flatten `(None, 30)` → Dense, 80 예측 → 71.75 |
| `keras58_kaggle_jena1..py` | Jena 데이터를 훈련 / 예측용 x / 정답 구간으로 잘라 float32 npy 저장, y를 `T (degC)`로 변경 |
| `keras58_kaggle_jena2.py` | 2차원 스케일링 → split `(420120, 144, 13)` → LSTM → `Dense(144)`, 풍향 RMSE 56.45 |
| `docs/Day20.md` | Day20 학습 기록 신규 작성 |
| `docs/Day19.md` | 하단 nav에 Day20 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조 정리 / 진행도(20일, 25%) 갱신 (수정) |
| `.gitignore` | `_data/kaggle_jena/` 추가 (수정) |

---

## 📊 실행 결과

### split 시계열 예측

| 파일 | 입력 | 정답 | 예측 | loss |
|---|---|:---:|:---:|:---:|
| `keras56_split3.py` | 96 ~ 105를 5개씩 6묶음 | 101 ~ 106 | **100.97, 101.98, 102.98, 104.01, 105.01, 106.01** | 0.00307 |
| `keras56_split4.py` | `[[96 97] [98 99] [100 101] [102 103] [104 105]]` | [106, 107] | **[105.77, 106.86]** | 0.01038 |

### return_sequences / Flatten 비교

| 파일 | 구조 | Total params | 80 예측 | loss |
|---|---|:---:|:---:|:---:|
| `keras57_01_return_sequence.py` | LSTM(rs=True) → LSTM → Dense | 1,377 | 72.31 | 0.00469 |
| `keras57_02_rnn_flatten.py` | LSTM(rs=True) → Flatten → Dense | 1,121 | 71.75 | 0.2415 |

**읽는 법**
- 두 구조 모두 80에 못 미친다 → 80은 훈련 y 최댓값(70) 밖의 값이다
- return_sequences / Flatten은 구조의 차이이지 범위 밖 예측을 해결하는 방법은 아니다

### Jena Climate (y = 풍향)

| 항목 | 값 |
|---|---|
| x / y | `(420120, 144, 13)` / `(420120, 144)` |
| train / test | 336096 / 84024 묶음 |
| 훈련 | 64 epoch에서 EarlyStopping, 314.54초 |
| test loss (mse) | 5648.32 |
| 마지막 하루 풍향 RMSE | **56.45도** |

---

## 💻 핵심 개념

### 예측 데이터는 size - 1 로 자른다

```python
size = 6                                   # x 5개 + y 1개
bbb = split_x(a, size)                     # 훈련 : (95, 6) → x (95, 5) / y (95,)
x_predict = split_x(x_predict, size - 1)   # 예측 : y 칸이 없으니 5개씩 → (6, 5)
```

### 1차원을 feature 2개로 바꾸기

```python
a = a.reshape(-1, 2)       # (100,) → (50, 2)
bbb = split_x(a, 6)        # (45, 6, 2)
x = bbb[:, :-1]            # (45, 5, 2) → input_shape=(5, 2)
y = bbb[:, -1]             # (45, 2)    → Dense(2)
```

### return_sequences

```
LSTM(10)                          : (None, 3, 1) → (None, 10)       마지막 시점만 (2차원)
LSTM(10, return_sequences=True)   : (None, 3, 1) → (None, 3, 10)    모든 시점 (3차원)

RNN → RNN     : 앞 RNN 은 return_sequences=True
RNN → Dense   : 기본값(False) 로 2차원, 또는 True + Flatten
```

### LSTM 파라미터 (여러 층)

```
LSTM = 4 × units × (feature + units + 1)

LSTM(10), feature=1  : 4 × 10 × (1 + 10 + 1)  = 480
LSTM(8),  feature=10 : 4 × 8  × (10 + 8 + 1)  = 608   ← 앞 층 units 가 feature
```

### Jena 구간 나누기

```python
y_cor     = datasets[-144:]['T (degC)']                      # 정답 144개 (훈련 X)
x_predict = datasets[-288:-144].drop(['T (degC)'], axis=1)   # 정답 바로 앞 하루
x_data    = datasets[:-288].drop(['T (degC)'], axis=1)       # 훈련 x
y_data    = datasets[144:-144]['T (degC)']                   # 훈련 y (x 보다 144칸 뒤)
```

### 스케일링 → split → 훈련 → RMSE

```python
x_data = scaler.fit_transform(x_data)       # 2차원에서, fit 은 훈련 구간만
x_predict = scaler.transform(x_predict)
x = split_x(x_data, 144)                    # (N, 144, 13)
y = split_x(y_data, 144)                    # (N, 144)
model.add(Dense(144))                       # 144개 한 번에
rmse = np.sqrt(mean_squared_error(y_cor, y_predict))
```

---

## 성과

- 1 ~ 100 시계열을 split_x로 잘라 LSTM을 훈련하고, 예측 데이터를 `size - 1`로 잘라 101 ~ 106 여섯 개를 한 번에 예측 (100.97 ~ 106.01)
- 1차원 데이터를 `reshape(-1, 2)` 후 split해 `(45, 5, 2)`를 만들고, `Dense(2)`로 다음 시점의 값 2개를 예측 ([105.77, 106.86])
- `return_sequences=True`로 LSTM을 2층 쌓고, 없을 때 나는 ndim 에러와 층별 파라미터(480 / 608)를 확인
- RNN의 3차원 출력을 Flatten으로 펴서 Dense로 넘기는 구조를 만들고 파라미터(Flatten 0, Dense 496)를 확인
- 80 예측이 72 근처에서 멈추는 이유를 훈련 범위 밖 값으로 설명
- Kaggle Jena Climate 42만 행을 훈련 / 예측용 x / 정답 구간으로 나누고, x와 y를 144칸 어긋나게 잘라 "하루 전 → 다음 하루" 데이터 구성
- float32 npy 저장, 2차원 스케일링, `(420120, 144, 13)` split, `Dense(144)` 모델로 다음 하루 풍향 예측 (RMSE 56.45도)
- y 컬럼을 `T (degC)`로 바꾸면서 기존 코드를 주석으로 남기고 주석 처리 이유를 정리

---

## 💡 주요 학습 포인트

1. **예측 데이터는 `size - 1`로 자른다**: 예측할 데이터에는 y 칸이 없다
2. **예측 묶음 개수 = 예측값 개수**: 96 ~ 105를 5개씩 자르면 6묶음 → 101 ~ 106
3. **`reshape(-1, 2)`로 feature를 늘릴 수 있다**: `(N, 10, 1)` 대신 `(N, 5, 2)`
4. **y의 마지막 축 크기 = 출력층 노드 수**: y `(45, 2)` → `Dense(2)`
5. **작은 시계열은 훈련 loss로 EarlyStopping**: validation_split은 뒤쪽 큰 값들을 떼어간다
6. **RNN 층은 3차원만 받는다**: RNN을 쌓으려면 앞 층이 `return_sequences=True`
7. **return_sequences는 모양만 바꾼다**: 파라미터 개수는 그대로다
8. **앞 층의 units가 다음 층의 feature**: `LSTM(8)` 뒤에 오면 feature=10 → 608
9. **RNN 출력도 Flatten할 수 있다**: `(None, 3, 10)` → `(None, 30)`, 모든 시점 정보를 Dense로
10. **Flatten은 파라미터 0**: 모양만 바꾼다. 2차원에서는 의미 없다
11. **층을 바꿔도 범위 밖 예측은 그대로다**: 80은 훈련 y(최대 70) 밖이라 못 미친다
12. **정답 구간은 훈련에서 완전히 뺀다**: `[:-288]`까지만 훈련, `[-144:]`는 채점에만
13. **x와 y를 예측 길이만큼 어긋나게 자른다**: `x_data [:-288]`, `y_data [144:-144]`
14. **스케일링은 split 전 2차원에서**: 컬럼별로 바로 되고, 메모리도 덜 쓴다
15. **fit은 훈련 구간, transform은 예측 구간**: 예측 구간 정보가 훈련에 새지 않는다
16. **큰 시계열은 float32로**: `(420120, 144, 13)`이 6.3GB → 3.1GB
17. **train_test_split은 묶음 단위로 섞는다**: 묶음 안의 시간 순서는 유지된다
18. **출력을 여러 개로 두면 한 번에 예측한다**: `Dense(144)` = 다음 하루 144개
19. **RMSE는 원래 단위로 해석한다**: mse의 루트 = 평균적으로 몇 도 틀렸나
20. **풍향은 원형 값이다**: 0도와 360도가 같아서 mse로 맞히기 어렵다
21. **y를 바꾸면 x에서 빼는 컬럼도 바꾼다**: y가 x에 남으면 안 되고, 기존 y였던 컬럼은 x로 쓴다

---

[⬅️ Day19](Day19.md) · [🏠 전체 목차](../README.md)
