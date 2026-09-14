# Day11 - 가중치 저장, ModelCheckpoint, Dropout, 함수형 모델

**학습 기간:** 2026-09-14

---

## 핵심 학습 내용

### 가중치만 저장·불러오기 (save_weights / load_weights)
- Day10의 `model.save()`는 모델 전체를 저장했다면, **`model.save_weights()`는 가중치만** 저장한다
  - Keras 3에서는 파일명이 반드시 **`.weights.h5`**로 끝나야 한다 (`keras29_5_save1.weights.h5`)
- 저장 위치를 두 군데로 나눠 실습
  - `fit` **전**에 저장 (`keras29_5_save1.weights.h5`) -> 아직 훈련하지 않은 **초기 가중치**
  - `fit` **후**에 저장 (`keras29_5_save2.weights.h5`) -> **훈련된 가중치** (`restore_best_weights=True`로 되돌린 값)
- `load_weights()`는 가중치만 채워 넣는 것이라 **모델 구조와 `compile`은 코드에 그대로 있어야 한다**
  - `model.save()` 파일은 컴파일 정보까지 들어 있어 `compile` 없이 `evaluate`가 됐지만(Day10), 가중치 파일은 그렇지 않다
- 기록 결과: `weights.h5` 적용 후 loss 3.6324

### ModelCheckpoint (MCP)
- `EarlyStopping`처럼 `callbacks`에 넣는 콜백으로, **훈련 중에 모델을 파일로 저장**한다
  - `monitor='val_loss'` - 무엇을 기준으로 좋은 모델을 고를지 (`loss`도 가능하지만 통상 검증 기준인 `val_loss`)
  - `mode='auto'` - val_loss는 낮을수록 좋으므로 `auto` 또는 `min`
  - `save_best_only=True` - 최고 기록이 갱신될 때만 저장 -> 마지막에 남는 파일 = 가장 좋았던 epoch의 모델
  - `callbacks=[es, mcp]` - 리스트에 넣지 않으면 만들기만 하고 **파일이 하나도 저장되지 않는다**
- `model.fit(..., verbose=1)` 로그에 `val_loss improved from A to B, saving model to ...`가 찍혀 저장 시점을 확인할 수 있다
- keras30_1이 저장한 `keras30_mcp1.keras`를 keras30_2가 `load_model`로 불러와 평가 -> **loss 2.1273으로 같은 값**
- **파일명에 날짜 / epoch / val_loss 넣기** (keras30_3)
  - `datetime.datetime.now().strftime('%m%d_%H%M')` -> `'0914_1323'` (문자열)
  - `filename = '{epoch:04d}_{val_loss:.4f}.keras'` -> MCP가 저장할 때 epoch와 val_loss를 채워 넣는다
  - 실제 저장 파일: `k30_0914_1323_0038_0.5591.keras` (38번째 epoch, val_loss 0.5591)

### MCP 저장(keras31) -> 불러오기(keras32) - 10개 데이터셋
- **save 파일과 load 파일의 평가 값이 소수점까지 같아야 한다**를 기준으로 10개 데이터셋 쌍을 점검
- 값이 같아지는 원리
  - `EarlyStopping`과 `ModelCheckpoint`가 **같은 `val_loss` 기준**으로 최고 epoch를 고른다
  - `restore_best_weights=True`가 훈련 끝에 그 가중치로 되돌린다 -> **훈련이 끝난 model = MCP가 저장한 파일**
- load 파일은 **`#1 데이터`가 save 파일과 한 글자도 달라지면 안 된다**
  - 모델은 '그때 변환된 x'에 맞춰 학습됐으므로, split / `random_state` / `dropna` / scaler가 하나만 달라도 결과가 달라진다
  - diabetes·따릉이의 **2단계 split은 x_val을 안 쓰는 load 파일에서도 지우면 안 된다** - scaler가 split 뒤의 x_train으로 fit하기 때문
  - `print('Min :', ...)` 출력이 save / load에서 같은지 보면 전처리가 같은지 확인할 수 있다
- load 파일은 `load_model` 한 줄이 `#2 모델 구성` + `#3 컴파일, 훈련`을 대신한다 (구조 + 가중치 + compile 정보가 파일에 있음)
  - `fit`을 하면 불러온 가중치에서 또 학습해버려서 값이 달라진다
- save 파일은 **고정 이름**(`keras31_mcp2.keras`)으로 저장 - 날짜 / epoch 파일명은 실행이 끝나기 전까지 이름을 알 수 없어 load 파일에 적을 수 없다
- 점검하면서 찾은 문제
  - keras32 파일들이 이름만 load이고 **날짜 파일명으로 다시 훈련하는 코드**였다 -> `load_model`로 교체
  - bike / cancer의 MCP 파일명이 `keras31_mcp2.keras`로 diabetes와 **중복** -> `mcp5` / `mcp6`
  - cancer는 `callbacks=[es]`에 **mcp가 빠져** 파일이 저장되지 않았다
  - 따릉이 load 파일은 `dropna`와 2단계 split이 주석 처리돼 있었다
- **9쌍 모두 save = load 확인** (loss, r2, mse, RMSE, acc까지 소수점 동일)
- keras32에서 불러온 모델을 **결과값이 들어간 이름으로 같은 폴더에 다시 저장** (`k32_02_0914_1854_3291.8149.keras`) - 원본 `keras31_mcp2.keras`와 가중치가 같은 것을 확인
- 회귀 파일에 없던 `r2 / mse / RMSE`를 전부 추가, 분류 파일은 확률 기준으로 참고용 출력

### Dropout
- `Dropout(0.2)` - **훈련할 때마다** 바로 앞 층 출력의 20% 노드를 랜덤으로 꺼서(0으로 만들어서) 특정 노드에만 의존하지 않게 한다 -> 과적합 방지
- `evaluate` / `predict` 때는 자동으로 꺼지고 모든 노드를 사용한다
- `Dropout` 층은 가중치가 없어서 `model.summary()`에서 **파라미터 0개**
- 10개 데이터셋에 적용 (회귀는 0.2, cancer / santander는 0.3, wine은 200층 뒤 0.4 + 100층 뒤 0.2 등 파일마다 다르게 배치)
- MCP 저장 경로를 `_save/keras33/`으로 분리 - california가 `_save/keras30/keras30_mcp1.keras`를 덮어쓰고 있었다
- **결과는 좋아진 데이터와 나빠진 데이터가 섞여 있다** - seed를 고정하지 않아 1회 실행으로 Dropout 효과를 단정할 수 없다

### 함수형 모델 (Input / Model)
- `keras34_hamsu00.py` - 같은 구조를 **Sequential(순차형)**과 **함수형**으로 각각 만들어 비교
  - 함수형은 `Input(shape=(3,))`으로 입력층을 만들고, `Dense(10)(input1)`처럼 **`층(이전 층)`으로 연결**한다
  - 마지막에 `Model(inputs=input1, outputs=output1)`로 **어디서부터 어디까지가 모델인지 범위를 정의**
  - `name='ys1'`처럼 층 이름을 붙일 수 있다 (없어도 된다)
- 층 구성이 같으면 Sequential과 함수형은 **Total params가 같은 똑같은 모델** - 만드는 방법만 다르다
- keras33(Sequential + Dropout)을 함수형으로 바꾸면서 생긴 실수와 결과
  - **출력층 activation 누락** - santander / covtype / digits의 `softmax`, cancer의 `sigmoid`가 빠져 acc가 0.09 / 0.488 / 0.099 / acc_score 0.333으로 무너졌다
  - **변수 이름 재사용** - cancer에서 `dense2`를 두 번 쓰고 `(drop1)`에 연결해 16층이 모델에서 빠졌다
  - wine은 Dropout 위치·비율이 keras33과 달랐고, bike는 출력층 `relu`가 빠져 있었다
- 수정 후 **10개 모두 keras33과 Total params / 출력층 activation / Dropout 비율이 같은 것을 확인**

---

## 학습 파일

| 파일 | 내용 |
|---|---|
| `keras29_5_save_weights.py` | `model.save_weights()` - `fit` 전(`save1`, 초기 가중치) / 후(`save2`, 훈련된 가중치) 두 번 저장. weights.h5 적용 후 loss 3.6324 |
| `keras29_6_load_weights.py` | `model.load_weights()` - `save2`(훈련된 가중치) 불러오기, 가중치만이라 `compile` 필요. `save1` 불러오기는 에러로 주석 처리 |
| `keras30_ModelCheckPoint1.py` | `ModelCheckpoint(monitor, mode, save_best_only, filepath)`로 `keras30_mcp1.keras` 저장, `callbacks=[es, mcp]`. loss 2.1273 |
| `keras30_ModelCheckPoint2_load.py` | keras30_1이 저장한 MCP 파일을 `load_model`로 불러와 평가 - loss 2.1273 동일 |
| `keras30_ModelCheckPoint3.py` | `datetime` + `'{epoch:04d}_{val_loss:.4f}.keras'`로 날짜 / epoch / val_loss가 들어간 파일명 만들기 |
| `keras31_MCP_save_02_diabetes.py` ~ `keras31_MCP_save_10_digits.py` | 9개 데이터셋 MCP 저장 (`keras31_mcp2 ~ 10.keras`). 07 ~ 10 신규 작성, 02 ~ 06 파일명 중복·mcp 누락 수정, 불필요한 코드 삭제 + 주석 정리, r2 / mse / RMSE 추가 |
| `keras32_MCP_load_02_diabetes.py` ~ `keras32_MCP_load_10_digits.py` | keras31 모델을 `load_model`로 불러와 평가 -> 9쌍 모두 save와 소수점까지 같은 값. 불러온 모델을 `k32_NN_날짜_loss.keras`로 다시 저장. 06 ~ 10 신규 작성, 02 ~ 05 재훈련 코드를 load로 교체 |
| `keras33_dropout01_california.py` ~ `keras33_dropout10_digits.py` | 10개 데이터셋 Dropout 적용 + MCP 저장(`_save/keras33/`) + r2 / mse / RMSE. 01은 함수형, 나머지 Sequential. 02 ~ 04 신규 작성 |
| `keras34_hamsu00.py` | Sequential과 함수형(`Input` / `Model`)으로 같은 구조를 만들어 비교 |
| `keras34_hamsu01_california.py` ~ `keras34_hamsu10_digits.py` | keras33을 함수형으로 변환 + MCP 저장(`_save/keras34/`). 출력층 activation 누락 / 층 연결 오류 수정, 02 ~ 04 신규 작성, ` copy` 붙은 파일명(01 / 08 / 09) 정리 |
| `keras/dataset.txt` | 실습 데이터셋 번호 목록 (01 california ~ 10 digits) |
| `_save/keras29/keras29_5_save1.weights.h5`, `keras29_5_save2.weights.h5` | save_weights 결과물 (훈련 전 / 훈련 후) |
| `_save/keras30/` | `keras30_mcp1.keras`, `k30_0914_1323_0038_0.5591.keras` (MCP 저장 결과물) |
| `_save/keras31/` | `keras31_mcp2 ~ 10.keras` (MCP 저장) + `k32_NN_날짜_loss.keras` (keras32가 불러와 다시 저장) |
| `_save/keras33/` | `keras33_mcp1 ~ 10.keras` (Dropout 모델 MCP 저장) |
| `_save/keras34/` | `keras34_mcp1 ~ 10.keras` (함수형 모델 MCP 저장) |
| `docs/Day11.md` | Day11 학습 기록 신규 작성 |
| `docs/Day10.md` | 하단 nav에 Day11 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조 / 진행도(11일, 13.75%) 갱신 (수정) |
| `.gitignore` | 로컬 에디터 / 도구 설정 폴더 제외 규칙 추가 (수정) |

---

## 실행 결과

### MCP save(keras31) = load(keras32) - 9쌍 모두 소수점까지 동일

| 데이터셋 | loss | r2 / acc | RMSE |
|---|---|---|---|
| Diabetes | 3291.8149 | r2 0.4262 | 57.37 |
| Boston | 19.2725 | r2 0.7685 | 4.390 |
| 따릉이 | 2351.5913 | r2 0.6808 | 48.49 |
| Kaggle Bike | 21296.7656 | r2 0.3180 | 145.93 |
| Cancer | 0.0974 | acc_score 0.9649 | - |
| Santander | 0.2407 | acc_score 0.9113 | - |
| Wine | 0.0873 | accuracy_score 0.9815 | - |
| Covtype | 0.1660 | accuracy_score 0.9427 | - |
| Digits | 0.2589 | accuracy_score 0.9513 | - |

### Dropout 적용 전 / 후 / 함수형 변환 후

| 데이터셋 | Dropout 없음 (keras31) | Dropout (keras33, Sequential) | Dropout (keras34, 함수형) |
|---|---|---|---|
| California (loss / r2) | 2.1273 (keras30, relu 없음) | 0.2531 / 0.8081 | 0.2499 / 0.8105 |
| Diabetes (loss / r2) | 3291.81 / 0.4262 | 3112.51 / 0.4575 | 3296.84 / 0.4253 |
| Boston (loss / r2) | 19.27 / 0.7685 | 20.07 / 0.7589 | 19.54 / 0.7653 |
| 따릉이 (loss / r2) | 2351.59 / 0.6808 | 2411.27 / 0.6727 | 2727.14 / 0.6298 |
| Kaggle Bike (loss / r2) | 21296.77 / 0.3180 | 22479.80 / 0.2801 | 22557.91 / 0.2776 |
| Cancer (acc_score) | 0.9649 | 0.9474 | 0.9532 |
| Santander (acc_score) | 0.9113 | 0.9075 | 0.9110 |
| Wine (accuracy_score) | 0.9815 | 0.9630 | 0.9444 |
| Covtype (accuracy_score) | 0.9427 | 0.9236 | 0.9235 |
| Digits (accuracy_score) | 0.9513 | 0.9694 | 0.9666 |

**읽는 법:** seed를 고정하지 않아 같은 코드도 실행할 때마다 값이 달라진다. keras33과 keras34는 같은 모델(Total params 동일)인데도 값이 다른 것이 그 증거이므로, 이 표로 Dropout이나 함수형의 우열을 단정하지 않는다. California의 큰 개선은 Dropout과 함께 은닉층에 `relu`가 들어간 영향도 섞여 있다.

---

## 성과

- `save_weights` / `load_weights`로 **가중치만 저장·불러오기**를 훈련 전 / 후로 나눠 실습
- `ModelCheckpoint`로 훈련 중 최고 모델을 저장하고, 날짜 / epoch / val_loss가 들어간 파일명까지 직접 만들어 저장
- 10개 데이터셋 MCP save / load 쌍을 점검해 **9쌍 모두 소수점까지 같은 값**을 확인 (파일명 중복, callbacks 누락, 전처리 불일치, 재훈련 코드 등 원인 수정)
- 모든 데이터셋에 **r2 / mse / RMSE 출력 추가**
- 10개 데이터셋에 `Dropout`을 적용하고 MCP 저장 경로를 데이터별 폴더로 분리
- Sequential 모델 10개를 **함수형 모델로 변환**하고, 출력층 activation 누락으로 acc가 0.09까지 떨어진 원인을 찾아 수정 -> 10개 모두 Total params 일치 확인

---

## 핵심 개념

### save_weights / load_weights - 가중치만 저장

```python
path = './_save/keras29/'

model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(50))
model.add(Dense(1))

model.save_weights(path + 'keras29_5_save1.weights.h5')   # fit 전 -> 초기 가중치

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10000, batch_size=32,
                 validation_split=0.2, callbacks=[es])

model.save_weights(path + 'keras29_5_save2.weights.h5')   # fit 후 -> 훈련된 가중치

# 불러올 때 (keras29_6)
model.load_weights(path + 'keras29_5_save2.weights.h5')
model.compile(loss='mse', optimizer='adam')   # weights 만 불러왔기 때문에 compile 이 있어야 한다
loss = model.evaluate(x_test, y_test)
```

### ModelCheckpoint - 훈련 중 최고 모델 저장

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', patience=30,
                   restore_best_weights=True, verbose=1)

mcp = ModelCheckpoint(
    monitor='val_loss',       # 무엇을 기준으로 좋은 모델을 고를지
    mode='auto',              # val_loss 는 낮을수록 좋다 -> auto(=min)
    save_best_only=True,      # 최고 기록이 갱신될 때만 저장
    filepath=path + 'keras30_mcp1.keras',
    verbose=1,
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=32,
                 validation_split=0.2,
                 callbacks=[es, mcp],   # 리스트에 넣어야 실제로 저장된다
                 verbose=1)
# 로그 : Epoch N: val_loss improved from A to B, saving model to ./_save/keras30/keras30_mcp1.keras
#        -> val_loss 가 좋아진 epoch 에서만 저장이 일어난다
```

### 파일명에 날짜 / epoch / val_loss 넣기

```python
import datetime
date = datetime.datetime.now()          # 2026-09-14 11:40:57.930454 (datetime 객체)
date = date.strftime('%m%d_%H%M')       # '0914_1147' (문자열)

path = './_save/keras30/'
filename = '{epoch:04d}_{val_loss:.4f}.keras'   # MCP 가 저장할 때 값을 채워 넣는다
filepath = ''.join([path, 'k30_', date, '_', filename])

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,
                      filepath=filepath, verbose=1)
# -> ./_save/keras30/k30_0914_1323_0038_0.5591.keras
```

### MCP 저장(keras31) -> 불러오기(keras32)

```python
# keras31 (save) - 고정 이름으로 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,
                      filepath=path + 'keras31_mcp2.keras', verbose=1)
hist = model.fit(..., callbacks=[es, mcp])
loss = model.evaluate(x_test, y_test)          # loss : 3291.81494140625

# keras32 (load) - #1 데이터는 keras31 과 똑같이 두고 #2, #3 만 한 줄로
from tensorflow.keras.models import load_model
model = load_model(path + 'keras31_mcp2.keras')   # 구조 + 가중치 + compile 정보
loss = model.evaluate(x_test, y_test)          # loss : 3291.81494140625  <- 같아야 정상
```

### load 파일에서도 지우면 안 되는 전처리

```python
# diabetes : x_val 은 안 쓰지만 이 split 이 x_train 을 49% 로 줄이고, scaler 가 그 x_train 으로 fit 한다
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.7, random_state=77)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)   # 훈련은 안 해도 scaler 기준을 만들려면 필요
x_test = scaler.transform(x_test)

# 따릉이 : dropna 가 빠지면 행 개수가 달라져 split 이 다른 행을 x_test 로 뽑는다
train_csv = train_csv.dropna()
```

### 불러온 모델을 결과값 이름으로 다시 저장 (keras32)

```python
import datetime
date = datetime.datetime.now().strftime('%m%d_%H%M')
filepath = ''.join([path, 'k32_02_', date, '_', f'{loss:.4f}', '.keras'])
model.save(filepath)   # ./_save/keras31/k32_02_0914_1854_3291.8149.keras
```

### Dropout

```python
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(32, input_dim=30, activation='relu'))
model.add(Dropout(0.3))     # 훈련할 때마다 앞 층 출력의 30% 를 랜덤으로 끈다
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.summary()             # Dropout 층은 Param # 0
# evaluate / predict 때는 Dropout 이 자동으로 꺼진다
```

### Sequential vs 함수형 (keras34_hamsu00)

```python
from tensorflow.keras.models import Sequential, Model   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input

#2-1. 순차형
model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dropout(0.2))
model.add(Dense(9))
model.add(Dropout(0.2))
model.add(Dense(1))

#2-2. 함수형
input1 = Input(shape=(3,))
dense1 = Dense(10, name='ys1')(input1)   # (input1) -> 앞 층에 연결, name 은 없어도 된다
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(9, name='ys2')(drop1)
drop2 = Dropout(0.2)(dense2)
output1 = Dense(1)(drop2)
model2 = Model(inputs=input1, outputs=output1)   # 어디서부터 어디까지인지 범위 정의
```

### 함수형으로 바꿀 때 실수했던 부분

```python
# ❌ 변수 이름 재사용 + 잘못된 연결 -> 16 층이 모델에서 빠진다
dense2 = Dense(16, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense2 = Dense(8, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
output1 = Dense(1)(drop2)                          # ❌ sigmoid 누락 -> acc_score 0.333

# ✅ 층마다 새 이름, 바로 앞 층에 연결, 출력층 activation 유지
dense2 = Dense(16, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(8, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense3)
output1 = Dense(1, activation='sigmoid')(drop3)

# 다중 분류도 마찬가지
# output1 = Dense(10)(drop4)                        # ❌ softmax 누락 -> acc 0.099
output1 = Dense(10, activation='softmax')(drop4)    # ✅
```

---

## 💡 주요 학습 포인트

1. **`save_weights`는 가중치만**: 모델 구조와 `compile`은 코드에 있어야 한다. 파일명은 `.weights.h5`로 끝낸다
2. **저장 시점이 내용을 결정한다**: `fit` 전 `save_weights` = 초기 가중치 / `fit` 후 = 훈련된 가중치
3. **MCP는 `callbacks`에 넣어야 동작한다**: 만들기만 하고 `callbacks=[es]`로 두면 파일이 하나도 안 생긴다
4. **`save_best_only=True` + `restore_best_weights=True`**: 둘 다 val_loss 최고 epoch를 고르므로 훈련이 끝난 model과 저장 파일이 같다
5. **파일명 템플릿**: `'{epoch:04d}_{val_loss:.4f}.keras'`는 MCP가 저장할 때 값을 채운다. 대신 load 파일에서 이름을 미리 알 수 없어 짝 파일에는 고정 이름을 쓴다
6. **load 파일의 `#1 데이터`는 save 파일과 완전히 같아야 한다**: 쓰지 않는 x_val split이나 `dropna`도 scaler 기준과 x_test를 바꾸므로 지우면 안 된다
7. **`load_model` 뒤에 `fit`하지 않는다**: 불러온 가중치에서 또 학습해 값이 달라진다
8. **MCP 파일명 번호를 겹치지 않게**: 복사한 코드의 `mcp2`를 그대로 두면 다른 데이터의 모델을 덮어쓴다 (저장 폴더도 keras30 / 31 / 33 / 34로 분리)
9. **Dropout은 훈련 때만 동작한다**: 파라미터 0개, evaluate / predict 때는 모든 노드 사용
10. **함수형 = `Input` -> `층(이전 층)` -> `Model(inputs, outputs)`**: 층 구성이 같으면 Sequential과 Total params가 같다
11. **함수형으로 옮길 때 출력층 activation을 빠뜨리지 않는다**: softmax / sigmoid가 없으면 acc가 0.09 수준으로 무너진다
12. **seed 미고정 편차**: keras33과 keras34는 같은 모델인데도 결과가 달랐다. 1회 실행으로 Dropout / 함수형의 우열을 단정하지 않는다

---

[⬅️ Day10](Day10.md) · [🏠 전체 목차](../README.md)
