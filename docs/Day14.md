# Day14 - MaxPooling · GlobalAveragePooling 적용, DNN vs CNN 비교, 정형 데이터에 Conv2D 적용

**학습 기간:** 2026-09-17

> Day13 에서 배운 **MaxPooling2D** 를 mnist / fashion / cifar10 / cifar100 네 모델에 넣어 다시 훈련했고, Flatten 대신 **GlobalAveragePooling2D** 를 적용했다. 같은 이미지를 **DNN(Dense만)** 으로도 훈련해 CNN 과 비교했고, 마지막으로 Day07 ~ Day11 에서 다룬 **정형(표) 데이터 10종에 Conv2D 를 적용**했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| MaxPooling2D 적용 | 4개 모델 모두 acc 가 오르거나 비슷했고, GPU 훈련 시간은 줄었다. mnist 0.9953 / cifar100 0.4371 로 목표 달성 |
| GlobalAveragePooling2D | 채널마다 가로×세로 전체 평균 하나만 남긴다 -> `(가로, 세로, 채널)` 이 `(채널,)` 이 된다. Flatten 대신 사용 |
| GAP 결과 | cifar10 0.7540 -> 0.7727, cifar100 0.4371 -> 0.4621 로 MaxPooling 만 쓴 것보다 올랐다 |
| DNN 으로 이미지 | `reshape(-1, 28*28*1)` 로 2차원으로 펴서 Dense 에 넣는다. 결과는 모두 CNN 보다 낮았다 |
| BatchNormalization | cifar10 DNN 에 Dense 뒤마다 넣어 봤다 (acc 0.5784) |
| 정형 데이터 + Conv2D | 컬럼 수를 `(행, 열, 1)` 로 나눠 reshape 하면 표 데이터도 Conv2D 에 넣을 수 있다 |
| 커널 크기 조절 | 가로가 1칸만 남으면 `(2,2)` 커널은 못 쓴다 -> `(2,1)` 로 세로 방향만 훑는다 |
| 컬럼 수가 소수 | 13 컬럼(boston, wine)은 직사각형으로 못 나눠 `(13, 1, 1)` 로 세웠다 |
| 공통 템플릿 | keras42 파일 10개를 같은 순서 / 같은 주석 형식(출력 크기 · param 계산식)으로 통일 |

---

## 📖 핵심 학습 내용

### 1. MaxPooling2D 를 네 이미지 모델에 적용 (keras39_MaxPooling1 ~ 4)

Day13 의 keras36 파일을 복사해 MaxPooling2D 를 넣고 다시 훈련했다.

- **mnist (keras39_MaxPooling1)**
  - keras36_cnn3_mnist2 기반. Conv2D 전 층을 `padding='same'` 으로 바꾸고, 첫 Conv2D 묶음(32 필터 3개) 뒤에 `MaxPooling2D()` 1개를 넣었다
  - GPU **acc 0.9953** (418.48초) -> Day13 에서 못 넘은 **목표 0.995 달성** (이전 0.9938)
- **fashion_mnist (keras39_MaxPooling2)**
  - keras36_cnn4 기반. `strides=(2,2)` 층 뒤와 마지막 Conv2D 뒤에 MaxPooling2D 3개 추가
  - CPU acc 0.9279 (763.36초) / GPU acc 0.923 (**116.61초**)
  - 이전(CPU 951.0초 / GPU 380.58초)보다 시간이 크게 줄었다
- **cifar10 (keras39_MaxPooling3)**
  - keras36_cnn5 기반. 세 번째 Conv2D `(5,5)` 뒤에 MaxPooling2D 1개
  - GPU acc 0.7267 -> **0.754**, 시간 415.61초 -> **202.33초**
- **cifar100 (keras39_MaxPooling4)**
  - Day13 에 keras36_cnn6 에서 작업하던 MaxPooling 2개 모델을 이 파일로 옮겨서 실행
  - GPU acc 0.2693 -> **0.4371** (227.8초) -> **목표 0.4 달성**
  - Total params 6,736,100 -> 627,972
- **keras36_cnn6_cifar100.py (수정)**
  - MaxPooling 코드를 keras39 로 옮기면서 원래 파일은 MaxPooling 이 없는 1차 모델로 되돌렸다
  - 1차 기록과 똑같은 숫자가 복사돼 있던 "2차 기록" 칸을 정리했다

### 2. GlobalAveragePooling2D (keras40_GAP01 ~ 04)

- keras39 네 파일을 복사해 `Flatten()` 을 주석 처리하고 **`GlobalAveragePooling2D()`** 로 바꿨다
- 동작
  - 채널 하나(가로×세로 판 하나)의 **모든 값을 평균** 내서 숫자 하나로 만든다
  - `(가로, 세로, 채널)` -> `(채널,)`. 예: `(7, 7, 128)` -> `(128,)`
  - Flatten 은 `7×7×128 = 6272` 개를 그대로 펴므로, 뒤에 붙는 Dense 의 파라미터가 GAP 보다 훨씬 많다
  - 평균만 내므로 GAP 층 자체의 파라미터는 0
- import : `from tensorflow.keras.layers import GlobalAveragePooling2D`
- 결과 (GPU, MaxPooling 적용 모델 -> GAP 적용)

| 데이터셋 | MaxPooling | + GAP | 시간 (MaxPooling -> GAP) |
|---|:---:|:---:|---|
| mnist | 0.9953 | 0.9951 | 418.48초 -> 285.72초 |
| fashion_mnist | 0.923 | 0.9248 | 116.61초 -> 118.45초 |
| cifar10 | 0.754 | **0.7727** | 202.33초 -> 199.62초 |
| cifar100 | 0.4371 | **0.4621** | 227.8초 -> 227.22초 |

- keras40_GAP02_fashion 에는 fashion 에서 처음으로 EarlyStopping(`val_acc`, `mode='max'`, patience 25)을 붙였다

### 3. 같은 이미지를 DNN 으로 (keras41_dnn1 ~ 4)

- 목표 : **"CNN 을 이겨라"** - Conv2D 없이 Dense 만으로 이미지 분류
- **x 를 2차원으로 편다**
  - mnist / fashion : `x_train.reshape(-1, 28 * 28 * 1)` -> `(60000, 784)`
  - cifar10 : `x_train.reshape(-1, 3072)` (32×32×3)
  - 첫 층 : `Dense(512, activation='relu', input_shape=(784,))`
- mnist / fashion DNN : Dense 512 -> 256 -> 128 -> ... -> 16, 중간에 Dropout 0.4 / 0.3 / 0.2
  - 이 두 파일은 스케일링(`(x - 127.5)/127.5`)을 주석 처리한 상태로 실행했다
- cifar10 DNN : Dense 1024 부터 시작하고, Dense 뒤마다 **`BatchNormalization()`** + Dropout 을 넣었다 (epochs 500, ES patience 25)
- y 원핫 : `ohe.fit_transform(y_train)` / **`ohe.transform(y_test)`** 로 test 는 transform 만 쓰도록 바꿨다 (cifar10 / cifar100 DNN)
- cifar100 DNN (keras41_dnn4) 은 결과 기록 칸이 비어 있다
- 결과 (GPU)

| 데이터셋 | CNN 최고 | DNN | 멈춘 epoch / 시간 |
|---|:---:|:---:|---|
| mnist | 0.9953 (MaxPooling) | 0.9785 | 59 / 223.62초 |
| fashion_mnist | 0.9279 (MaxPooling, CPU) | 0.8661 | 75 / 77.2초 |
| cifar10 | 0.7727 (GAP) | 0.5784 | 70 / 130.08초 |

-> 세 데이터셋 모두 DNN 이 CNN 보다 낮았다. 컬러 사진(cifar10)에서 차이가 가장 컸다 (0.19)

### 4. 정형 데이터에 Conv2D 적용 (keras42_cnn1 ~ 10)

- Day11 의 함수형 모델 파일(keras34_hamsu01 ~ 10) 10개를 Conv2D 모델로 바꿨다
- **표 데이터를 4차원으로** : 컬럼 수를 `행 × 열` 로 나눠 `(샘플, 행, 열, 1)` 로 reshape
  - 8 컬럼 -> `(4, 2, 1)` / 9 -> `(3, 3, 1)` / 10 -> `(5, 2, 1)` / 30 -> `(5, 6, 1)` / 54 -> `(6, 9, 1)` / 200 -> `(10, 20, 1)` / 64 -> `(8, 8, 1)`
  - 13 컬럼(boston, wine)은 13 이 소수라 `(13, 1, 1)` 로 세웠다
  - digits 는 원래 8×8 이미지라 `(8, 8, 1)` 이 원래 모양이다
- **입력이 작으면 층마다 크기를 계산해야 한다**
  - 처음 복사한 모델(`(3,3)` 커널 6층)은 `(6, 9)` 입력에서 `6 -> 4 -> 2 -> 0` 이 돼 에러가 났다
  - 커널을 `(2,2)` 로 줄이고, 첫 층에만 `padding='same'` 을 줘서 크기를 유지했다
  - 가로가 1칸이 되면 `(2,2)` 를 쓸 수 없어 `(2,1)` 커널로 바꿨다 (padding 없는 `(2,1)` 은 세로만 1씩 줄어든다)
- **복사하면서 생긴 에러를 찾아 고쳤다**
  - california / diabetes 같은 회귀 데이터에 분류용 `to_categorical` / `stratify=y` / `Dense(8, softmax)` 가 남아 있었다 -> 제거하고 `Dense(1)` (linear)
  - 함수형 모델 코드(`Input`, `Model`)가 import 없이 남아 `NameError` -> Sequential 로 통일
  - diabetes 는 val 을 나눠 놓고 스케일링도, 사용도 안 하고 있었다 -> val 도 transform + reshape 후 `validation_data=(x_val, y_val)`
  - `es` 를 만들고 callbacks 에 안 넣었거나 주석 처리한 파일 -> 모두 `callbacks=[es, mcp]`
- **공통 템플릿으로 통일** (keras42_cnn9 구조 기준)
  - `#1. 데이터` : 불러오기 -> (원핫) -> 분할 -> RobustScaler -> reshape
  - `#2. 모델 구성` : `Conv2D(64, same)` -> `Conv2D(64)` -> `Conv2D(32)` -> Flatten -> Dense 128 -> Dense 64 -> 출력층 (원래 Dropout 이 있던 파일은 Dropout 유지)
  - `#3. 컴파일, 훈련` : EarlyStopping + ModelCheckpoint -> fit -> 소요 시간
  - `#4. 평가 예측` : 회귀 / 이진 분류 / 다중 분류
  - 층마다 `출력 : (shape)  param 계산식` 주석을 같은 열에 맞춰 달았다
- MCP 저장 파일 `_save/keras34/keras34_mcp1 ~ 10.keras` 가 Conv2D 모델로 다시 저장됐다
- 결과 (GPU, 이전 DNN / 함수형 기록 -> Conv2D)

| 파일 | 데이터 | reshape | 이전 | Conv2D |
|---|---|---|:---:|:---:|
| cnn1 | california (회귀) | (4, 2, 1) | r2 0.8014 | r2 0.7998 |
| cnn2 | diabetes (회귀) | (5, 2, 1) | r2 0.4236 | r2 **0.4498** |
| cnn3 | boston (회귀) | (13, 1, 1) | r2 0.7554 | r2 **0.7726** |
| cnn4 | ddareung (회귀) | (3, 3, 1) | r2 0.6298 | r2 **0.7108** |
| cnn5 | kaggle_bike (회귀) | (4, 2, 1) | r2 0.2841 | r2 **0.3048** |
| cnn6 | cancer (이진) | (5, 6, 1) | acc 0.9591 | acc **0.9649** |
| cnn7 | santander (이진) | (10, 20, 1) | acc 0.9087 | acc **0.9110** |
| cnn8 | wine (다중) | (13, 1, 1) | acc 0.9815 | acc 0.9630 |
| cnn9 | fetch_covtype (다중) | (6, 9, 1) | acc 0.8924 | acc **0.9061** |
| cnn10 | digits (다중) | (8, 8, 1) | acc 0.9555 | 기록 없음 |

-> 9개 중 7개에서 Conv2D 가 이전 모델보다 좋았다. 표 데이터도 4차원으로 바꾸면 Conv2D 로 훈련할 수 있다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras39_MaxPooling1_mnist.py` | mnist + `padding='same'` + MaxPooling2D 1개, acc 0.9953 (목표 0.995 달성) |
| `keras39_MaxPooling2_fashion.py` | fashion_mnist + MaxPooling2D 3개, CPU 0.9279 (763초) / GPU 0.923 (117초) |
| `keras39_MaxPooling3_cifar10.py` | cifar10 + MaxPooling2D 1개, acc 0.7267 -> 0.754, 415초 -> 202초 |
| `keras39_MaxPooling4_cifar100.py` | cifar100 + MaxPooling2D 2개, acc 0.2693 -> 0.4371 (목표 0.4 달성), params 627,972 |
| `keras40_GAP01_mnist.py` | Flatten -> GlobalAveragePooling2D, acc 0.9951 (285.72초) |
| `keras40_GAP02_fashion.py` | GAP + EarlyStopping(val_acc), acc 0.9248 |
| `keras40_GAP03_cifar10.py` | GAP 적용, acc 0.7727 |
| `keras40_GAP04_cifar100.py` | GAP 적용, acc 0.4621 |
| `keras41_dnn1_mnist.py` | mnist DNN - `reshape(-1, 784)`, Dense 11층, acc 0.9785 |
| `keras41_dnn2_fashion.py` | fashion_mnist DNN, acc 0.8661 |
| `keras41_dnn3_cifar10.py` | cifar10 DNN - `reshape(-1, 3072)`, BatchNormalization, acc 0.5784 |
| `keras41_dnn4_cifar100.py` | cifar100 DNN - Dense 모델 작성, `ohe.transform(y_test)` (결과 기록 없음) |
| `keras42_cnn1_california.py` | 캘리포니아 집값 - `(4,2,1)` Conv2D 회귀, r2 0.7998 |
| `keras42_cnn2_diabetes.py` | 당뇨병 - `(5,2,1)`, val 세트 스케일링 / `validation_data` 적용, r2 0.4498 |
| `keras42_cnn3_boston.py` | 보스턴 집값 - `(13,1,1)` + `(2,1)` 커널, r2 0.7726 |
| `keras42_cnn4_dacon_ddareung.py` | 따릉이 - `(3,3,1)` + Dropout, r2 0.7108 |
| `keras42_cnn5_kaggle_bike.py` | 자전거 대여 - `(4,2,1)`, 출력층 relu, r2 0.3048 |
| `keras42_cnn6_cancer.py` | 유방암 이진 분류 - `(5,6,1)` + sigmoid, acc 0.9649 |
| `keras42_cnn7_santander.py` | Santander 이진(원핫) - `(10,20,1)` + softmax, acc 0.9110 |
| `keras42_cnn8_wine.py` | 와인 다중 분류 - `(13,1,1)` + `(2,1)` 커널, acc 0.9630 |
| `keras42_cnn9_fetch_covtype.py` | 산림 피복 다중 분류 - `(6,9,1)`, acc 0.9061 |
| `keras42_cnn10_digits.py` | 손글씨 숫자 - 원래 모양 `(8,8,1)` 로 Conv2D (결과 기록 없음) |
| `keras36_cnn1.py` | summary 뒤에 `exit()` 추가 (수정) |
| `keras36_cnn6_cifar100.py` | MaxPooling 코드를 keras39_MaxPooling4 로 옮기고 1차 모델로 되돌림, 기록 정리 (수정) |
| `_save/keras34/keras34_mcp1 ~ 10.keras` | keras42 Conv2D 모델로 다시 저장된 MCP 파일 (수정) |
| `docs/Day14.md` | Day14 학습 기록 신규 작성 |
| `docs/Day13.md` | 하단 nav 에 Day14 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조 / 진행도(14일, 17.5%) 갱신 (수정) |

---

## 📊 실행 결과

### 이미지 4종 - 모델 구조별 acc (GPU)

| 데이터셋 | Day13 CNN | + MaxPooling | + GAP | DNN | 목표 |
|---|:---:|:---:|:---:|:---:|:---:|
| mnist | 0.9938 | **0.9953** | 0.9951 | 0.9785 | 0.995 ✅ |
| fashion_mnist | 0.9258 | 0.923 | 0.9248 | 0.8661 | 0.92 ✅ |
| cifar10 | 0.7267 | 0.754 | **0.7727** | 0.5784 | 0.67 ✅ |
| cifar100 | 0.2693 | 0.4371 | **0.4621** | - | 0.4 ✅ |

### GPU 훈련 시간

| 데이터셋 | Day13 CNN | + MaxPooling | + GAP | DNN |
|---|---:|---:|---:|---:|
| mnist | 322.36초 | 418.48초 | 285.72초 | 223.62초 |
| fashion_mnist | 380.58초 | 116.61초 | 118.45초 | 77.2초 |
| cifar10 | 415.61초 | 202.33초 | 199.62초 | 130.08초 |
| cifar100 | 394.21초 | 227.8초 | 227.22초 | - |

**읽는 법:** mnist 는 EarlyStopping 으로 멈춘 epoch 가 실행마다 달라 시간 비교가 정확하지 않다. MaxPooling 을 넣은 뒤 fashion(epochs 50 고정) / cifar10 은 시간이 절반 이하로, cifar100 은 394초 -> 228초로 줄었다.

---

## 💻 핵심 개념

### MaxPooling2D 로 크기 줄이기

```python
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Conv2D(32, (5,5), activation='relu', padding='same'))   # (28, 28, 32)
model.add(MaxPooling2D())                                         # (14, 14, 32)  param 0
model.add(Dropout(0.25))
```

### Flatten 대신 GlobalAveragePooling2D

```python
from tensorflow.keras.layers import GlobalAveragePooling2D

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))   # (7, 7, 128)
model.add(MaxPooling2D())
model.add(Dropout(0.25))
# model.add(Flatten())            # 모든 값을 그대로 펴서 Dense 로
model.add(GlobalAveragePooling2D())   # 채널마다 평균 1개 -> (채널,)
model.add(Dense(units=256, activation='relu'))
```

### 이미지를 DNN 으로 (2차원으로 펴기)

```python
x_train = x_train.reshape(-1, 28 * 28 * 1)   # (60000, 28, 28) -> (60000, 784)
x_test = x_test.reshape(-1, 28 * 28 * 1)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(256, activation='relu'))
...
model.add(Dense(10, activation='softmax'))
```

### BatchNormalization (cifar10 DNN)

```python
from tensorflow.keras.layers import BatchNormalization

x_train = x_train.reshape(-1, 3072)          # 32 x 32 x 3

model.add(Dense(1024, activation='relu', input_shape=(3072,)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
```

### y 원핫 - test 는 transform 만

```python
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)   # train 으로 기준(클래스 목록)을 만든다
y_test = ohe.transform(y_test)         # test 는 그 기준으로 변환만
```

### 정형 데이터를 Conv2D 에 넣기

```python
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Conv2D 는 (행, 열, 채널) 4차원 입력이 필요하다 -> 컬럼 30개를 5 x 6 x 1 로 바꾼다
x_train = x_train.reshape(-1, 5, 6, 1)  # (398, 5, 6, 1)
x_test = x_test.reshape(-1, 5, 6, 1)    # (171, 5, 6, 1)

model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', activation='relu', input_shape=(5, 6, 1)))  # 출력 : (5, 6, 64)  param 320 = (2x2x1+1)x64
model.add(Conv2D(64, (2,2), activation='relu'))                                         # 출력 : (4, 5, 64)  param 16448 = (2x2x64+1)x64
model.add(Conv2D(32, (2,2), activation='relu'))                                         # 출력 : (3, 4, 32)  param 8224 = (2x2x64+1)x32
model.add(Flatten())                                                                    # 출력 : (384,)  3x4x32
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
```

### 가로가 1칸일 때는 (2,1) 커널

```python
# 컬럼 13개 -> 13 은 소수라 직사각형으로 못 나눈다 -> 13 x 1 x 1
x_train = x_train.reshape(-1, 13, 1, 1)

model.add(Conv2D(64, (2,1), padding='same', activation='relu', input_shape=(13, 1, 1)))  # (13, 1, 64)  param 192 = (2x1x1+1)x64
model.add(Conv2D(64, (2,1), activation='relu'))                                          # (12, 1, 64)  param 8256
model.add(Conv2D(32, (2,1), activation='relu'))                                          # (11, 1, 32)  param 4128
model.add(Flatten())                                                                     # (352,)

# (2,2) 커널을 쓰면 가로 1 -> 0 이 되어 에러
```

### val 세트도 스케일링 + reshape

```python
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)         # val 도 transform 만

x_train = x_train.reshape(-1, 3, 3, 1)  # (849, 3, 3, 1)
x_val = x_val.reshape(-1, 3, 3, 1)      # (213, 3, 3, 1)
x_test = x_test.reshape(-1, 3, 3, 1)    # (266, 3, 3, 1)

hist = model.fit(x_train, y_train,
                 validation_data=(x_val, y_val),
                 callbacks=[es, mcp])
```

---

## 성과

- MaxPooling2D 를 적용해 **mnist 0.9953, cifar100 0.4371 로 Day13 에서 미달이던 목표 두 개를 달성**
- GlobalAveragePooling2D 로 cifar10 0.7727, cifar100 0.4621 까지 올림
- 같은 이미지를 DNN 으로 훈련해 세 데이터셋 모두 **CNN 이 DNN 보다 정확하다**는 것을 확인 (cifar10 0.7727 vs 0.5784)
- BatchNormalization 을 DNN 에 처음 넣어 봄
- 정형 데이터 10종을 Conv2D 모델로 바꾸고, 기록한 9개 중 7개에서 이전 모델보다 좋은 결과
- 복사 과정에서 생긴 에러(크기 0, 회귀에 원핫, 함수형 import 누락, val 미사용, es 미적용)를 찾아 수정
- keras42 파일 10개를 공통 템플릿으로 통일

---

## 💡 주요 학습 포인트

1. **MaxPooling2D 는 성능과 속도를 같이 올릴 수 있다**: cifar10 0.7267 -> 0.754, 415초 -> 202초
2. **GlobalAveragePooling2D**: 채널마다 평균 하나 -> `(가로, 세로, 채널)` 이 `(채널,)`. Flatten 자리에 쓴다
3. **GAP 는 Flatten 보다 뒤 Dense 입력이 작다**: `(7,7,128)` 이면 Flatten 6272 개, GAP 128 개
4. **DNN 에 이미지를 넣으려면 2차원으로 편다**: `reshape(-1, 가로*세로*채널)`
5. **이미지는 CNN 이 DNN 보다 정확하다**: mnist 0.9953 vs 0.9785, fashion 0.9279 vs 0.8661, cifar10 0.7727 vs 0.5784
6. **OneHotEncoder 도 test 는 `transform` 만 쓴다**: 스케일러와 같은 원리
7. **정형 데이터도 `(샘플, 행, 열, 1)` 로 바꾸면 Conv2D 에 들어간다**: 행 × 열 = 컬럼 수
8. **입력이 작으면 층마다 출력 크기를 먼저 계산한다**: padding 없는 커널은 `(커널 - 1)` 씩 줄고, 0 이하가 되면 에러
9. **가로가 1칸이면 `(2,1)` 커널**: 세로 방향으로만 훑는다
10. **첫 층 `padding='same'`**: 작은 입력에서 크기를 한 번 지켜 줘서 층을 하나 더 쌓을 수 있다
11. **파일을 복사해 쓸 때는 문제 유형부터 확인한다**: 회귀에 `to_categorical` / `stratify` / softmax 가 남으면 에러
12. **val 을 직접 나눴으면 스케일링 · reshape · `validation_data` 까지 같이 적용한다**
13. **EarlyStopping 은 callbacks 에 넣어야 동작한다**: 만들기만 하면 아무 일도 안 한다

---

[⬅️ Day13](Day13.md) · [🏠 전체 목차](../README.md)
