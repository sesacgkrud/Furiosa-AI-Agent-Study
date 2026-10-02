# Day23 - Reuters · IMDB 텍스트 분류, sparse_categorical_crossentropy, 표 데이터 LSTM, Reshape 층

**학습 기간:** 2026-10-02

> Day22에서는 Tokenizer · pad_sequences · Embedding으로 문장 15개를 분류했다. Day23은 keras에 들어 있는 **실제 텍스트 데이터셋 2개**를 Embedding + LSTM으로 분류했다. **Reuters 뉴스 46종 다중 분류**(목표 acc 0.67)와 **IMDB 영화 리뷰 긍정 / 부정 이진 분류**(목표 acc 0.6)를 하고, 두 모델 모두 ModelCheckpoint와 `save_weights`로 저장했다. 이어서 y를 원핫하지 않는 **`sparse_categorical_crossentropy`**를 mnist · fashion · cifar10 · cifar100에 적용했다. 그다음 Day14에서 Conv2D로 바꿨던 **표 데이터 10종과 mnist를 LSTM으로** 바꿔 Conv2D · DNN 결과와 비교했다. 마지막으로 모델 안에서 모양을 바꾸는 **`Reshape` 층**으로 Dense → Conv2D, Conv2D → LSTM → Conv2D를 연결했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| reuters / imdb | `load_data(num_words=1000)` → 많이 나온 단어 1,000개만, 기사(리뷰) 하나 = 단어 번호 리스트 |
| `num_words` ↔ `input_dim` | 단어 번호가 0 ~ 999 → `Embedding(input_dim=1000)` (가장 큰 번호 + 1) |
| pad_sequences 기본값 | `padding='pre'` / `truncating='pre'` → 실제 단어가 뒤에 붙어 LSTM 마지막 상태에 유리 |
| Reuters 결과 | 46종 softmax, acc **0.7418** (목표 0.67) |
| IMDB 결과 | sigmoid, acc **0.8619** (목표 0.6) |
| sparse | y 원핫 생략 + `loss='sparse_categorical_crossentropy'`, 출력층은 그대로 클래스 수 + softmax |
| sparse 평가 | `y_predict`만 `argmax(axis=1)`, **`y_test`는 argmax 하지 않는다** |
| 표 데이터 LSTM | `(N, 컬럼 수, 1)` → 컬럼 하나 = 시점 하나, Conv2D 3층 + Flatten → `LSTM(64)` 1층 |
| 이미지 LSTM | digits `(N, 8, 8)` / mnist `(N, 28, 28)` → 가로줄 = 시점, 줄의 픽셀 = feature |
| Reshape 층 | `Reshape(target_shape=...)` → 값 개수가 같으면 모델 안에서 모양만 바꿈 (param 0) |
| Conv2D ↔ LSTM | `Reshape(400, 32)` → `LSTM(return_sequences=True)` → `Reshape(20, 20, 10)` → Conv2D |

---

## 📖 핵심 학습 내용

### 1. Reuters 뉴스 다중 분류 (keras62_1_reuters)

```python
from tensorflow.keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,     # 많이 나온 단어 1,000개만 사용
    test_split=0.2,     # test 비율
)

print(x_train.shape, y_train.shape)     # (8982,) (8982,)
print(x_test.shape, y_test.shape)       # (2246,) (2246,)
print(np.unique(y_train))               # [ 0  1  2 ... 45]  → 뉴스 주제 46종
print(type(x_train), type(x_train[0]))  # <class 'numpy.ndarray'> <class 'list'>
print(len(x_train[0]), len(x_train[1])) # 87 56

print('뉴스기사의 최대 길이 :', max(len(i) for i in x_train))         # 2376
print('뉴스기사의 최소 길이 :', min(len(i) for i in x_train))         # 13
print('뉴스기사의 평균 길이 :', sum(map(len, x_train))/len(x_train))  # 145.53
```

- `x_train`은 기사마다 길이가 다른 **단어 번호 리스트**의 배열 → Day22에서 Tokenizer로 만든 결과가 이미 들어 있는 형태
- 길이가 13 ~ 2376으로 제각각이라 `pad_sequences`로 맞춰야 한다

**전처리**

```python
x_train = pad_sequences(x_train, maxlen=200)   # padding / truncating 기본값 = 'pre'
x_test = pad_sequences(x_test, maxlen=200)
print(x_train.shape, x_test.shape)             # (8982, 200) (2246, 200)

y_train = to_categorical(y_train)              # 0 ~ 45 → 46칸 One-Hot
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)             # (8982, 46) (2246, 46)
```

- 평균 길이 145 → `maxlen=200`이면 대부분의 기사를 거의 다 담는다
- 기본값 `'pre'` : 앞을 0으로 채우고, 긴 기사는 앞쪽을 자른다
  - LSTM은 **마지막 시점의 상태**를 출력하므로 실제 단어가 뒤에 붙어 있는 `pre`가 유리하다

**모델**

```python
model = Sequential()
model.add(Embedding(1000, 200))              # input_dim = num_words 1000, 단어 하나를 200칸 벡터로
model.add(LSTM(128))                         # (None, 200, 200) → (None, 128)
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))   # 46종 다중 분류

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```

- `input_dim`은 단어 개수 + 1이 아니라 **`num_words=1000`** 그대로
  - 데이터를 불러올 때부터 번호가 0 ~ 999로 잘려 있다 → 가장 큰 번호 + 1 = 1000 (Day22의 `input_dim` 규칙과 같다)

**저장 (ModelCheckpoint + save_weights)**

```python
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

path = './_save/keras62/'
mcp = ModelCheckpoint(
    monitor='val_loss', mode='min',
    save_best_only=True,                       # val_loss 가 가장 좋았던 epoch 의 모델만 덮어쓰며 저장
    filepath=path + 'keras62_1_mcp.keras',
    verbose=1,
)

model.fit(x_train, y_train, epochs=100, batch_size=64,
          validation_split=0.2, callbacks=[es, mcp])

model.save_weights(path + 'keras62_1_save.weights.h5')   # 가중치만 저장
```

- `restore_best_weights=True` → `save_weights`로 저장되는 것도 val_loss가 가장 좋았던 epoch의 가중치 → mcp와 같은 가중치
- `.keras` (구조 + 가중치) / `.weights.h5` (가중치만, 불러올 때 같은 모델을 만든 뒤 `load_weights`)

**평가**

```python
y_predict = np.argmax(model.predict(x_test), axis=1)   # 46칸 확률 → 가장 큰 칸의 번호
y_true = np.argmax(y_test, axis=1)                     # 원핫 정답 → 번호
print("acc_score :", accuracy_score(y_true, y_predict))
```

- 결과 : loss **1.1545**, acc **0.7418**, 64.92초 → 목표 0.67 달성

### 2. IMDB 영화 리뷰 이진 분류 (keras62_2_imdb)

```python
from tensorflow.keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

print(x_train.shape, y_train.shape)    # (25000,) (25000,)
print(x_test.shape, y_test.shape)      # (25000,) (25000,)
print(np.unique(y_train))              # [0 1]

print('리뷰의 최대 길이 :', max(len(i) for i in x_train))         # 2494
print('리뷰의 최소 길이 :', min(len(i) for i in x_train))         # 11
print('리뷰의 평균 길이 :', sum(map(len, x_train))/len(x_train))  # 238.71

x_train = pad_sequences(x_train, maxlen=250)
x_test = pad_sequences(x_test, maxlen=250)
print(x_train.shape, x_test.shape)     # (25000, 250) (25000, 250)
# y 는 이미 0 / 1 → 이진 분류라서 원핫하지 않는다

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=100, input_length=250))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 긍정 / 부정

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
```

- 리뷰가 Reuters보다 길어서(평균 238.71) `maxlen=250`
- 훈련 데이터가 25,000개로 많아서 `batch_size=128`, `patience=5`
- ModelCheckpoint `keras62_2_mcp.keras` + `save_weights('keras62_2_save.weights.h5')`
- 평가 : `np.round(y_predict)` (0.5 기준) → `accuracy_score`
- 결과 : loss **0.3241**, acc **0.8619**, 35.69초 → 목표 0.6 달성

| | Reuters | IMDB |
|---|---|---|
| 문제 | 뉴스 주제 46종 | 리뷰 긍정 / 부정 |
| maxlen | 200 (평균 145.53) | 250 (평균 238.71) |
| y 전처리 | `to_categorical` → (N, 46) | 0 / 1 그대로 |
| 출력층 | `Dense(46, softmax)` | `Dense(1, sigmoid)` |
| loss | `categorical_crossentropy` | `binary_crossentropy` |
| 예측값 변환 | `np.argmax(axis=1)` | `np.round` |
| acc | **0.7418** | **0.8619** |

- 같은 Embedding + LSTM 구조라도 46개 중 고르는 Reuters보다 둘 중 하나를 고르는 IMDB가 정확도가 높다

### 3. Embedding 다시 보기 (keras61_Embedding04_important)

- **`input_dim` = 단어 수 + 1** : Embedding은 번호 → 행으로 찾는 표, 행 번호가 **0 ~ input_dim-1**
  - 단어 번호 1 ~ 31 + padding 0번 = 서로 다른 번호 **32개** → 행 32개 필요
  - `input_dim=31`이면 31번 단어(`'구라친다'`)가 찾을 행이 없다
- **Embedding 다음에 바로 Dense를 붙이면** `(None, 5, 100)` → `(None, 5, 1)`
  - Dense는 **마지막 축(100칸)에만** 적용 → 단어 5개마다 같은 가중치로 답 1개씩 → Param 100 + 1 = **101**
  - 정답은 문장 하나에 1개 → `SimpleRNN(10)`이 단어 5개를 읽고 마지막 상태 하나만 내보내야 `(None, 1)`
- 주석 제목을 `임베딩 Layer 1 / 2 / 3`으로 정리 (수정)

### 4. sparse_categorical_crossentropy (keras63_sparse1 ~ 4)

Day12 ~ 13의 CNN 파일(keras36)을 그대로 가져와 y 원핫만 빼고 loss를 바꿨다.

```python
# 원핫 (기존)                                   # sparse (Day23)
ohe = OneHotEncoder(sparse_output=False)        # → 원핫 부분 삭제
y_train = ohe.fit_transform(y_train.reshape(-1,1))  #   y 는 정답 숫자 그대로 (60000,)
model.compile(loss='categorical_crossentropy')  model.compile(loss='sparse_categorical_crossentropy')
```

- y가 원핫이 아닌 **정답 숫자**여도 loss 안에서 원핫처럼 계산해 준다
- 출력층은 그대로 `Dense(클래스 수, softmax)` → 예측은 여전히 클래스 수만큼의 확률

**평가할 때 차이**

```python
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)   # (10000, 10) → (10000,)
# y_test 는 이미 정답 숫자 → argmax 하지 않는다
acc_score = accuracy_score(y_test, y_predict)
```

| y_test에 한 처리 | 결과 |
|---|---|
| `np.argmax(y_test, axis=1)` | y_test가 1차원 `(10000,)` → `AxisError: axis 1 is out of bounds for array of dimension 1` |
| 둘 다 `axis=0` | `y_predict`는 `(10,)` (클래스마다 확률이 가장 큰 **샘플 번호**), `y_test`는 숫자 1개 → `ValueError: Found input variables with inconsistent numbers of samples: [1, 10]` |
| **argmax 하지 않음** | `(10000,)` vs `(10000,)` → 정상 |

- cifar10 / cifar100의 y는 처음부터 `(N, 1)` 2차원 → sparse loss는 `(N,)` / `(N, 1)` 둘 다 받는다
  - `(10000, 1)`에 `argmax(axis=1)`을 하면 칸이 1개라 **전부 0** → 에러 없이 acc만 틀리게 나온다
- 클래스가 많을수록 원핫을 만들지 않는 sparse가 메모리를 아낀다 (cifar100 원핫 = (50000, 100))

| 파일 | 베이스 | y shape | 출력층 |
|---|---|:---:|:---:|
| keras63_sparse1_mnist | keras36_cnn3_mnist2 | (60000,) | 10 |
| keras63_sparse2_fashion | keras36_cnn4_fashion | (60000,) | 10 |
| keras63_sparse3_cifar10 | keras36_cnn5_cifar10 | (50000, 1) | 10 |
| keras63_sparse4_cifar100 | keras36_cnn6_cifar100 | (50000, 1) | 100 |

**정리**
- `categorical_crossentropy` : y가 원핫 → `y_test`에도 `argmax(axis=1)`
- `sparse_categorical_crossentropy` : y가 정답 숫자 → **`y_predict`에만** `argmax(axis=1)`

### 5. 표 데이터를 LSTM으로 (keras64_LSTM01 ~ 10)

Day14에서 표 데이터를 Conv2D용 4차원으로 바꿨던 keras42_cnn1 ~ 10을 LSTM으로 바꿨다.

```python
# Conv2D (keras42)                              # LSTM (keras64)
x_train = x_train.reshape(-1, 4, 2, 1)          x_train = x_train.reshape(-1, 8, 1)   # (N, timesteps, feature)

model.add(Conv2D(64, (2,2), padding='same',     model.add(LSTM(64, input_shape=(8, 1)))
              input_shape=(4, 2, 1)))
model.add(Conv2D(64, (2,2)))
model.add(Conv2D(32, (2,1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))        model.add(Dense(128, activation='relu'))   # 이후 그대로
```

- 컬럼 하나를 **시점 하나**로 보고 시점마다 값 1개씩 순서대로 읽는다 → `(N, 컬럼 수, 1)`
- LSTM은 마지막 시점의 출력만 내보내서 `(None, 64)` 2차원 → **Flatten 없이** 바로 Dense
- train / val을 직접 나눈 파일(diabetes, ddareung)은 `x_val`도 똑같이 reshape
- digits(8x8 이미지)는 `(N, 8, 8)` → 가로줄 8개 = 시점 8개, 줄마다 픽셀 8개
- ModelCheckpoint 저장 경로 `_save/keras64/keras64_mcpN.keras`

| 파일 | Conv2D 입력 | LSTM 입력 |
|---|:---:|:---:|
| 01 california | (4, 2, 1) | (8, 1) |
| 02 diabetes | (5, 2, 1) | (10, 1) |
| 03 boston | (13, 1, 1) | (13, 1) |
| 04 dacon_ddareung | (3, 3, 1) | (9, 1) |
| 05 kaggle_bike | (4, 2, 1) | (8, 1) |
| 06 cancer | (5, 6, 1) | (30, 1) |
| 07 santander | (10, 20, 1) | (200, 1) |
| 08 wine | (13, 1, 1) | (13, 1) |
| 09 fetch_covtype | (6, 9, 1) | (54, 1) |
| 10 digits | (8, 8, 1) | **(8, 8)** |

### 6. mnist를 함수형 LSTM으로 (keras64_LSTM11_mnist)

keras43_hamsu01_mnist(함수형 DNN)를 LSTM으로 바꿨다.

```python
x_train = (x_train - 127.5)/127.5             # 스케일링을 다시 켬
x_train = x_train.reshape(-1, 28, 28)          # (-1, 784) 대신 3차원

input1 = Input(shape=(28, 28))                 # (784,) → (28, 28)
lstm1 = LSTM(128)(input1)                      # 입력층과 첫 Dense 사이에 LSTM 추가
dense1 = Dense(512, activation='relu')(lstm1)  # (input1) → (lstm1) 뒤에 연결
...                                            # 이후 Dense / Dropout 10층은 베이스 그대로
output1 = Dense(10, activation='softmax')(drop3)
model = Model(inputs=input1, outputs=output1)
```

- 가로줄 28개 = 시점 28개, 줄마다 픽셀 28개 = feature
  - `(N, 784, 1)`로 픽셀을 하나씩 읽으면 시점이 784개라 훨씬 느리다
- 스케일링 : LSTM 안의 tanh / sigmoid는 입력이 0 ~ 255처럼 크면 값이 끝에 몰려(포화) 학습이 거의 안 된다
- `lstm` Param = 4 × (128 × (28 + 128) + 128) = **80,384**
- `dense1` Param : 784 × 512 + 512 = 401,920 → 128 × 512 + 512 = **66,048**
- Total params : 599,178 → **343,690**

### 7. Reshape 층 (keras65_Reshape1, 2)

데이터를 reshape하지 않고 **모델 안에서** 모양을 바꾼다.

**Reshape1 - Dense → Conv2D**

```python
# x 는 (N, 28, 28) 그대로 (4차원 reshape 하지 않음)
model.add(Dense(280, input_shape=(28, 28)))          # (N, 28, 28) → (N, 28, 280)
model.add(Reshape(target_shape=(28, 28, 10)))        # (N, 28, 280) → (N, 28, 28, 10)
model.add(Conv2D(32, (3,3), activation='relu'))      # (N, 26, 26, 32)
```

- Dense는 마지막 축(28칸)에만 적용 → 가로줄 28개마다 같은 가중치로 280칸 → Param = 28 × 280 + 280 = **8120**
- Reshape : 28 × 280 = 28 × 28 × 10 = **7840** → 값 개수가 같아야 바꿀 수 있다, Param **0**
- 다음 Conv2D는 채널 10 → Param = (3 × 3 × 10 + 1) × 32 = **2912**
- Total params : **3,588,898**

**Reshape2 - Conv2D → LSTM → Conv2D**

```python
model.add(Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)))   # 데이터 reshape 대신
model.add(Conv2D(32, (3,3), activation='relu'))                      # (26, 26, 32)
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))          # (24, 24, 32)
model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))          # (20, 20, 32)
model.add(Reshape(target_shape=(20*20, 32)))                         # (N, 400, 32)  4차원 → 3차원
model.add(LSTM(10, return_sequences=True))                           # (N, 400, 10)
model.add(Reshape(target_shape=(20, 20, 10)))                        # (N, 20, 20, 10)  3차원 → 4차원
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))          # (18, 18, 64)
```

- Conv2D 출력의 픽셀 400개(20 × 20)를 시점 400개, 채널 32를 feature로 LSTM에 넣는다
- `return_sequences=True` → 시점 400개의 출력을 모두 내보낸다 `(N, 400, 10)`
- LSTM 출력은 3차원인데 Conv2D는 4차원이 필요 → 다시 `Reshape(20, 20, 10)` (400 = 20 × 20)
- `lstm` Param = 4 × (10 × (32 + 10) + 10) = **1720**
- `conv2d_3` Param = (3 × 3 × 10 + 1) × 64 = **5824** (채널이 32 → LSTM units 10)
- Total params : **3,567,234**

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras2/keras62_1_reuters.py` | `reuters.load_data(num_words=1000, test_split=0.2)`, 기사 길이 확인, `pad_sequences(maxlen=200)`, `to_categorical` 46칸, Embedding(1000, 200) + LSTM(128) softmax, ModelCheckpoint + `save_weights` |
| `keras2/keras62_2_imdb.py` | `imdb.load_data(num_words=1000)`, `pad_sequences(maxlen=250)`, Embedding(1000, 100) + LSTM(64) sigmoid, `np.round` 평가, ModelCheckpoint + `save_weights` |
| `keras2/keras63_sparse1_mnist.py` | keras36_cnn3_mnist2 → `sparse_categorical_crossentropy`, `y_predict`만 argmax |
| `keras2/keras63_sparse2_fashion.py` | keras36_cnn4_fashion → OneHotEncoder 삭제 + sparse |
| `keras2/keras63_sparse3_cifar10.py` | keras36_cnn5_cifar10 → sparse, y (50000, 1) 그대로 |
| `keras2/keras63_sparse4_cifar100.py` | keras36_cnn6_cifar100 → sparse, 출력 100 |
| `keras2/keras64_LSTM01_california.py` ~ `keras64_LSTM10_digits.py` | keras42_cnn1 ~ 10 → `(N, 컬럼 수, 1)` + LSTM(64), digits는 `(N, 8, 8)` |
| `keras2/keras64_LSTM11_mnist.py` | keras43_hamsu01_mnist → 함수형 `Input(shape=(28, 28))` + LSTM(128), 스케일링 |
| `keras2/keras65_Reshape1.py` | Dense(280) → `Reshape(28, 28, 10)` → Conv2D |
| `keras2/keras65_Reshape2.py` | `Reshape(28, 28, 1)` → Conv2D → `Reshape(400, 32)` → LSTM(return_sequences) → `Reshape(20, 20, 10)` → Conv2D |
| `keras2/keras61_Embedding04_important.py` | 주석 제목 `임베딩 Layer 1 / 2 / 3`으로 정리 (수정) |
| `docs/Day23.md` | Day23 학습 기록 신규 작성 |
| `docs/Day22.md` | 하단 nav에 Day23 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day23, `_save/keras62 · keras64`, 설명 열 정렬) / 진행도(23일, 29%) 갱신 (수정) |

---

## 📊 실행 결과

### 텍스트 분류

| 파일 | 데이터 | loss | acc | 시간 | 목표 |
|---|---|:---:|:---:|:---:|:---:|
| keras62_1_reuters | 뉴스 46종 | 1.1545 | **0.7418** | 64.92초 | 0.67 ✅ |
| keras62_2_imdb | 리뷰 긍정 / 부정 | 0.3241 | **0.8619** | 35.69초 | 0.6 ✅ |

### 표 데이터 : DNN vs Conv2D vs LSTM (GPU)

| 파일 | 지표 | DNN | Conv2D (keras42) | LSTM (keras64) | LSTM 시간 (Conv2D 시간) |
|---|:---:|:---:|:---:|:---:|:---:|
| 01 california | r2 | **0.8014** | 0.7998 | 0.8010 | - |
| 02 diabetes | r2 | 0.4236 | **0.4498** | 0.4294 | - |
| 03 boston | r2 | 0.7554 | **0.7726** | 0.7069 | - |
| 04 dacon_ddareung | r2 | 0.6298 | **0.7108** | 0.6562 | 35.75초 (22.04초) |
| 05 kaggle_bike | r2 | 0.2841 | 0.3048 | **0.3318** | 70.85초 (45.25초) |
| 06 cancer | acc | 0.9591 | **0.9649** | 0.9123 | 7.91초 (5.99초) |
| 07 santander | acc | 0.9087 | **0.9110** | 0.8995 | 369.84초 (84.21초) |
| 08 wine | acc | **0.9815** | 0.9630 | 0.9259 | 7.37초 (6.49초) |
| 09 fetch_covtype | acc | 0.8924 | **0.9061** | 0.8224 | 499.69초 (248.45초) |
| 10 digits | acc | 0.9555 | 0.9708 | **0.9722** | 67.96초 (132.79초) |

- 04 ddareung의 DNN 값은 함수형 + Dropout 기록 (CPU / GPU 구분 없음)

**읽는 법**
- 표 데이터는 컬럼 순서에 의미가 없어서 LSTM이 대부분 Conv2D보다 낮거나 비슷했다
- LSTM이 가장 좋았던 것은 kaggle_bike · digits, california는 DNN과 거의 같았다
  - digits는 가로줄 단위로 읽는 이미지라 순서에 의미가 있다
- 시점이 많을수록 느리다 : santander(시점 200) 4.4배, covtype(시점 54) 2배
  - digits는 시점 8개 `(8, 8)`이라 Conv2D보다 빨랐다

### 모델 summary (Total params)

| 파일 | 구성 | Total params |
|---|---|:---:|
| keras64_LSTM11_mnist | 함수형 LSTM(128) + Dense 11층 | 343,690 |
| keras65_Reshape1 | Dense(280) → Reshape → Conv2D 7층 | 3,588,898 |
| keras65_Reshape2 | Reshape → Conv2D 3층 → Reshape → LSTM(10) → Reshape → Conv2D 4층 | 3,567,234 |

---

## 💻 핵심 개념

### keras 텍스트 데이터셋 흐름

```
load_data(num_words=1000) ──▶ 단어 번호 리스트 (길이 제각각)
        │
        ▼
pad_sequences(maxlen) ──▶ (N, maxlen)    기본값 pre : 앞을 0으로, 긴 문장은 앞쪽을 자름
        │
        ▼
Embedding(input_dim=num_words, output_dim) ──▶ (N, maxlen, output_dim)
        │
        ▼
LSTM ──▶ (N, units) ──▶ Dense ──▶ softmax(다중) / sigmoid(이진)
```

### categorical vs sparse

```python
# categorical_crossentropy
y = to_categorical(y)                                    # (N, 클래스 수)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)                       # y_test 도 되돌림

# sparse_categorical_crossentropy
# y 는 정답 숫자 그대로                                   # (N,) 또는 (N, 1)
y_predict = np.argmax(y_predict, axis=1)                 # 예측만 argmax
```

### LSTM 입력 만들기

```
표 데이터 (N, 컬럼 수)       ──reshape──▶ (N, 컬럼 수, 1)     컬럼 하나 = 시점 하나
이미지    (N, 가로, 세로)    ──그대로──▶ (N, 가로, 세로)     가로줄 = 시점, 줄의 픽셀 = feature
```

### Reshape 층

```python
Reshape(target_shape=(...))      # 배치(N)를 뺀 모양, 값 개수가 같아야 한다, Param 0

(N, 28, 280)    → Reshape(28, 28, 10) → (N, 28, 28, 10)   28 x 280 = 28 x 28 x 10
(N, 20, 20, 32) → Reshape(400, 32)    → (N, 400, 32)      Conv2D → LSTM
(N, 400, 10)    → Reshape(20, 20, 10) → (N, 20, 20, 10)   LSTM → Conv2D
```

### LSTM 파라미터

```
Param = 4 × (units × (feature + units) + units)
LSTM(128), feature 28 → 4 × (128 × 156 + 128) = 80,384
LSTM(10),  feature 32 → 4 × (10 × 42 + 10)    = 1,720
```

---

## 성과

- `reuters.load_data(num_words=1000, test_split=0.2)`로 뉴스 46종 데이터를 불러와 길이(최대 2376 / 최소 13 / 평균 145.53)를 확인
- `pad_sequences(maxlen=200)` + `to_categorical` + Embedding(1000, 200) + LSTM(128)으로 Reuters acc **0.7418** (목표 0.67)
- `imdb.load_data(num_words=1000)` + `pad_sequences(maxlen=250)` + sigmoid로 IMDB acc **0.8619** (목표 0.6)
- 두 모델에 ModelCheckpoint(`save_best_only`)와 `save_weights`를 적용해 `.keras` / `.weights.h5` 저장
- `num_words`로 단어 번호 범위가 정해지면 Embedding `input_dim` = `num_words`임을 확인
- mnist · fashion · cifar10 · cifar100에 `sparse_categorical_crossentropy`를 적용하고, `y_test`에 argmax를 하면 생기는 AxisError / ValueError 원인 확인
- 표 데이터 10종을 `(N, 컬럼 수, 1)` LSTM으로 바꿔 DNN · Conv2D 결과와 비교, 시점 수에 따라 훈련 시간이 늘어나는 것 확인
- mnist를 함수형 `Input(shape=(28, 28))` + LSTM(128)으로 바꾸고 Param 80,384 / Total 343,690 확인
- `Reshape` 층으로 Dense → Conv2D (Total 3,588,898), Conv2D → LSTM → Conv2D (Total 3,567,234) 연결

---

## 💡 주요 학습 포인트

1. **keras 텍스트 데이터셋은 이미 단어 번호로 되어 있다**: Tokenizer 없이 바로 `pad_sequences`부터 한다
2. **`num_words`는 많이 나온 단어만 남긴다**: 1000이면 단어 번호가 0 ~ 999
3. **`input_dim`은 가장 큰 번호 + 1**: `num_words=1000`이면 `input_dim=1000`
4. **`test_split`으로 test 비율을 정한다**: reuters `load_data`의 인자
5. **기사 길이를 먼저 확인하고 `maxlen`을 정한다**: 평균보다 조금 크게 (145 → 200, 238 → 250)
6. **`pad_sequences` 기본값은 `pre`**: 실제 단어가 뒤에 있어 LSTM 마지막 상태에 유리하다
7. **다중 분류는 softmax + categorical, 이진 분류는 sigmoid + binary**: 예측값은 argmax / round로 되돌린다
8. **`save_weights`는 가중치만 저장한다**: `restore_best_weights=True`면 최고 epoch 가중치가 저장된다
9. **Embedding 다음 Dense는 단어마다 답을 낸다**: 문장 하나에 답 하나는 RNN이 단어들을 하나로 모아야 한다
10. **sparse는 y 원핫을 생략한다**: loss 안에서 원핫처럼 계산하고 출력층은 그대로
11. **sparse에서는 `y_test`를 argmax 하지 않는다**: 1차원이면 AxisError, axis=0으로 바꾸면 뜻이 완전히 달라진다
12. **`(N, 1)` 정답에 argmax를 하면 전부 0**: 에러 없이 acc만 틀려서 더 위험하다
13. **클래스가 많을수록 sparse가 유리하다**: cifar100 원핫은 (50000, 100)
14. **표 데이터 LSTM은 `(N, 컬럼 수, 1)`**: 컬럼 하나를 시점 하나로 본다
15. **LSTM 출력은 2차원이라 Flatten이 필요 없다**: `return_sequences=False`면 마지막 시점만 내보낸다
16. **x_val도 같은 모양으로 바꾼다**: 직접 나눈 검증 데이터를 빠뜨리지 않는다
17. **순서에 의미가 없는 표 데이터는 LSTM 효과가 작다**: 대부분 Conv2D보다 낮거나 비슷했다
18. **시점이 많을수록 LSTM은 느리다**: 시점마다 순서대로 계산해서 santander(200)는 Conv2D의 4.4배
19. **이미지는 가로줄을 시점으로**: digits `(8, 8)`, mnist `(28, 28)` → 시점이 적어 빠르다
20. **LSTM 입력은 스케일링한다**: tanh / sigmoid가 큰 입력에서 포화된다
21. **함수형에서 LSTM 추가는 연결만 바꾸면 된다**: `lstm1 = LSTM(128)(input1)`, `dense1 = Dense(...)(lstm1)`
22. **LSTM Param = 4 × (units × (feature + units) + units)**: 게이트 4개
23. **Reshape는 모델 안에서 모양만 바꾼다**: 값 개수가 같아야 하고 Param은 0
24. **Dense는 3차원 입력의 마지막 축에만 적용된다**: (N, 28, 28) → Dense(280) → (N, 28, 280), Param 8120
25. **Conv2D와 LSTM은 Reshape로 이어 붙인다**: 4차원 ↔ 3차원, `return_sequences=True`로 시점 출력을 모두 남긴다
26. **LSTM 뒤에 Conv2D가 오면 다시 4차원으로**: Conv2D는 `(가로, 세로, 채널)`이 필요하다

---

[⬅️ Day22](Day22.md) · [🏠 전체 목차](../README.md)
