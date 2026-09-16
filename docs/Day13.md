# Day13 - CNN 이미지 분류 실습 (mnist / fashion / cifar10 / cifar100)과 padding · strides · MaxPooling

**학습 기간:** 2026-09-16

> Day12에서 구조만 확인한 CNN을 **mnist / fashion_mnist / cifar10 / cifar100 네 데이터셋으로 끝까지 훈련**했고, 출력 크기를 조절하는 **padding · strides · MaxPooling2D**를 배웠다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 4차원 reshape | Conv2D 입력은 `(장수, 가로, 세로, 채널)` - 흑백 mnist는 `reshape(-1, 28, 28, 1)` 로 채널을 붙인다 |
| OneHotEncoder | 2차원만 받는다 - `(60000,)` 를 넣으면 `ValueError: Expected 2D array` -> `reshape(-1, 1)` |
| Flatten | Conv2D의 4차원 출력을 Dense에 넣기 전에 2차원으로 펴준다 |
| EarlyStopping 기준 변경 | `monitor='val_acc'` 로 바꾸면 `mode='max'` 도 같이 바꾼다 |
| CNN은 GPU가 빠르다 | mnist1 CPU 557초 / GPU 141초 (3.9배), fashion CPU 951초 / GPU 381초 (2.5배) |
| 컬러 이미지 | cifar는 `(50000, 32, 32, 3)` 4차원으로 들어와 reshape가 필요 없다 |
| padding | `'valid'`(default) 는 줄어들고, `'same'` 은 크기를 유지한다 |
| strides | 커널이 움직이는 칸 수 (default 1). `2` 면 크기가 절반 |
| MaxPooling2D | 2×2 구역에서 최대값만 남겨 크기를 절반으로 줄인다. 파라미터 0 |

---

## 📖 핵심 학습 내용

### 1. mnist를 CNN으로 끝까지 훈련 (keras36_cnn3_mnist1)

- Day12의 `keras36_cnn3_mnist.py`(스케일링까지)를 `keras36_cnn3_mnist1.py` 로 이어서 모델 / 훈련 / 평가까지 완성
- **x는 4차원으로 reshape** : `mnist.load_data()` 는 `(60000, 28, 28)` 3차원이라 채널 1을 붙여 `(60000, 28, 28, 1)` 로 만든다
- **y는 OneHotEncoder로 원핫** : `(60000,)` 1차원을 그대로 넣으면 에러
  - `ValueError: Expected 2D array, got 1D array instead`
  - `y_train.reshape(-1, 1)` 로 2차원을 만든 뒤 `fit_transform` -> `(60000, 10)`
- **Flatten** : Conv2D 출력 `(20, 20, 16)` 을 `6400` 으로 펴서 Dense에 넣는다
  - Flatten과 Dense를 붙이기 전에는 Conv2D 4개까지만 쌓아 summary를 확인했다 (Total params 25,296)
  - Flatten + Dense 추가 후 Total params 232,906 - 그중 `dense` 한 층이 6400 × 32 + 32 = 204,832
- padding 없는 Conv2D는 `(커널 크기 - 1)` 만큼 줄어든다 : `(3,3)` -> 2씩, `(2,2)` -> 1씩
- `filters=`, `kernel_size=`, `units=` 는 인자의 이름이라 이름을 써도 되고 순서로 넣어도 된다
- 평가 : `np.argmax(axis=1)` 로 원핫과 확률을 숫자로 되돌려 `accuracy_score` 계산
- **EarlyStopping 없이 epochs=50 고정 -> CPU / GPU 시간 비교**
  - CPU 557.11초 / GPU 141.31초 -> **GPU가 약 3.9배 빨랐다**
  - Day12의 Dense 모델에서는 CPU가 이긴 경우가 많았지만, 연산량이 큰 Conv2D에서는 GPU가 확실히 유리했다

### 2. mnist 성능 올리기 (keras36_cnn3_mnist2)

- 목표 acc 0.995 를 두고 mnist1 을 수정
  - Conv2D 7층, 필터 32 -> 64 -> 128, `(5,5)` 커널을 섞어 크기를 빠르게 줄임
  - Dense 256 / 128, Dropout 0.25 ~ 0.4
  - `validation_split` 0.2 -> 0.1
- **EarlyStopping 기준을 `val_acc` 로 변경**
  - 목표가 정확도라서 기준도 정확도로 잡았다. val_loss 최저점 epoch 와 val_acc 최고점 epoch 는 다르다
  - acc는 높을수록 좋으므로 **`mode='max'`** 를 같이 지정
  - `patience=25` 로 넉넉히 줬다
- 결과 : CPU / GPU 모두 **acc 0.9938** (mnist1 0.9893 -> 0.9938), 목표 0.995 에는 미달
- 멈춘 epoch 가 CPU 54 / GPU 70 으로 달랐다 -> 훈련량이 달라 이 파일의 시간(CPU 1861초 / GPU 322초)은 순수한 속도 비교가 아니다

### 3. fashion_mnist + padding / strides 적용 (keras36_cnn4_fashion)

- fashion_mnist 는 mnist와 모양이 같다 (28×28 흑백 6만 장, 정답 0~9) - 옷 / 신발 / 가방 사진
- Conv2D 전 층에 **`padding='same'`** -> 크기 유지, 줄이고 싶은 층에만 **`strides=(2,2)`** -> 절반
  - `28 -> 28 -> 14 -> 14 -> 14 -> 7 -> 7`
  - Flatten 6272, Total params 1,907,018 (mnist2 3,578,186 의 절반 수준)
- 목표 acc 0.92 -> **CPU 0.9262 / GPU 0.9258 로 달성**
- epochs=50 고정 -> CPU 951.0초 / GPU 380.58초 -> **GPU 2.5배 빠름**, acc는 거의 같음
- `plt.imshow(x_train[50000], 'gray')` 로 실제 이미지 확인

### 4. cifar10 - 컬러 이미지 (keras36_cnn5_cifar10)

- 32×32 **컬러(3채널)** 5만 장, 10종
- `load_data()` 가 처음부터 `(50000, 32, 32, 3)` 4차원 -> **x reshape 불필요**
- y 도 처음부터 `(50000, 1)` 2차원 -> OneHotEncoder에 바로 넣을 수 있다
- R / G / B 모두 0~255 라 `(x - 127.5)/127.5` 한 번으로 전 채널이 스케일링된다
- 첫 Conv2D 파라미터가 채널 수만큼 늘어난다 : `(3×3×3 + 1) × 32 = 896` (흑백이면 320)
- mnist2와 같은 층 구성인데 입력이 32×32 라 Flatten 이 25088, Total params 6,724,490
- 목표 acc 0.67 -> **GPU 0.7267 로 달성** (415.61초)
- 같은 구조로 mnist 0.9938 / cifar10 0.7267 -> 컬러 사진이 손글씨보다 훨씬 어렵다

### 5. cifar100 - 100종 분류 (keras36_cnn6_cifar100)

- cifar10과 이미지는 같고 정답이 0~99 -> 마지막 `Dense(100)`, 원핫 `(50000, 100)`
- 1차 (MaxPooling 없음, Total params 6,736,100) : **acc 0.2693** (GPU 394.21초), 목표 0.4 미달
  - 클래스는 10배, 장수는 그대로 -> 한 종류당 이미지 5000장 -> 500장
- 이후 모델에 **MaxPooling2D 2개**를 넣고 Conv2D 구성을 조정
  - `32 -> 30 -> 28 -> (pool) 14 -> 12 -> 10 -> 8 -> 6 -> (pool) 3`
  - Flatten 1152, **Total params 627,972** (1차 대비 약 1/10)
  - MaxPooling을 너무 많이 넣으면 가로 세로가 1×1 까지 내려가고, 그 뒤에 또 넣으면 `Negative dimension size` 에러가 난다

### 6. Conv2D 인자 의미 확인 (keras37_cnn_identify)

- keras36_cnn1 과 같은 모델로 `Conv2D(10, (2,2), input_shape=(5,5,1))` 의 각 숫자를 정리
  - `10` = 필터 개수 = 출력 채널 수 / `(2,2)` = kernel_size / `input_shape` = (가로, 세로, 채널), 첫 층에만
- 파라미터 수의 `+ 1` 은 필터마다 붙는 bias

### 7. padding과 strides (keras38_padding_stride_0)

- **padding**
  - `'valid'` (default) : 패딩 없음 -> `입력 - 커널 + 1` 로 줄어든다
  - `'same'` : 가장자리에 0을 둘러 입력과 같은 크기 유지 (strides=1 기준) -> 가장자리 픽셀 정보도 덜 버려진다
- **strides**
  - 커널이 한 번에 움직이는 칸 수 (default 1)
  - `strides=2` -> 크기가 절반, 대신 훑는 영역이 겹치지 않아 정보가 듬성듬성해진다
  - 나누어떨어지지 않는 자투리는 버려진다
- 출력 크기
  - `same` : `ceil(입력 / strides)` -> `(10,10)` 입력, strides 2 -> `5`
  - `valid` : `floor((입력 - 커널) / strides) + 1` -> `(5,5)` 입력, 커널 3, strides 2 -> `2`
- 파라미터 수는 padding / strides 와 무관 (Total params 869)

### 8. MaxPooling2D (keras39_MaxPolling0)

- 2×2 구역마다 **가장 큰 값 하나만 남긴다** -> 가로 세로 절반 (`pool_size` default `(2,2)`, strides 기본값 = pool_size)
- 계산 없이 고르기만 하므로 **파라미터 0**
- 통상적으로 Conv2D 다음에 붙인다
- `Conv2D(padding='same')` `(10,10,10)` -> `MaxPooling2D` `(5,5,10)` -> `Conv2D(strides=2)` `(2,2,9)`
- keras38(strides로 줄임)과 Total params 869 로 같다
  - strides=2 : 건너뛰며 계산 / MaxPooling : 전부 계산한 뒤 최대값만 남김

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras36_cnn3_mnist1.py` | mnist CNN 첫 훈련 - 4차원 reshape, OneHotEncoder 2D 에러, Flatten, CPU 557초 / GPU 141초, acc 0.9893 / 0.9897 |
| `keras36_cnn3_mnist2.py` | mnist 튜닝 - Conv2D 7층, EarlyStopping `val_acc` + `mode='max'`, acc 0.9938 (목표 0.995) |
| `keras36_cnn4_fashion.py` | fashion_mnist - `padding='same'` + `strides=(2,2)`, acc 0.9262 (목표 0.92 달성), CPU 951초 / GPU 381초 |
| `keras36_cnn5_cifar10.py` | cifar10 컬러 이미지 - `(32,32,3)` 입력, acc 0.7267 (목표 0.67 달성) |
| `keras36_cnn6_cifar100.py` | cifar100 100종 - 1차 acc 0.2693 (목표 0.4), 이후 MaxPooling2D 2개 적용 (Total params 627,972) |
| `keras37_cnn_identify.py` | `Conv2D(10, (2,2), input_shape=(5,5,1))` 각 인자의 의미와 파라미터 계산 정리 |
| `keras38_padding_stride_0.py` | padding / strides 에 따른 출력 크기 - `(5,5,10)` -> `(2,2,9)`, Total params 869 |
| `keras39_MaxPolling0.py` | `MaxPooling2D` - `(10,10,10)` -> `(5,5,10)`, 파라미터 0 |
| `keras36_cnn1.py` | Conv2D 인자 설명 주석 보완 (수정) |
| `keras36_cnn3_mnist.py` | `keras36_cnn3_mnist1.py` 로 이어서 작성하고 삭제 (삭제) |
| `docs/Day13.md` | Day13 학습 기록 신규 작성 |
| `docs/Day12.md` | 하단 nav 에 Day13 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조 / 진행도(13일, 16.25%) 갱신 (수정) |

---

## 📊 실행 결과

### 데이터셋별 결과

| 데이터셋 | 입력 | 클래스 | 목표 acc | 결과 acc | 달성 |
|---|---|:---:|:---:|:---:|:---:|
| mnist (mnist1) | 28×28×1 | 10 | - | 0.9893 (CPU) / 0.9897 (GPU) | - |
| mnist (mnist2) | 28×28×1 | 10 | 0.995 | 0.9938 | ❌ |
| fashion_mnist | 28×28×1 | 10 | 0.92 | 0.9262 (CPU) / 0.9258 (GPU) | ✅ |
| cifar10 | 32×32×3 | 10 | 0.67 | 0.7267 (GPU) | ✅ |
| cifar100 (1차) | 32×32×3 | 100 | 0.4 | 0.2693 (GPU) | ❌ |

### CPU vs GPU (CNN)

| 파일 | 훈련량 | CPU | GPU | 배율 |
|---|---|---:|---:|:---:|
| mnist1 | epochs 50 고정 | 557.11초 | **141.31초** | 3.9배 |
| fashion | epochs 50 고정 | 951.0초 | **380.58초** | 2.5배 |
| mnist2 | ES (CPU 54 / GPU 70 epoch) | 1861.89초 | 322.36초 | 훈련량이 달라 비교 불가 |

**읽는 법:** Day12의 Dense 모델은 10개 중 6개에서 CPU가 빨랐지만, Conv2D를 쌓은 CNN은 두 경우 모두 GPU가 2.5 ~ 3.9배 빨랐다. acc는 CPU와 GPU가 거의 같다.

### 출력 크기와 파라미터 수

| 모델 | 크기를 줄인 방법 | Flatten | Total params |
|---|---|---:|---:|
| mnist2 | padding 없는 Conv2D | 12,800 | 3,578,186 |
| fashion | padding='same' + strides=2 | 6,272 | 1,907,018 |
| cifar10 | padding 없는 Conv2D | 25,088 | 6,724,490 |
| cifar100 (1차) | padding 없는 Conv2D | 25,088 | 6,736,100 |
| cifar100 (MaxPooling 적용) | Conv2D + MaxPooling2D 2개 | 1,152 | 627,972 |

-> 파라미터 대부분은 **Flatten 바로 뒤 Dense** 에서 나온다. 크기를 일찍 줄일수록 Flatten이 작아져 모델이 가벼워진다.

---

## 💻 핵심 개념

### 이미지 데이터 준비 (흑백)

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()   # (60000, 28, 28) (60000,)

x_train = (x_train - 127.5)/127.5                           # -1 ~ 1
x_train = x_train.reshape(-1, 28, 28, 1)                    # Conv2D 는 4차원 입력 -> 채널 1 추가

ohe = OneHotEncoder(sparse_output=False)
# y_train = ohe.fit_transform(y_train)                      # ValueError: Expected 2D array, got 1D array
y_train = y_train.reshape(-1, 1)                            # (60000,) -> (60000, 1)
y_train = ohe.fit_transform(y_train)                        # (60000, 10)
```

### 컬러 이미지는 reshape 가 필요 없다

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3) (50000, 1)  <- 처음부터 4차원 / 2차원

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
# (30, 30, 32)  param 896 = (3 x 3 x 3 + 1) x 32   <- 입력 채널이 3
```

### Conv2D -> Flatten -> Dense

```python
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))   # (26, 26, 32)
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))                # (24, 24, 32)
model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))                # (20, 20, 32)
model.add(Dropout(0.25))
...
model.add(Flatten())                           # 4차원 -> 2차원 (값과 순서 그대로)
model.add(Dense(units=256, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

### EarlyStopping 을 정확도 기준으로

```python
es = EarlyStopping(
    monitor='val_acc',          # 목표가 정확도
    mode='max',                 # acc 는 높을수록 좋다
    patience=25,
    restore_best_weights=True,
    verbose=1,
)
```

### padding / strides

```python
model.add(Conv2D(10, (2,2), input_shape=(10,10,1),
                 strides=2,           # default 1
                 padding='same',      # default 'valid'
                 ))                   # (None, 5, 5, 10)   ceil(10 / 2)

model.add(Conv2D(filters=9, kernel_size=(3,3),
                 strides=2,
                 padding='valid',
                 ))                   # (None, 2, 2, 9)    floor((5 - 3) / 2) + 1

# same  : ceil(입력 / strides)
# valid : floor((입력 - 커널) / strides) + 1
# 파라미터 수는 padding / strides 와 무관 -> Total params: 869
```

### MaxPooling2D

```python
from tensorflow.keras.layers import MaxPooling2D

model.add(Conv2D(10, (2,2), input_shape=(10,10,1), strides=1, padding='same'))   # (None, 10, 10, 10)
model.add(MaxPooling2D())      # (None, 5, 5, 10)  2x2 중 최대값만 남김, param 0
model.add(Conv2D(filters=9, kernel_size=(3,3), strides=2, padding='valid'))      # (None, 2, 2, 9)
```

### cifar100 에 MaxPooling 적용

```python
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))   # (30, 30, 32)
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))                # (28, 28, 32)
model.add(MaxPooling2D())                                                  # (14, 14, 32)
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))                # (12, 12, 64)
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))                # (10, 10, 64)
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))               # (8, 8, 128)
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))               # (6, 6, 128)
model.add(MaxPooling2D())                                                  # (3, 3, 128)
model.add(Dropout(0.25))
model.add(Flatten())                                                       # 1152
# Total params: 627,972  (MaxPooling 없던 1차 6,736,100)
```

---

## 성과

- mnist / fashion_mnist / cifar10 / cifar100 네 가지 이미지 데이터셋을 CNN으로 끝까지 훈련하고 평가
- fashion_mnist(0.9262), cifar10(0.7267) 목표 정확도 달성
- mnist1 / fashion 을 epochs 고정으로 CPU·GPU 두 번 돌려 **CNN에서는 GPU가 2.5 ~ 3.9배 빠르다**는 것을 확인
- OneHotEncoder 1차원 입력 에러를 `reshape(-1, 1)` 로 해결
- EarlyStopping 을 `val_acc` + `mode='max'` 로 바꿔 적용
- padding / strides / MaxPooling2D 의 출력 크기 변화를 summary 로 검증
- cifar100 에 MaxPooling2D 를 적용해 파라미터를 6,736,100 -> 627,972 로 줄임

---

## 💡 주요 학습 포인트

1. **Conv2D 입력은 4차원**: `(장수, 가로, 세로, 채널)` - 흑백은 `reshape(-1, 28, 28, 1)`, 컬러(cifar)는 처음부터 4차원
2. **OneHotEncoder 는 2차원만 받는다**: `(N,)` 은 `reshape(-1, 1)` 후 넣는다
3. **Flatten**: Conv2D 출력을 Dense 에 넣기 전에 2차원으로 편다
4. **padding 없는 Conv2D 는 (커널 - 1) 만큼 줄어든다**: `(3,3)` -> 2, `(5,5)` -> 4
5. **`padding='same'`**: 크기를 유지한다. default 는 `'valid'`
6. **`strides=2`**: 크기가 절반이 되지만 건너뛴 자리는 보지 않는다
7. **MaxPooling2D**: 최대값만 남겨 절반으로 줄인다. 파라미터 0. 너무 많이 넣으면 `Negative dimension size` 에러
8. **파라미터는 Flatten 뒤 Dense 에 몰린다**: 크기를 일찍 줄이면 모델이 가벼워진다
9. **EarlyStopping 을 `val_acc` 로 잡으면 `mode='max'`**
10. **CNN 은 GPU 가 확실히 빠르다**: mnist1 3.9배, fashion 2.5배 (Dense 모델과 반대 결과)
11. **컬러 첫 층 파라미터**: 입력 채널이 3이라 `(3×3×3 + 1) × 32 = 896`
12. **클래스가 늘면 어려워진다**: 같은 구조로 cifar10 0.7267 -> cifar100 0.2693

---

[⬅️ Day12](Day12.md) · [🏠 전체 목차](../README.md)
