# Day15 - 이미지 DNN 함수형 전환, ImageDataGenerator로 폴더 이미지 읽기, npy 저장/불러오기

**학습 기간:** 2026-09-18

> Day14의 DNN 모델을 **함수형(Model)** 으로 바꿔 mnist / fashion / cifar10 / cifar100에 적용했고, **ImageDataGenerator**로 내 폴더에 있는 이미지(brain MRI, 고양이/개)를 직접 읽어 이진 분류를 했다. 마지막으로 읽는 데 오래 걸리는 이미지를 **npy로 저장해 두고 불러 쓰는 방법**으로 데이터 준비 시간을 줄였다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 함수형 전환 | Sequential ↔ 함수형은 모델 종류가 아니라 `#2. 모델 구성`을 쓰는 문법 차이다 |
| 변환 검증 | 층을 그대로 옮겼으면 `model.summary()`의 Total params가 같아야 한다 |
| ImageDataGenerator | 폴더 구조(클래스별 하위 폴더)를 그대로 읽어 x, y를 만들어 준다 |
| flow_from_directory | `target_size`로 크기를 맞추고 `class_mode='binary'`로 0/1 정답을 만든다 |
| DirectoryIterator | `xy_train[0][0]`이 x, `xy_train[0][1]`이 y. `[0]`을 꺼내는 순간 이미지를 실제로 읽는다 |
| 이진 분류 출력층 | `Dense(1, sigmoid)` + `binary_crossentropy` + 예측은 `np.round` (argmax 아님) |
| batch_size | '한 번에 GPU에 올릴 장수'다. 전체 장수만큼 주면 OOM이 난다 |
| ModelCheckpoint 기준 | es와 mcp의 `monitor`가 다르면 서로 다른 epoch를 고른다 |
| patience와 epochs | `patience`가 `epochs`보다 크면 ES가 발동하지 않아 `restore_best_weights`도 동작하지 않는다 |
| npy 저장/불러오기 | 한 번 읽은 이미지를 `np.save` 해 두면 다음부터 `np.load`로 수십 배 빠르게 쓴다 |
| 환경 차이 | `ImageDataGenerator`는 Keras 3에서 삭제됐다 → TF 2.9 환경에서 실행 |

---

## 📖 핵심 학습 내용

### 1. DNN을 함수형으로 전환 (keras43_hamsu01 ~ 04)

Day14에서 만든 `keras41_dnn1~4`(Dense만 쓰는 DNN)를 함수형으로 옮겼다.

- **바뀌는 건 `#2. 모델 구성` 뿐**이다. Dense든 Conv2D든 상관없이 문법만 다르다

| | Sequential | 함수형 |
|---|---|---|
| 입력 | 첫 층에 `input_shape=` | 맨 위에 `Input(shape=...)` 층을 따로 만든다 |
| 층 연결 | `model.add()` 로 자동 | 층마다 뒤에 `(앞층)` 을 붙여 직접 연결 |
| 모델 완성 | `Sequential()` | `Model(inputs=input1, outputs=output1)` |

- **변수 이름은 층마다 새로 붙인다**. 같은 이름을 다시 쓰면 그 변수가 덮어써져서 해당 층이 연결에서 빠진다
- **Dropout 다음 층은 Dropout에 연결**한다 (`Dense(64)(drop1)`). 실수로 `(dense4)`라고 쓰면 Dropout이 건너뛰어진다
- 검증은 `model.summary()`의 **Total params**로 한다. mnist는 **599,178** 로 Sequential과 같았다
- 결과 : mnist 0.9784 / fashion 0.8788 / cifar10 0.519 / cifar100 0.2972 → 층이 같으므로 Sequential DNN과 비슷한 수준이다
- cifar100(keras43_hamsu04)은 `Dense → BatchNormalization → Dropout` 이 섞여 있어 `(dense1)`, `(bn1)`, `(drop1)` 로 이어 붙이는 순서를 특히 주의해야 한다. Total params **4,201,316** 으로 Sequential과 같았다

### 2. keras41_dnn4(cifar100 DNN) 구조 수정

- 기존 모델은 `256 → ... → 16 → Dense(100)` 이라 **마지막 은닉층이 16개**였다
  - 100종을 숫자 16개로 구분해야 하는 병목이라 acc가 크게 떨어진다 (dnn1/dnn2는 10종이라 통했다)
- dnn3(cifar10)와 같은 구성으로 맞췄다
  - `Dense(1024) → 512·512 → 256·256 → 128·128 → Dense(100)`, 묶음마다 `BatchNormalization` + Dropout
  - **출력이 100칸이라 마지막 은닉층을 128로 끝냈다**
  - Total params 4,201,316 (그중 첫 층 `3072 × 1024 + 1024 = 3,146,752` 가 75%)
  - BatchNormalization의 이동평균 / 이동분산은 훈련으로 배우지 않아 **Non-trainable params 3,840** 으로 따로 표시된다
- 결과 : **acc 0.299** (201.95초) → CNN(GAP 0.4621)에는 못 미친다

### 3. ImageDataGenerator로 내 폴더의 이미지 읽기 (keras44_ImageDataGenerator1)

- mnist / cifar처럼 이미 준비된 데이터가 아니라 **내 폴더에 있는 사진**을 읽는 방법
- 폴더 구조가 곧 정답이 된다 : `train/ad/`, `train/normal/` → 폴더 이름 알파벳순으로 0, 1
```
_data/image/brain/
├── train/ (ad 80장 + normal 80장 = 160장)
└── test/  (ad 60장 + normal 60장 = 120장)
```
- `ImageDataGenerator(rescale=1./255)` : **rescale은 필수**(0~255 → 0~1). 나머지 옵션(뒤집기 / 이동 / 회전 / 확대 / 기울이기)은 원 데이터를 왜곡하므로 이번엔 주석 처리
- `flow_from_directory()` 로 읽는다
  - `target_size=(100,100)` : 사진 크기가 제각각이어도 이 크기로 맞춰 준다
  - `class_mode='binary'` : y가 0 / 1 한 칸
  - `color_mode='grayscale'` → `(batch, x, y, 1)` / `'rgb'` → `(batch, x, y, 3)`
  - 실행하면 `Found 160 images belonging to 2 classes` 가 찍힌다
- 결과물은 `DirectoryIterator` 라는 **반복자**다
  - `xy_train[0]` = 첫 batch, `xy_train[0][0]` = x, `xy_train[0][1]` = y
  - `print(xy_train.next())` 로 다음 batch를 하나씩 꺼내 볼 수 있다
  - batch_size=10, 160장이면 길이가 16이라 `xy_train[16]` 은 `ValueError: Asked to retrieve element 16, but the Sequence has length 16`
  - `type` : `xy_train[0]` 은 tuple(수정 불가), `xy_train[0][0]` 은 numpy 배열
- **batch_size를 전체 장수로 주면** `xy_train[0][0]` 하나에 전 데이터가 담긴다 → 이 방식으로 x_train / y_train을 만들었다

### 4. brain 이미지 이진 분류 (keras44_imageDataGenerator2)

- `target_size=(150,150)`, `batch_size=160` 으로 160장을 한 번에 받아 x_train으로 사용
- **Dense만 쓰는 DNN** 이라 `reshape(-1, 150*150)` 으로 2차원(22500 컬럼)으로 폈다
- 이진 분류 3종 세트
  - 출력층 `Dense(1, activation='sigmoid')` → 0~1 확률
  - `loss='binary_crossentropy'`
  - 예측은 `np.round(y_predict)` (원핫이 아니므로 argmax가 아니다)
- 결과 : **acc 1.0** (42.08초, 목표 달성)

### 5. 고양이 / 개 분류와 batch_size (keras44_ImageDataGenerator3_CatDog)

- 8005장(train) / 2023장(test), `target_size=(100,100)`, `color_mode='rgb'`
- 모델은 Conv2D 4층 + MaxPooling 2개 + `Dense(1, sigmoid)`
- **OOM 에러를 겪었다**
```
ResourceExhaustedError: OOM when allocating tensor with shape[7204,64,100,100]
Node: 'sequential/conv2d/Conv2D'
```
  - `model.fit(batch_size=10000)` 으로 줘서 `validation_split=0.1` 을 뺀 **7204장을 한 번에** Conv2D에 넣으려 했다
  - 첫 Conv2D 출력만 `7204 × 64 × 100 × 100 × 4바이트 ≈ 18.4GB` (GPU는 약 5.4GB)
  - **batch_size는 '한 번에 GPU에 올릴 장수'** 라서 데이터 전체 장수와는 다른 값이다. 크게 준다고 빨라지지 않는다
  - `batch_size=32` 로 바꾸니 `8005 × 0.9 ÷ 32 = 226` 스텝으로 정상 훈련됐다
- **최적 가중치 저장 (ModelCheckpoint)**
  - `es` 는 `val_acc` 를 보는데 `mcp` 가 `val_loss` 를 보면 **서로 다른 epoch** 를 고른다 → 기준을 맞춘다
  - 파일명도 기준과 같게 : `'{epoch:04d}_{val_acc:.4f}.keras'`
  - 저장되면 `Epoch 1: val_acc improved from -inf to 0.53308, saving model to ...` 가 찍힌다
  - **`patience`(130)가 `epochs`(100)보다 크면 EarlyStopping이 끝까지 발동하지 않는다** → `restore_best_weights` 도 동작하지 않아 evaluate는 마지막 epoch 가중치로 한다. 최고 기록은 mcp 파일에만 남는다
- 결과 : acc 0.5002 (807.22초) → 목표 0.77 미달

### 6. npy로 저장해서 데이터 준비 시간 줄이기 (keras45_01 ~ 04)

- ImageDataGenerator로 폴더를 읽는 데 시간이 오래 걸린다 → **한 번 읽은 결과를 npy로 저장**해 두고 다음부터 불러 쓴다
- **저장 (01 brain / 03 cat_dog)** : `np.save(np_path + '파일명.npy', arr=x_train)`
- **불러오기 (02 brain / 04 cat_dog)** : `x_train = np.load(np_path + '파일명.npy')`
- 시간을 비교하려고 load 파일에서 **두 방법을 모두 실행**해 각각 측정했다
  - `flow_from_directory()` 자체는 파일 목록만 훑는다. **실제로 이미지를 읽는 시점은 `xy_train[0]` 을 꺼낼 때**라서 그 뒤에서 시간을 재야 한다
  - 시간만 재고 `del` 로 지운다. cat_dog는 `x_train` 하나가 약 960MB라 npy까지 같이 들고 있으면 메모리를 두 배로 쓴다
- 비교가 목적이라 **훈련 코드는 `''' '''` 로 묶어 두었다**
- 결과 : brain 0.24초 → **0.05초**, cat_dog 13.01초 → **0.44초**
- 같은 cat_dog인데도 ImageDataGenerator 시간이 실행할 때마다 13 ~ 56초로 달랐다. 윈도우가 한 번 읽은 파일을 캐싱하기 때문이다. npy는 캐시와 상관없이 0.3 ~ 0.5초로 일정하다
- npy 파일은 용량이 크다 (cat_dog `x_train` 약 960MB) → `.gitignore` 에 넣고 GitHub에는 올리지 않는다

### 7. 실행 환경 (ImageDataGenerator는 Keras 3에 없다)

- `py311` 환경(TF 2.21 / Keras 3.15)에서 실행하면 import 자체가 실패한다
```
ImportError: cannot import name 'ImageDataGenerator' from 'keras.preprocessing.image'
```
- **Keras 3에서 `ImageDataGenerator` 가 삭제됐다.** keras44 / keras45는 TF 2.9 GPU 환경에서 실행했다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras43_hamsu01_mnist.py` | mnist DNN을 함수형으로 전환 - `Input(784,)`, Total params 599,178, acc 0.9784 |
| `keras43_hamsu02_fashion.py` | fashion_mnist 함수형 DNN, acc 0.8788 |
| `keras43_hamsu03_cifar10.py` | cifar10 함수형 DNN - `Input(32*32*3,)`, acc 0.519 |
| `keras43_hamsu04_cifar100.py` | cifar100 함수형 DNN - BatchNormalization 포함, Total params 4,201,316, acc 0.2972 |
| `keras44_ImageDataGenerator1.py` | ImageDataGenerator 옵션 / `flow_from_directory` / DirectoryIterator 구조 확인 |
| `keras44_imageDataGenerator2.py` | brain 이미지 이진 분류 (Dense DNN) - sigmoid + binary_crossentropy, acc 1.0 |
| `keras44_ImageDataGenerator3_CatDog.py` | 고양이/개 8005장 Conv2D 분류 - batch_size OOM, ModelCheckpoint 저장, acc 0.5002 |
| `keras45_01_brain_save_npy.py` | brain 이미지를 읽어 npy로 저장 (`np.save`) |
| `keras45_02_brain_load_npy.py` | npy 불러오기(`np.load`) + 두 방법 시간 비교 - 0.24초 → 0.05초 |
| `keras45_03_catdog_save_npy.py` | cat_dog 8005장을 읽어 npy로 저장 |
| `keras45_04_catdog_load_npy.py` | npy 불러오기 + 시간 비교 - 13.01초 → 0.44초 |
| `keras41_dnn4_cifar100.py` | 마지막 은닉층 16 → 128, BatchNormalization 추가, acc 0.299 (수정) |
| `keras42_cnn10_digits.py` | Conv2D 실행 기록 추가 - acc 0.9708 (수정) |
| `.gitignore` | 데이터 / 모델 파일(`_data/image`, `_data/kaggle_santander`, npy, `_save`)을 제외 (수정) |
| `docs/Day15.md` | Day15 학습 기록 신규 작성 |
| `docs/Day14.md` | 하단 nav에 Day15 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조 / 진행도(15일, 18.75%) 갱신 (수정) |

---

## 📊 실행 결과

### 함수형 전환 (GPU)

| 데이터셋 | Sequential DNN (Day14) | 함수형 | 멈춘 epoch / 시간 |
|---|:---:|:---:|---|
| mnist | 0.9785 | 0.9784 | 78 / 323.16초 |
| fashion_mnist | 0.8661 | **0.8788** | - / 102.52초 |
| cifar10 | 0.5784 | 0.519 | 83 / 99.77초 |
| cifar100 | 0.299 | 0.2972 | 107 / 185.42초 |

### cifar100 DNN 구조 수정

| 모델 | 마지막 은닉층 | BatchNormalization | acc |
|---|:---:|:---:|:---:|
| 수정 전 (dnn1 구성 복사) | 16 | 없음 | 기록 없음 |
| 수정 후 (dnn3 구성) | 128 | 4개 | **0.299** (201.95초) |

### 이미지 폴더 분류

| 파일 | 데이터 | 입력 | 모델 | 목표 acc | 결과 acc |
|---|---|---|---|:---:|:---:|
| keras44_2 | brain 160 / 120장 | (150,150,1) → 22500 | Dense DNN | 1.0 | **1.0** ✅ |
| keras44_3 | cat_dog 8005 / 2023장 | (100,100,3) | Conv2D 4층 | 0.77 | 0.5002 ❌ |

### 데이터 준비 시간 (ImageDataGenerator vs npy)

| 데이터 | 장수 | ImageDataGenerator | npy 로드 | 차이 |
|---|:---:|---:|---:|:---:|
| brain | 280장 | 0.24초 | **0.05초** | 약 5배 |
| cat_dog | 10,028장 | 13.01초 | **0.44초** | 약 30배 |

**읽는 법:** cat_dog의 ImageDataGenerator 시간은 실행할 때마다 13 ~ 56초로 달라졌다(윈도우 파일 캐시). npy는 항상 0.3 ~ 0.5초다. 장수가 많을수록 npy로 저장해 두는 효과가 크다.

---

## 💻 핵심 개념

### Sequential → 함수형

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input

input1 = Input(shape=(784,))                      # Sequential 의 input_shape=(784,) 자리
dense1 = Dense(512, activation='relu')(input1)    # (input1) -> 앞 층에 연결
dense2 = Dense(256, activation='relu')(dense1)
drop1 = Dropout(0.4)(dense2)
dense3 = Dense(64, activation='relu')(drop1)      # Dropout 다음 층은 drop1 에 연결
output1 = Dense(10, activation='softmax')(dense3)

model = Model(inputs=input1, outputs=output1)
model.summary()     # Sequential 과 Total params 가 같아야 제대로 변환한 것
```

### ImageDataGenerator + flow_from_directory

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,           # 필수 (0~255 -> 0~1)
    # horizontal_flip=True,   # 원 데이터 왜곡을 피하려고 나머지 옵션은 주석 처리
    # rotation_range=5,
    # zoom_range=1.2,
)

xy_train = train_datagen.flow_from_directory(
    './_data/image/brain/train/',
    target_size=(150,150),    # 사진 크기를 이 크기로 맞춘다
    batch_size=160,           # 전체 장수로 주면 xy_train[0] 하나에 전 데이터가 담긴다
    class_mode='binary',      # 이진 분류 -> y 는 0 / 1 한 칸
    color_mode='grayscale',   # 흑백 -> (batch, x, y, 1)
    shuffle=True,
)
# Found 160 images belonging to 2 classes

x_train = xy_train[0][0]   # x (160, 150, 150, 1)
y_train = xy_train[0][1]   # y (160,)
```

### 이진 분류 3종 세트

```python
model.add(Dense(1, activation='sigmoid'))   # 0 ~ 1 확률

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)   # 0.5 기준 반올림 (원핫이 아니므로 argmax 가 아니다)
acc_score = accuracy_score(y_test, y_predict)
```

### batch_size와 OOM

```python
# x_train 8005장, validation_split=0.1 -> 훈련 7204장
model.fit(x_train, y_train, epochs=100, batch_size=10000)   # 7204장을 한 번에 -> OOM
model.fit(x_train, y_train, epochs=100, batch_size=32)      # 226 스텝으로 나눠서 정상

# OOM when allocating tensor with shape[7204,64,100,100]
# 7204 x 64 x 100 x 100 x 4바이트 = 약 18.4GB (GPU 5.4GB)
```

### 최적 가중치 저장 (es와 mcp 기준 맞추기)

```python
es = EarlyStopping(monitor='val_acc', mode='max', patience=130,
                   restore_best_weights=True, verbose=1)
# patience(130) > epochs(100) 이면 ES 가 발동하지 않는다
# -> restore_best_weights 도 동작하지 않아 evaluate 는 마지막 epoch 가중치로 한다

filename = '{epoch:04d}_{val_acc:.4f}.keras'    # mcp 가 보는 기준을 파일명에도 쓴다
filepath = ''.join([path, 'k_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_acc',      # es 와 같은 기준 (한쪽만 val_loss 면 서로 다른 epoch 를 고른다)
    mode='max',
    save_best_only=True,
    filepath=filepath,
    verbose=1,
)
# Epoch 1: val_acc improved from -inf to 0.53308, saving model to ...
```

### npy로 저장하고 불러오기

```python
##### 저장 (keras45_01 / 03) #####
np_path = './_data/brain_npy/'
np.save(np_path + 'keras45_01_x_train.npy', arr=x_train)
np.save(np_path + 'keras45_01_y_train.npy', arr=y_train)

##### 불러오기 (keras45_02 / 04) #####
x_train = np.load(np_path + 'keras45_01_x_train.npy')
y_train = np.load(np_path + 'keras45_01_y_train.npy')
```

### 두 방법의 시간 비교

```python
########## 방법 1 : ImageDataGenerator ##########
start_time1 = time.time()
xy_train = train_datagen.flow_from_directory(...)   # 여기서는 파일 목록만 훑는다
x_train_idg = xy_train[0][0]                        # [0] 을 꺼낼 때 실제로 이미지를 읽는다
end_time1 = time.time()

del xy_train, xy_test, x_train_idg, ...             # 시간만 재고 메모리에서 지운다

########## 방법 2 : npy ##########
start_time2 = time.time()
x_train = np.load(np_path + 'keras45_03_x_train.npy')
end_time2 = time.time()

print('ImageDataGenerator 시간 :', round(end_time1 - start_time1, 2), '초')   # 13.01 초
print('npy 로드 시간 :', round(end_time2 - start_time2, 2), '초')             # 0.44 초
```

---

## 성과

- Day14의 DNN 4종(mnist / fashion / cifar10 / cifar100)을 함수형으로 전환하고 Total params로 검증
- cifar100 DNN의 마지막 은닉층 병목(16 → 128)을 찾아 고치고 BatchNormalization 적용
- ImageDataGenerator로 내 폴더의 이미지를 직접 읽어 x, y를 만드는 방법을 익힘
- brain 이미지 이진 분류에서 **acc 1.0 달성**
- 고양이/개 8005장 훈련 중 OOM을 겪고 batch_size의 의미를 정확히 이해
- ModelCheckpoint의 monitor를 EarlyStopping과 맞춰 최적 가중치를 파일로 저장
- ImageDataGenerator 결과를 npy로 저장해 데이터 준비 시간을 **cat_dog 기준 13.01초 → 0.44초**로 단축
- keras42_cnn10(digits) Conv2D 실행 - acc 0.9708

---

## 💡 주요 학습 포인트

1. **함수형은 모델 종류가 아니라 문법이다**: Dense든 Conv2D든 `#2. 모델 구성`만 바뀐다
2. **함수형 변환 검증은 Total params**: 층을 그대로 옮겼으면 숫자가 같아야 한다
3. **층마다 변수 이름을 새로 붙인다**: 같은 이름을 다시 쓰면 그 층이 연결에서 조용히 빠진다
4. **출력 칸 수보다 앞 층이 좁으면 병목**: cifar100(100종)인데 마지막 은닉층이 16이면 acc가 크게 떨어진다
5. **폴더 구조가 곧 정답**: `flow_from_directory` 는 하위 폴더 이름을 알파벳순으로 0, 1로 매긴다
6. **`rescale=1./255` 는 필수**, 뒤집기 / 회전 등은 원 데이터를 왜곡하므로 필요할 때만 쓴다
7. **DirectoryIterator는 `[0][0]` = x, `[0][1]` = y**, `[0]`을 꺼내는 순간 실제로 이미지를 읽는다
8. **이진 분류는 sigmoid + binary_crossentropy + np.round** 세트로 움직인다
9. **batch_size는 한 번에 GPU에 올릴 장수**다. 전체 장수만큼 주면 OOM이 난다
10. **es와 mcp의 monitor는 맞춘다**: 기준이 다르면 저장된 모델과 되돌린 가중치가 서로 다른 epoch가 된다
11. **`patience`가 `epochs`보다 크면 EarlyStopping은 발동하지 않는다** → 최고 가중치는 mcp 파일에만 남는다
12. **오래 걸리는 데이터 준비는 npy로 저장**: 장수가 많을수록 효과가 크다 (cat_dog 약 30배)
13. **`ImageDataGenerator`는 Keras 3에서 삭제됐다**: TF 2.9 환경에서 실행해야 한다
14. **대용량 데이터 / 모델 파일은 GitHub에 올리지 않는다**: npy, 이미지, `.keras` 는 `.gitignore` 로 제외

---

[⬅️ Day14](Day14.md) · [🏠 전체 목차](../README.md)
