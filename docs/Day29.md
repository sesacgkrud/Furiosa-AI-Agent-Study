# Day29 - VGG16 전이학습 이미지 데이터셋 적용, 층별 trainable · 부분 동결, keras.applications 전이학습 모델 비교

**학습 기간:** 2026-10-14

> Day28에서 cifar10 · cifar100에 붙여 본 **VGG16 전이학습을 Day15 ~ Day18의 이미지 데이터셋**(man_horse · rps · man_woman · cat_dog)에 적용했다. 베이스 파일의 데이터 · 훈련 · 평가 코드는 그대로 두고, 직접 쌓던 Conv2D를 **VGG16(imagenet, `include_top=False`, 동결) + Flatten / GlobalAveragePooling2D + Dense**로 바꿨다. 이어서 `model.layers`를 돌면서 **층마다 `trainable`을 pandas 표로 확인**하고, `model.layers[인덱스].trainable = False`로 **원하는 층 하나만 동결(부분 동결)** 해 summary · 가중치 개수가 어떻게 바뀌는지 봤다. 마지막으로 `keras.applications`의 **전이학습 모델 23종**(VGG19 · ResNet · DenseNet · Inception · MobileNet · NASNet · EfficientNet · Xception)의 가중치 개수를 출력하고, 그중 5 ~ 7개를 **`model_list`에 담아 for 문으로 하나씩 같은 데이터에 붙여 비교**하는 코드를 cifar10 · cifar100 · 이미지 데이터셋 4개에 맞게 만들었다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 파일 번호 정리 | `keras73_vgg16_cifar10 / cifar100` → `keras73_vgg16_01_cifar10 / 02_cifar100`, 데이터셋 번호를 `keras1/dataset.txt`에 맞춤 |
| VGG16 이미지 데이터셋 | 베이스의 Conv2D 대신 `model.add(vgg16)` + 분류 부분만 붙인다, 입력 100x100 → VGG16 출력 `(3, 3, 512)` |
| 결과 | rps **1.0** / man_horse 0.9854 / man_woman 0.8728 / cat_dog 0.8646 |
| 층별 trainable 표 | `[(type(layer).__name__, layer.name, layer.trainable) for layer in model.layers]` → `pd.DataFrame` |
| 부분 동결 | `model.layers[2].trainable = False` → Dense(100) 하나만 Non-trainable |
| VGG16 안의 층 동결 | `VGG16(...).layers[11]` = `block4_conv1` → Non-trainable 1,180,160 |
| 전이학습 모델 23종 | `keras.applications`에서 import → `model.weights` · `model.trainable_weights` 개수 비교 |
| model_list | 모델을 리스트에 담고 `for model_i in model_list:` → 같은 Dense · 같은 설정으로 모델만 바꿔 비교 |
| GlobalAveragePooling2D | 모델마다 마지막 채널 수(512 / 2048 / 1920 ...)가 달라도 같은 코드로 Dense를 붙일 수 있다 |
| cifar10 model_list | densenet201 **0.6588** > vgg19 0.4765 > mobilenetv2 0.3927 > resnet101 0.3361 > resnet50 0.2009 |

---

## 📖 핵심 학습 내용

### 1. Day28 VGG16 파일 번호 정리 (keras73_vgg16_01 · 02, keras1/dataset.txt)

- `keras73_vgg16_cifar10.py` → `keras73_vgg16_01_cifar10.py`, `keras73_vgg16_cifar100.py` → `keras73_vgg16_02_cifar100.py` (내용은 그대로)
- 같은 VGG16 실습을 데이터셋마다 `keras73_vgg16_01 ~ 06`, `keras76_all_T_01 ~ 06` 번호로 맞췄다
- `keras1/dataset.txt` : 데이터셋 번호 목록에 `15 man_horse`, `16 rps`, `17 man_woman`, `18 cat_dog`, `19 jena`로 정리

### 2. VGG16 전이학습 이미지 데이터셋 적용 (keras73_vgg16_03 ~ 06)

Day15 ~ Day18에서 Conv2D를 직접 쌓아 훈련했던 파일을 베이스로, **모델 부분만** VGG16으로 바꿨다.

```python
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(100, 100, 3)   # 100 → 50 → 25 → 12 → 6 → 3 → 출력 (3, 3, 512)
)

vgg16.trainable=False      # 가중치 동결
# vgg16.trainable=True     # 가중치 동결 X

model = Sequential()
model.add(vgg16)
model.add(Flatten())            # (3, 3, 512) -> (4608,)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))   # rps 3 / cat_dog Dense(1, sigmoid)
```

| 파일 | 베이스 | 데이터 | 분류 부분 | Trainable params |
|---|---|---|---|:---:|
| `keras73_vgg16_03_man_horse` | keras47_01 | horse npy (821 / 206장) | Flatten → Dense(64) → Dense(2, softmax) | 295,106 |
| `keras73_vgg16_04_rps` | keras47_02 | rps npy (1639 / 410장) | Flatten → Dense(64) → Dense(3, softmax) | 295,171 |
| `keras73_vgg16_05_man_woman` | keras52_ReduceLR15 | man_woman npy + 여자 사진 5240장 증폭 | GlobalAveragePooling2D → Dense(128) → Dense(1, sigmoid) | 65,793 |
| `keras73_vgg16_06_cat_dog` | keras45_04 | cat_dog npy (8005 / 2023장) | Flatten → Dense(64) → Dense(1, sigmoid) | 295,041 |

- **`trainable`을 적지 않으면 기본값 True (동결 X)** → 동결 O / 동결 X를 직접 골라 적는다
- 특징 추출은 VGG16(Conv2D 13개)이 하므로 **Conv2D를 더 쌓지 않는다** : VGG16 출력이 `(3, 3)`이라 MaxPooling2D를 두 번 넣으면 `(1, 1)` → `(0, 0)`이 되어 에러
- 데이터 · compile · EarlyStopping(`val_acc`, patience 20) · ModelCheckpoint · 평가 코드는 베이스 그대로
- npy는 저장할 때 `rescale=1./255`로 이미 0 ~ 1이라 따로 스케일링하지 않는다
- man_woman
  - GlobalAveragePooling2D : `(3, 3, 512)` → 채널마다 평균 `(512,)` → Flatten `(4608,)`보다 Dense 파라미터가 9배 적다
  - `learning_rate=0.0018` + ReduceLROnPlateau : **rlr patience 5 < es patience 10** 으로 두어 es가 멈추기 전에 rlr이 학습률을 줄일 수 있게 했다
  - 베이스(Conv2D 직접 쌓기) 0.9067 / 538.36초 → VGG16 동결 0.8728 / 411.34초 : 시간은 줄었지만 acc는 낮았다

### 3. 층별 가중치 개수와 trainable 표 (keras74_1 · keras74_2)

```python
vgg16.trainable = False
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

print(len(model.weights))           # 30 = 13 (vgg16 Conv2D) * 2 (w, b) + 2 (dense) * 2 (w, b)
print(len(model.trainable_weights)) # 4  = Dense 2개 * 2 (w, b)

import pandas as pd
pd.set_option('max_colwidth', None)

layers = [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in model.layers
]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
```

- 가중치는 **가중치가 있는 층 수 x 2** (kernel · bias) : VGG16(`include_top=False`)은 Conv2D 13개 → 26개, MaxPooling2D · InputLayer · Flatten은 가중치가 없다
- `vgg16.trainable = False` → vgg16의 26개가 `trainable_weights`에서 빠지고 Dense 2개의 4개만 남는다
- `model.layers` : 모델 안의 층 리스트, **vgg16은 통째로 층 1개(`Functional`)** 로 들어간다
- `type(layer).__name__` : 층 종류 이름, `layer.name` : 층 이름, `layer.trainable` : 훈련 여부
- `pd.set_option('max_colwidth', None)` : 표의 칸 내용을 `...`으로 자르지 않고 다 보여준다

```
   Layer Type Layer Name  Layer Trainable
0  Functional      vgg16            False
1     Flatten    flatten             True
2       Dense      dense             True
3       Dense    dense_1             True
```

### 4. 원하는 층 하나만 동결 - 부분 동결 (keras74_3)

```python
model.add(vgg16)            # index 0
model.add(Flatten())        # index 1
model.add(Dense(100))       # index 2  <- 이 층만 동결
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(10, activation='softmax'))

model.layers[2].trainable = False   # Flatten() 포함한 갯수로 index 2에 해당하는 layers 동결
```

- `model.layers[인덱스].trainable = False` : 모델 전체(Day28 `model.trainable`)나 vgg16 통째가 아니라 **층 하나만** 동결
- 인덱스는 `model.layers` 순서 → 가중치가 없는 Flatten도 하나로 센다
- summary : Trainable **22,750** (Dense 90 · 80 · 70 · 10) / Non-trainable **14,765,988** (vgg16 14,714,688 + Dense(100) 51,300)
- `len(model.weights)` **36** = 13 x 2 + 5 x 2, `len(model.trainable_weights)` **8** = 동결하지 않은 Dense 4개 x 2

### 5. VGG16 안의 층 하나만 동결 (keras74_4)

```python
model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

model.layers[11].trainable = False  # VGG16 의 index 11 = block4_conv1 하나만 동결
```

- Sequential에 넣으면 vgg16 한 줄로만 보이지만, **VGG16 자체의 `model.layers`는 19개 층**(input_layer + Conv2D 13 + MaxPooling2D 5)이 다 보인다
- index 0이 `input_layer`라서 index 11 = `block4_conv1`
- Non-trainable **1,180,160** = block4_conv1 `(3 x 3 x 256 + 1) x 512`, Trainable 13,534,528
- `len(model.weights)` **26**, `len(model.trainable_weights)` **24** (block4_conv1의 w, b 2개가 빠진다)

### 6. keras.applications 전이학습 모델 23종 (keras75_All_TransferLearning)

```python
import warnings
warnings.filterwarnings('ignore')   # console 창에 나오는 warnings 안 보이게 함

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception

model_list = [VGG16(), VGG19(), ResNet50(), ...]

for model in model_list:
    print('모델명 : ', model.name)
    print('전체 가중치 갯수 : ', len(model.weights))
    print('전체 훈련 가능 가중치 갯수 : ', len(model.trainable_weights))
```

- `keras.applications` : VGG16 말고도 imagenet으로 미리 훈련된 모델을 같은 방식(`weights`, `include_top`, `input_shape`)으로 불러온다
- 모델을 **리스트에 담아 for 문**으로 돌리면 모델마다 같은 코드를 반복할 수 있다
- `warnings.filterwarnings('ignore')` : 실행할 때 나오는 경고 메시지를 숨긴다

| 모델명 | 전체 가중치 | 훈련 가능 가중치 |
|---|:---:|:---:|
| xception | 236 | 156 |
| vgg16 | 32 | 32 |
| vgg19 | 38 | 38 |
| resnet50 | 320 | 214 |
| resnet50v2 | 272 | 174 |
| resnet101 | 626 | 418 |
| resnet101v2 | 544 | 344 |
| resnet152 | 932 | 622 |
| resnet152v2 | 816 | 514 |
| densenet201 | 1006 | 604 |
| densenet169 | 846 | 508 |
| densenet121 | 606 | 364 |
| inception_v3 | 378 | 190 |
| inception_resnet_v2 | 898 | 490 |
| mobilenet_1.00_224 | 137 | 83 |
| mobilenetv2_1.00_224 | 262 | 158 |
| MobileNetV3Small | 210 | 142 |
| MobileNetV3Large | 266 | 174 |
| nasnet_mobile | 1126 | 742 |
| nasnet_large | 1546 | 1018 |
| efficientnetb0 | 314 | 213 |
| efficientnetb1 | 442 | 301 |
| efficientnetb2 | 442 | 301 |

- VGG16 · VGG19는 **전체 = 훈련 가능** (Conv2D · Dense의 kernel · bias만 있다, VGG16 16층 x 2 = 32)
- 나머지는 전체 > 훈련 가능 : **BatchNormalization 층의 이동 평균 · 이동 분산**도 가중치로 세지만 역전파로 고치는 값이 아니라서 `trainable_weights`에는 들어가지 않는다
- `Xception()` summary : Total params 22,910,480 / Trainable params 22,855,952

### 7. model_list로 여러 전이학습 모델 비교 (keras76_all_T_01 ~ 06)

```python
model_list = [VGG19(include_top=False, input_shape=(32, 32, 3)),
              ResNet50(include_top=False, input_shape=(32, 32, 3)),
              ResNet101(include_top=False, input_shape=(32, 32, 3)),
              DenseNet201(include_top=False, input_shape=(32, 32, 3)),
              MobileNetV2(include_top=False, input_shape=(32, 32, 3)),
              ]

for model_i in model_list:
    model_i.trainable = False       # 동결 O
    # model_i.trainable = True      # 동결 X

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=9999999, batch_size=32, verbose=0,
              validation_split=0.2, callbacks=[es])

    loss = model.evaluate(x_test, y_test, verbose=0)
    print('모델명 :', model_i.name, 'loss :', loss[0], 'acc :', loss[1], '시간 :', ...)
```

- 모든 모델이 **같은 데이터 · 같은 Dense · 같은 compile / fit 설정** → 특징 추출 부분(모델)만 바꿨을 때를 비교
- **GlobalAveragePooling2D** : 모델마다 마지막 출력 채널이 다르다 (VGG19 512, ResNet50 2048, DenseNet201 1920 ...) → 채널마다 평균내서 `(채널,)`로 만들면 어떤 모델이 와도 같은 코드로 Dense를 붙일 수 있다
- `epochs=9999999` : 사실상 무한 → EarlyStopping이 멈출 때까지 훈련
- `verbose=0` : 모델이 여러 개라 epoch 출력은 끄고 모델마다 결과 한 줄만 출력
- 모델마다 똑같은 데이터를 쓰므로 데이터는 for 문 밖에서 한 번만 불러온다
- 데이터셋별 차이

| 파일 | 데이터 | input_shape | 출력층 · loss | 모델 |
|---|---|:---:|---|---|
| `keras76_all_T_01_cifar10` | cifar10, `/255.` | (32, 32, 3) | Dense(10, softmax) · sparse_categorical_crossentropy | VGG19 · ResNet50 · ResNet101 · DenseNet201 · MobileNetV2 |
| `keras76_all_T_02_cifar100` | cifar100, `/255.` | (32, 32, 3) | Dense(100, softmax) · sparse_categorical_crossentropy | 위와 같음 |
| `keras76_all_T_03_man_horse` | horse npy | (100, 100, 3) | Dense(2, softmax) · categorical_crossentropy (y 원핫) | 위와 같음 |
| `keras76_all_T_04_rps` | rps npy | (100, 100, 3) | Dense(3, softmax) · categorical_crossentropy (y 원핫) | 위와 같음 |
| `keras76_all_T_05_man_woman` | man_woman npy + val 분리 · 여자 사진 증폭 | (100, 100, 3) | Dense(1, sigmoid) · binary_crossentropy | + InceptionV3 · InceptionResNetV2 |
| `keras76_all_T_06_cat_dog` | cat_dog npy | (100, 100, 3) | Dense(1, sigmoid) · binary_crossentropy | + InceptionV3 · InceptionResNetV2 |

- InceptionV3 · InceptionResNetV2는 **입력이 최소 75x75**라서 32x32 cifar에는 넣을 수 없고, 100x100인 man_woman · cat_dog에 넣었다
- man_woman은 keras73_vgg16_05와 같게 val을 먼저 떼고 여자 사진을 증폭한 뒤 `validation_data=(x_val, y_val)`로 검증
- MobileNetV2는 imagenet 가중치가 (224, 224) 기준이라 32x32를 넣어도 이름이 `mobilenetv2_1.00_224`로 나온다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras2/keras73_vgg16_01_cifar10.py` | Day28 `keras73_vgg16_cifar10.py` 파일 번호 변경 (수정) |
| `keras2/keras73_vgg16_02_cifar100.py` | Day28 `keras73_vgg16_cifar100.py` 파일 번호 변경 (수정) |
| `keras2/keras73_vgg16_03_man_horse.py` | VGG16 동결 + Flatten · Dense, man_horse acc 0.9854 |
| `keras2/keras73_vgg16_04_rps.py` | VGG16 동결 + Flatten · Dense, rps acc 1.0 |
| `keras2/keras73_vgg16_05_man_woman.py` | VGG16 동결 + GlobalAveragePooling2D · Dense, 증폭 · ReduceLROnPlateau, acc 0.8728 |
| `keras2/keras73_vgg16_06_cat_dog.py` | VGG16 동결 + Flatten · Dense, cat_dog acc 0.8646 |
| `keras2/keras74_1_layers.py` | VGG16 모델의 `model.weights` 30 / `trainable_weights` 4 |
| `keras2/keras74_2_layers.py` | `model.layers`의 층 종류 · 이름 · trainable을 pandas 표로 출력 |
| `keras2/keras74_3_layers_partial_freeze.py` | `model.layers[2].trainable = False`로 Dense(100)만 동결, weights 36 / trainable 8 |
| `keras2/keras74_4_layers_partial_freeze.py` | VGG16 자체의 19개 층 표, `layers[11]` (block4_conv1)만 동결, weights 26 / trainable 24 |
| `keras2/keras75_All_TransferLearning.py` | `keras.applications` 전이학습 모델 23종 import, 모델별 가중치 개수 출력 |
| `keras2/keras76_all_T_01_cifar10.py` | model_list 5개 모델 cifar10 비교 (동결 X 기록 densenet201 0.6588 최고) |
| `keras2/keras76_all_T_02_cifar100.py` | model_list 5개 모델 cifar100 비교 코드, Dense(100) |
| `keras2/keras76_all_T_03_man_horse.py` | model_list 5개 모델 man_horse 비교 코드 |
| `keras2/keras76_all_T_04_rps.py` | model_list 5개 모델 rps 비교 코드 |
| `keras2/keras76_all_T_05_man_woman.py` | model_list 7개 모델 (+ InceptionV3 · InceptionResNetV2) man_woman 비교 코드 |
| `keras2/keras76_all_T_06_cat_dog.py` | model_list 7개 모델 (+ InceptionV3 · InceptionResNetV2) cat_dog 비교 코드 |
| `keras2/keras72_1_vgg16_include_top.py` | 주석의 Day28 파일 이름을 바뀐 번호로 수정 (수정) |
| `keras1/dataset.txt` | 데이터셋 번호 15 ~ 19 정리 (수정) |
| `docs/Day29.md` | Day29 학습 기록 신규 작성 |
| `docs/Day28.md` | 바뀐 파일 이름 반영, 하단 nav에 Day29 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day29) / 진행도(29일, 36%) 갱신 (수정) |

---

## 📊 실행 결과

### VGG16 전이학습 이미지 데이터셋 (keras73_vgg16_03 ~ 06, 동결 O)

| 데이터 | 분류 | EarlyStopping | test acc | 소요 시간 |
|---|---|:---:|:---:|:---:|
| man_horse | 2종 (원핫 · softmax) | Epoch 30 | 0.9854 | 23.43초 |
| rps | 3종 (원핫 · softmax) | Epoch 21 | **1.0** | 29.77초 |
| man_woman | 이진 (sigmoid) | Epoch 20 | 0.8728 | 411.34초 |
| cat_dog | 이진 (sigmoid) | Epoch 24 | 0.8646 | 148.4초 |

- man_woman 베이스(keras52_ReduceLR15, Conv2D 직접 쌓기) 0.9067 / 538.36초 → VGG16 동결 0.8728 / 411.34초

### 부분 동결 (keras74)

| 파일 | 동결한 것 | len(weights) | len(trainable_weights) | Trainable params | Non-trainable params |
|---|---|:---:|:---:|:---:|:---:|
| keras74_1 · 2 | vgg16 전체 | 30 | 4 | - | - |
| keras74_3 | vgg16 전체 + Dense(100) | 36 | 8 | 22,750 | 14,765,988 |
| keras74_4 | VGG16의 block4_conv1 | 26 | 24 | 13,534,528 | 1,180,160 |

### model_list cifar10 (keras76_all_T_01, 동결 X 기록)

| 모델 | test loss | test acc |
|---|:---:|:---:|
| densenet201 | 1.0415 | **0.6588** |
| vgg19 | 1.3448 | 0.4765 |
| mobilenetv2_1.00_224 | 1.8287 | 0.3927 |
| resnet101 | 2.5121 | 0.3361 |
| resnet50 | 4.3282 | 0.2009 |

- 같은 Dense · 같은 설정인데 모델에 따라 acc가 3배 넘게 차이 났다
- Day28 keras73_vgg16_01 (VGG16 + Flatten · Dense(100) x 2, 동결 X) 0.8158보다는 모두 낮았다

---

## 💻 핵심 개념

### VGG16 전이학습 모델 구조 (이미지 데이터셋)

```
(100, 100, 3) ─▶ [VGG16 동결 : Conv2D 13 + MaxPooling2D 5] ─▶ (3, 3, 512) ─┬─ Flatten (4608,)           ─▶ Dense(64) ─▶ Dense(2 / 3 softmax, 1 sigmoid)
                 └──── Non-trainable 14,714,688 ────┘                       └─ GlobalAveragePooling2D (512,) ─▶ Dense(128) ─▶ Dense(1, sigmoid)   (man_woman)
```

### 동결 범위

```
model.trainable = False              모델 전체 동결                          (Day28 keras71)
vgg16.trainable = False              층으로 넣은 vgg16 통째 동결 (Functional 1줄)  (keras74_1 · 2)
model.layers[2].trainable = False    Sequential 안의 층 하나만 동결            (keras74_3)
VGG16(...).layers[11].trainable      VGG16 안의 Conv2D 하나만 동결             (keras74_4)
```

### model_list 비교 흐름

```
데이터 (for 문 밖에서 한 번)
for model_i in model_list:
    model_i.trainable = False
    Sequential [model_i ─▶ GlobalAveragePooling2D ─▶ Dense(32) ─▶ Dense(16) ─▶ 출력층]
    compile ─▶ fit (EarlyStopping) ─▶ evaluate ─▶ 모델명 · loss · acc · 시간 출력
```

---

## 성과

- VGG16 전이학습을 이미지 데이터셋 4개에 적용해 rps **1.0**, man_horse 0.9854, man_woman 0.8728, cat_dog 0.8646을 기록했다
- `model.layers`와 pandas 표로 층마다 `trainable` 여부를 확인했다
- `model.layers[인덱스].trainable`로 Sequential 안의 층 하나, VGG16 안의 Conv2D 하나만 동결하고 summary · 가중치 개수로 확인했다
- `keras.applications`의 전이학습 모델 23종을 불러와 가중치 개수를 비교했다
- `model_list` + for 문으로 여러 전이학습 모델을 같은 조건에서 비교하는 코드를 cifar10 · cifar100 · 이미지 데이터셋 4개에 맞게 만들었고, cifar10에서는 densenet201이 0.6588로 가장 높았다

---

## 💡 주요 학습 포인트

1. **전이학습은 모델 부분만 바꾼다**: 베이스의 데이터 · 훈련 · 평가 코드는 그대로 두고 Conv2D 대신 `model.add(vgg16)`
2. **`trainable`을 적지 않으면 동결 X**: 기본값이 True라서 동결하려면 `vgg16.trainable=False`를 직접 적는다
3. **VGG16 뒤에 Conv2D를 더 쌓지 않는다**: 100x100 입력이면 출력이 (3, 3)이라 MaxPooling2D 두 번이면 크기가 0
4. **가중치 개수 = 가중치가 있는 층 수 x 2**: MaxPooling2D · InputLayer · Flatten은 가중치가 없다
5. **Sequential에 넣은 vgg16은 층 1개(Functional)**: `model.layers`에서 한 줄로 보인다
6. **`model.layers[인덱스].trainable = False`로 층 하나만 동결**: 인덱스는 Flatten까지 포함한 층 순서
7. **VGG16 자체의 `layers`는 19개**: index 0이 input_layer라서 index 11 = block4_conv1
8. **동결해도 Total params는 그대로**: Trainable → Non-trainable로 옮겨 갈 뿐
9. **`keras.applications`에는 VGG16 말고도 여러 전이학습 모델이 있다**: 같은 `weights` · `include_top` · `input_shape`로 불러온다
10. **모델을 리스트에 담아 for 문으로 비교한다**: 같은 데이터 · 같은 Dense · 같은 설정 → 모델만 바꾼 비교
11. **GlobalAveragePooling2D는 출력 채널 수가 달라도 같은 코드로 붙는다**: 512 / 2048 / 1920 → (채널,)
12. **모델마다 최소 입력 크기가 있다**: InceptionV3 · InceptionResNetV2는 75x75 이상이라 32x32 cifar에는 못 넣는다
13. **같은 조건이어도 모델에 따라 결과가 크게 다르다**: cifar10 densenet201 0.6588 vs resnet50 0.2009
14. **같은 데이터는 for 문 밖에서 한 번만 불러온다**: 모델마다 다시 불러올 필요가 없다

---

[⬅️ Day28](Day28.md) · [🏠 전체 목차](../README.md)
