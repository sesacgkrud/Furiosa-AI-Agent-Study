# Day12 - CPU / GPU 속도 비교와 CNN 입문

**학습 기간:** 2026-09-15

> keras34 함수형 모델 10개를 **CPU와 GPU로 각각 실행해 소요 시간을 비교**하고, **CNN(Conv2D)** 과 이미지 데이터(mnist)를 처음 다뤘다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| GPU 인식 확인 | `tf.config.experimental.list_physical_devices('GPU')` 가 빈 리스트면 CPU로만 돈다 |
| CPU vs GPU | 10개 중 **CPU 6 / GPU 4** - 데이터와 모델이 작으면 GPU가 오히려 느리다 |
| 공정한 시간 비교 | EarlyStopping을 빼고 **epochs를 고정**해야 시간을 비교할 수 있다 |
| 가상환경 분리 | GPU 환경은 별도 conda 환경(`tf29xgpu`)이라 pandas 같은 패키지를 따로 설치해야 한다 |
| Conv2D | `Conv2D(필터 수, (커널 크기), input_shape=(가로, 세로, 채널))` |
| mnist | 28×28 흑백 6만 장, 픽셀값 0 ~ 255, 정답 0 ~ 9 |
| 이미지 스케일링 | `x/255.` -> `0~1` / `(x-127.5)/127.5` -> `-1~1` |

---

## 📖 핵심 학습 내용

### 1. GPU 인식 확인 (keras35_gpu_test00)

- `tf.config.experimental.list_physical_devices('GPU')` 로 텐서플로가 GPU를 잡고 있는지 확인
  - 잡혔을 때: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`
  - 빈 리스트 `[]` 면 CPU로만 돌고 있다는 뜻
- `if(gpus):` 로 분기해서 `GPU 실행!` / `GPU 실행X` 를 출력하도록 작성
- **GPU 환경은 CPU 환경과 별개의 가상환경**이라, 따릉이(keras35_04) 실행 시 `ModuleNotFoundError: No module named 'pandas'` 발생
  -> GPU 환경에도 필요한 패키지를 따로 설치해야 한다

### 2. CPU vs GPU 속도 비교 (keras35_gpu_test01 ~ 10)

- keras34(함수형 + Dropout + MCP) 10개를 그대로 복사해 **코드는 건드리지 않고** CPU / GPU 로 각각 실행
- **시간을 비교하려면 훈련량이 같아야 한다**
  - `EarlyStopping`을 주석 처리하고 `epochs`를 고정 (대부분 100, santander / covtype은 10)
  - EarlyStopping이 켜져 있으면 멈추는 epoch가 매번 달라져서 시간 비교가 성립하지 않는다
  - `ModelCheckpoint`는 그대로 두고 `callbacks=[mcp]` 로만 넣음
- `time.time()` 을 `fit` 앞뒤에 찍어서 소요 시간을 재고, 파일 하단에 **CPU 기록 / GPU 기록**을 나눠서 저장
- **결과: 10개 중 CPU가 6개, GPU가 4개에서 더 빨랐다**
  - GPU가 이긴 쪽: diabetes, boston, cancer, wine -> 모두 **데이터가 작은(수백 행) 데이터**
  - CPU가 이긴 쪽: california, 따릉이, bike, santander, covtype, digits
  - covtype(58만 행)은 CPU 136초 / GPU 235초로 **GPU가 1.7배 느렸다**
- **왜 GPU가 항상 빠르지 않은가**
  - GPU는 계산을 한꺼번에 많이 하는 장치라, 데이터를 GPU 메모리로 보내고 받아오는 **전송 시간**이 따로 든다
  - 모델이 작고 batch_size가 작으면 계산 시간보다 전송/호출 시간이 커져서 CPU가 유리하다
  - digits는 `batch_size=4` 로 작아서 GPU가 29초 -> 58초로 두 배 느렸다
- loss / acc 같은 성능 지표는 CPU와 GPU가 **거의 같다** (소수점 아래에서만 차이) -> 속도만 달라지고 결과는 같다

### 3. CNN 입문 (keras36_cnn1 ~ 3)

- **Conv2D** - 이미지를 다루는 층. `Conv2D(필터 개수, (커널 크기), input_shape=(가로, 세로, 채널))`
  - 커널(필터)이 이미지 위를 한 칸씩 훑으면서 특징을 뽑는다
  - `(2,2)` 커널이 지나가면 가로 세로가 1씩 줄어든다 : `5×5 -> 4×4 -> 3×3`
  - `input_shape` 의 마지막 값은 **채널** (흑백 1, 컬러 3) -> `input_dim` 으로는 표현할 수 없어 `input_shape` 를 쓴다
  - 파라미터 수 = `(커널 가로 × 커널 세로 × 입력 채널 + 1) × 필터 개수`
    - `Conv2D(10, (2,2), input_shape=(5,5,1))` -> `(2×2×1 + 1) × 10 = 50`
    - `Conv2D(5, (2,2))` -> `(2×2×10 + 1) × 5 = 205`
- **mnist 데이터 구조 확인** (keras36_cnn2)
  - `mnist.load_data()` 는 처음부터 train / test 로 나뉘어 들어온다 : `(60000, 28, 28)` / `(10000, 28, 28)`
  - 픽셀값은 **0 ~ 255** 정수, 정답 y는 0 ~ 9
  - `np.unique(y_train, return_counts=True)` 로 숫자별 개수 확인 (5421 ~ 6742개로 비교적 고르다)
  - `plt.imshow(x_train[50000], 'gray')` 로 실제 손글씨 이미지를 눈으로 확인
- **이미지 스케일링** (keras36_cnn3) - 픽셀 0~255를 그대로 넣지 않고 줄인다
  - 방법 1 : `x/255.` -> **0 ~ 1** (MinMaxScaler / MaxAbsScaler와 같은 결과)
  - 방법 2 : `(x - 127.5)/127.5` -> **-1 ~ 1**
  - 이미지는 **모든 컬럼(픽셀)의 범위가 0~255로 같아서** sklearn scaler 없이 나눗셈만으로 스케일링이 된다
  - `255.` 처럼 점을 찍는 이유는 결과를 실수(float)로 만들기 위해서

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras35_gpu_test00.py` | GPU 인식 확인 - `list_physical_devices('GPU')` 결과로 `GPU 실행!` / `GPU 실행X` 분기 |
| `keras35_gpu_test01_california.py` | California 회귀 - CPU 39.3초 / GPU 59.3초 (epochs=100, batch=32) |
| `keras35_gpu_test02_diabetes.py` | Diabetes 회귀 - CPU 8.13초 / **GPU 6.36초** |
| `keras35_gpu_test03_boston.py` | Boston 회귀 - CPU 7.96초 / **GPU 5.92초** |
| `keras35_gpu_test04_dacon_ddareung.py` | 따릉이 회귀 - CPU 11.74초 / GPU 14.32초. GPU 환경에 pandas가 없어 `ModuleNotFoundError` 발생 후 설치 |
| `keras35_gpu_test05_kaggle_bike.py` | Kaggle Bike 회귀 - CPU 24.9초 / GPU 42.76초 |
| `keras35_gpu_test06_cancer.py` | Cancer 이진 분류 - CPU 8.34초 / **GPU 6.49초** |
| `keras35_gpu_test07_santander.py` | Santander 다중 분류 - CPU 65.38초 / GPU 77.38초 (20만 행이라 epochs=10) |
| `keras35_gpu_test08_wine.py` | Wine 다중 분류 - CPU 8.02초 / **GPU 4.65초** (최대 격차, 1.7배) |
| `keras35_gpu_test09_fetch_covtype.py` | Covtype 다중 분류 - CPU 136.02초 / GPU 234.58초 (58만 행이라 epochs=10) |
| `keras35_gpu_test10_digits.py` | Digits 다중 분류 - CPU 29.17초 / GPU 58.05초 (batch_size=4) |
| `keras36_cnn1.py` | `Conv2D` 첫 실습 - `(5,5,1)` 입력에 `(2,2)` 커널 2번 -> `4×4×10` -> `3×3×5`, Total params 255 |
| `keras36_cnn2_mnist_imshow.py` | mnist 구조 확인 - `(60000,28,28)`, 픽셀 0~255, 숫자별 개수 확인, `plt.imshow` 로 이미지 출력 |
| `keras36_cnn3_mnist.py` | 이미지 스케일링 2가지 - `x/255.` -> `0~1` / `(x-127.5)/127.5` -> `-1~1` |
| `_save/keras34/keras34_mcp1.keras` ~ `keras34_mcp10.keras` | keras35 실행으로 MCP 파일 갱신 (keras35는 저장 경로가 `_save/keras34/`) (수정) |
| `_data/kaggle_santander/submit/submit_0910_1724_scaler.csv` | Santander 스케일링 적용본 제출 파일 갱신 (수정) |
| `docs/Day12.md` | Day12 학습 기록 신규 작성 |
| `docs/Day01.md` ~ `docs/Day11.md` | 가독성 개선 - `🎯 한눈에 보기` 요약표 추가, 긴 나열식 본문을 소제목으로 분리 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day11 한 줄로 통합) / 진행도(12일, 15.00%) 갱신 (수정) |

---

## 📊 실행 결과

### CPU vs GPU 소요 시간 (10개 데이터셋)

| 데이터셋 | 데이터 크기 | epochs / batch | CPU | GPU | 더 빠른 쪽 |
|---|---|---|---:|---:|:---:|
| California | 20,640 × 8 | 100 / 32 | **39.3초** | 59.3초 | CPU |
| Diabetes | 442 × 10 | 100 / 32 | 8.13초 | **6.36초** | GPU |
| Boston | 506 × 13 | 100 / 32 | 7.96초 | **5.92초** | GPU |
| 따릉이 | 1,328 × 9 | 100 / 16 | **11.74초** | 14.32초 | CPU |
| Kaggle Bike | 10,886 × 8 | 100 / 32 | **24.9초** | 42.76초 | CPU |
| Cancer | 569 × 30 | 100 / 32 | 8.34초 | **6.49초** | GPU |
| Santander | 200,000 × 200 | 10 / 32 | **65.38초** | 77.38초 | CPU |
| Wine | 178 × 13 | 100 / 32 | 8.02초 | **4.65초** | GPU |
| Covtype | 581,012 × 54 | 10 / 32 | **136.02초** | 234.58초 | CPU |
| Digits | 1,797 × 64 | 100 / 4 | **29.17초** | 58.05초 | CPU |

**읽는 법:** GPU가 이긴 4개(diabetes, boston, cancer, wine)는 모두 **수백 행짜리 작은 데이터**다. 데이터가 커지면 오히려 CPU가 이겼는데, 이는 모델이 Dense 몇 층뿐이라 GPU의 병렬 계산 이점보다 데이터를 주고받는 시간이 더 크기 때문이다. 층이 깊고 연산량이 많은 CNN에서는 결과가 달라질 수 있다.

### 성능 지표는 CPU / GPU가 거의 같다

| 데이터셋 | CPU | GPU |
|---|---|---|
| California (r2) | 0.8006 | 0.8014 |
| Cancer (acc_score) | 0.9649 | 0.9591 |
| Covtype (accuracy_score) | 0.8892 | 0.8924 |
| Digits (accuracy_score) | 0.9638 | 0.9555 |

-> 장치가 바뀌어도 **결과는 같고 속도만 달라진다** (남은 차이는 시드 미고정 때문)

---

## 💻 핵심 개념

### GPU 인식 확인

```python
import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)   # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

if(gpus):
    print('GPU 실행!')
else:
    print('GPU 실행X')     # 빈 리스트 [] 면 CPU 로만 돌고 있다
```

### 시간을 비교하려면 훈련량을 고정한다

```python
# es = EarlyStopping(          # 시간 비교가 목적이라 통째로 주석 처리
#     monitor='val_loss',
#     patience=20,
#     restore_best_weights=True,
# )

start_time = time.time()
hist = model.fit(x_train, y_train,
                 epochs=100,          # 고정! ES 가 켜져 있으면 멈추는 시점이 매번 달라진다
                 batch_size=32,
                 validation_split=0.3,
                 callbacks=[mcp],     # mcp 만 남긴다
                 verbose=1)
end_time = time.time()
print("소요 시간 :", round(end_time - start_time, 2), "초")
```

### Conv2D - 출력 크기와 파라미터 수

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))   # (2,2) = 커널(필터) 크기
model.add(Conv2D(5, (2,2)))
model.summary()

#  Layer (type)          Output Shape          Param #
#  conv2d (Conv2D)       (None, 4, 4, 10)      50
#  conv2d_1 (Conv2D)     (None, 3, 3, 5)       205
#                                     Total params: 255

# 출력 크기 : (2,2) 커널이 훑으면 가로 세로가 1씩 줄어든다  5x5 -> 4x4 -> 3x3
# 파라미터  : (커널 가로 x 커널 세로 x 입력 채널 + 1) x 필터 개수
#             (2 x 2 x 1  + 1) x 10 =  50
#             (2 x 2 x 10 + 1) x  5 = 205
# input_shape=(5,5,1) 의 마지막 1 = 채널 (흑백 1 / 컬러 3)
```

### mnist 데이터 확인

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()   # 처음부터 train/test 분리
print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0..9]), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]))

import matplotlib.pyplot as plt
plt.imshow(x_train[50000], 'gray')    # 손글씨 한 장을 직접 그려서 확인
plt.show()
```

### 이미지 스케일링 - scaler 없이 나눗셈으로

```python
print(np.max(x_train), np.min(x_train))   # 255 0

########## 스케일링 1 ##########
# x_train = x_train/255.      # . 을 찍어서 실수로 만든다 -> 0 ~ 1
# x_test  = x_test/255.
# print(np.max(x_train), np.min(x_train))   # 1.0 0.0

########## 스케일링 2 ##########
x_train = (x_train - 127.5)/127.5          # -1 ~ 1
x_test  = (x_test - 127.5)/127.5
print(np.max(x_train), np.min(x_train))    # 1.0 -1.0

# 이미지는 모든 픽셀의 범위가 0~255 로 같아서 sklearn scaler 없이 나눗셈만으로 된다
```

---

## 성과

- keras34 함수형 모델 10개를 **CPU / GPU 두 번씩 실행**해 소요 시간과 성능 지표를 파일 하단에 전부 기록
- GPU가 항상 빠른 것이 아니라는 것을 **10개 중 CPU 6승**이라는 수치로 확인하고, 데이터 크기 / batch_size와 연결해서 설명
- 시간 비교를 위해 **EarlyStopping을 끄고 epochs를 고정**해야 한다는 실험 설계 원칙을 적용
- GPU 가상환경이 CPU 환경과 분리되어 있어 패키지를 따로 설치해야 한다는 것을 직접 겪음
- `Conv2D` 의 출력 크기 변화(5×5 -> 4×4 -> 3×3)와 파라미터 계산식을 `summary()` 로 검증
- mnist 6만 장의 구조를 확인하고 이미지 스케일링 2가지 방법을 비교

---

## 💡 주요 학습 포인트

1. **GPU 확인은 `list_physical_devices('GPU')`**: 빈 리스트면 CPU로 돌고 있는 것이다
2. **GPU가 항상 빠르지 않다**: 데이터를 GPU로 보내고 받는 시간이 따로 들기 때문에, 작은 데이터 / 얕은 모델은 CPU가 유리하다
3. **batch_size가 작으면 GPU가 불리하다**: digits는 `batch_size=4` 라 GPU가 2배 느렸다
4. **시간을 비교하려면 훈련량을 고정한다**: EarlyStopping을 끄고 epochs를 같게 맞춰야 비교가 성립한다
5. **장치가 바뀌어도 결과는 같다**: CPU / GPU의 loss·acc는 거의 같고 속도만 달라진다
6. **GPU 환경은 별도 가상환경**: pandas 같은 패키지를 CPU 환경과 따로 설치해야 한다
7. **Conv2D의 출력 크기**: `(2,2)` 커널이 지나가면 가로 세로가 1씩 줄어든다 (5×5 -> 4×4 -> 3×3)
8. **Conv2D의 파라미터 수**: `(커널 가로 × 커널 세로 × 입력 채널 + 1) × 필터 개수`
9. **이미지는 `input_shape`가 필수**: `(28, 28, 1)` 처럼 3차원이라 `input_dim` 으로는 표현할 수 없다
10. **이미지 스케일링은 나눗셈으로 충분**: 모든 픽셀이 0~255로 단위가 같아서 scaler가 필요 없다

---

[⬅️ Day11](Day11.md) · [🏠 전체 목차](../README.md) · [Day13 ➡️](Day13.md)
