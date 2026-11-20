# Day56 - BatchNormalization 위치 비교 (없음 · keras 방식 · 2015 논문 방식) & 논리 게이트 AND · OR · XOR & XOR 문제 해결 (다층 퍼셉트론 · DecisionTree)

**학습 기간:** 2026-11-20

> mnist CNN 에 BatchNormalization 을 넣지 않은 모델 / `층(relu) → BatchNormalization` (keras 방식) / `층(bias 없음) → BatchNormalization → Activation('relu')` (2015 논문 방식) 세 가지를 같은 시드 · 같은 구조로 돌려 파라미터 수와 loss · acc 를 비교했다 (keras81_batchNormalization1 ~ 3).
>
> Perceptron · LinearSVC 로 AND · OR 게이트는 정확도 1.0 으로 풀리지만, XOR 은 직선 하나로 나눌 수 없어 풀리지 않는 것을 확인했다 (m11_1 ~ m11_3). keras 에서도 은닉층 없는 `Dense(1, sigmoid)` 는 XOR 을 못 풀고 (m11_4), 은닉층(relu) 을 넣은 다층 퍼셉트론 (m14_5_xor_resolved1) 과 DecisionTreeClassifier (m14_5_xor_resolved2) 로 1.0 을 만들었다.

---

## 🎯 한눈에 보기

| 파일 | 모델 | 결과 |
|---|---|---|
| keras81_batchNormalization1 | Conv2D(relu) — BN 없음 | params 3,994,410 / loss 0.0345 / acc 0.9927 |
| keras81_batchNormalization2 | Conv2D(relu) → BN (keras 방식) | params 3,995,434 / loss 0.0296 / acc 0.9925 |
| keras81_batchNormalization3 | Conv2D(use_bias=False) → BN → Activation('relu') (논문 방식) | params 3,995,178 / loss 0.0242 / acc 0.9929 |
| m11_1_and | LinearSVC | AND 1.0 |
| m11_2_or | LinearSVC | OR 1.0 |
| m11_3_xor | Perceptron | XOR 0.5 (해결 안 됨) |
| m11_4_xor_unresolved | keras Dense(1, sigmoid) | XOR 0.75 이하 (해결 안 됨) |
| m14_5_xor_resolved1 | keras Dense(10, relu) → Dense(1, sigmoid) | XOR 1.0 |
| m14_5_xor_resolved2 | DecisionTreeClassifier | XOR 1.0 |

---

## 📖 핵심 학습 내용

### 1. BatchNormalization 없는 기준 모델 (keras81_batchNormalization1)

keras36_cnn3_mnist2 베이스로 층을 줄인 mnist CNN. 세 파일 모두 시드 333 고정, 스케일링 · Conv2D · Dense · Dropout · EarlyStopping(val_acc, patience 25) · batch_size 128 을 똑같이 맞추고 BatchNormalization 유무와 위치만 바꿨다.

```python
random.seed(333)        # Python 초기값(랜덤값) 고정
np.random.seed(333)     # Numpy 초기값(랜덤값) 고정
tf.random.set_seed(333) # Tensorflow 초기값(랜덤값) 고정

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))   # (22, 22, 64)
model.add(Dropout(0.25))
model.add(Flatten())                                          # 30976
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
# Total params: 3,994,410
```

### 2. keras 방식 : 층(relu) → BatchNormalization (keras81_batchNormalization2)

```python
from tensorflow.keras.layers import BatchNormalization

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
...
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
# Total params: 3,995,434 (훈련 가능 3,994,922 / 훈련 불가 512)
```

- 배치 단위로 층 출력을 채널마다 평균 0 · 분산 1 로 정규화한 뒤 `gamma`(크기) · `beta`(이동) 로 다시 조절
- 출력 shape 는 그대로, 파라미터는 채널마다 4 개 (gamma · beta 훈련 가능 / moving_mean · moving_variance 훈련 불가)
- BN 4 개 = (32 + 32 + 64 + 128) × 4 = 1,024 개 추가, 그중 512 개 훈련 불가
- 훈련 때는 그 배치의 평균 · 분산, 평가 · 예측 때는 모아 둔 moving_mean · moving_variance 사용

### 3. 2015 논문 방식 : 층(bias 없음) → BatchNormalization → Activation (keras81_batchNormalization3)

```python
from tensorflow.keras.layers import BatchNormalization, Activation

model.add(Conv2D(32, (3,3), activation=None, use_bias=False, input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
...
model.add(Dense(128, activation=None, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))      # 출력층은 BN 없이 softmax
# Total params: 3,995,178 (훈련 가능 3,994,666 / 훈련 불가 512)
```

- 논문(Ioffe & Szegedy, 2015) 은 활성화 함수 **전** 의 값을 정규화한 뒤 relu 를 통과시킨다 → 층에서 activation 을 빼고 `Activation('relu')` 층을 따로 둔다
- BN 이 평균을 빼면 bias 효과가 사라지고 beta 가 대신하므로 `use_bias=False` → bias 256 개(32 + 32 + 64 + 128) 만큼 파라미터가 줄어든다
- `Activation` 층 : 층 안의 activation='relu' 와 같은 계산, 파라미터 0 개

### 4. 세 방식 비교 (GPU, 시드 333)

| 파일 | 방식 | 멈춘 epoch (best) | loss | acc | 시간 |
|---|---|---|---|---|---|
| 1. BN 없음 | Conv2D(relu) | 54 (29) | 0.0345 | 0.9927 | 178.45초 |
| 2. keras 방식 | Conv2D(relu) → BN | 75 (50) | 0.0296 | 0.9925 | 462.44초 |
| 3. 2015 논문 방식 | Conv2D(bias X) → BN → relu | 80 (끝까지) | 0.0242 | 0.9929 | 238.62초 |

- loss 는 0.0345 → 0.0296 → 0.0242 로 BN 을 쓰면 낮아지고, relu 앞에 둘 때 더 낮았다
- acc 는 0.9925 ~ 0.9929 로 10000 장 중 2 ~ 4 장 차이 → mnist 에서는 거의 같다
- BN 을 쓴 2 · 3 번은 val_acc 가 늦게까지 올라 더 오래 훈련됐다. 3 번은 80 epoch 끝까지 가서 restore_best_weights 가 적용되지 않았다
- GPU 계산은 시드를 고정해도 완전히 같게 나오지 않아 다시 돌리면 순위가 바뀔 수 있다

### 5. 논리 게이트 AND · OR · XOR (m11_1 ~ m11_3)

```python
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC

x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 0, 0, 1]      # AND
# y_data = [0, 1, 1, 1]    # OR
# y_data = [0, 1, 1, 0]    # XOR

model = LinearSVC()        # 또는 Perceptron()
model.fit(x_data, y_data)
print(model.score(x_data, y_data))
# AND 1.0 / OR 1.0 / XOR(Perceptron) 0.5
```

- Perceptron · LinearSVC 는 직선(초평면) 하나로 0 과 1 을 나누는 선형 모델
- AND · OR 는 직선 하나로 나눌 수 있어(선형 분리 가능) 1.0
- XOR 은 1 끼리 대각선 위치라 직선 하나로 나눌 수 없다 → 단층 퍼셉트론의 한계 (XOR 문제), 최대 4 개 중 3 개(0.75)

### 6. keras 로 XOR — 은닉층 없음 vs 은닉층 있음 (m11_4 · m14_5_xor_resolved1)

```python
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 1, 1, 0])

# m11_4 : 해결 안 됨 (단층 퍼셉트론 = 로지스틱 회귀)
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
# acc 0.75 (실행마다 0.25 / 0.5 / 0.75, 0.75 를 넘지 못함)

# m14_5_xor_resolved1 : 해결 (다층 퍼셉트론)
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))   # 은닉층
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=500)
# acc 1.0

acc = accuracy_score(y_data, np.round(y_predict))     # sigmoid 확률 -> 0 / 1
```

- 은닉층이 입력을 새 특성으로 바꾸면 원래 공간에서는 꺾인 경계가 되어 XOR 을 나눌 수 있다
- 은닉층에 활성화 함수가 없으면 Dense 를 쌓아도 `(x·W1 + b1)·W2 + b2 = x·W + b` 로 직선 하나와 같다 → relu 같은 비선형 활성화가 핵심
- 데이터 4 개라 epoch 당 가중치 갱신이 한 번뿐 → epochs 100 에서는 0.5, 500 에서 1.0

### 7. DecisionTreeClassifier 로 XOR (m14_5_xor_resolved2)

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_data, y_data)                 # sklearn 모델 : compile · epochs 없음
print(model.score(x_data, y_data))        # 1.0
```

- "x1 <= 0.5 ?" → "x2 <= 0.5 ?" 처럼 여러 번 나눠 영역을 쪼개므로 선형 분리 불가능한 XOR 도 바로 풀린다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras2/keras81_batchNormalization1.py` | BatchNormalization 없는 mnist CNN 기준 모델, 시드 333 고정 (keras36_cnn3_mnist2 베이스) |
| `keras2/keras81_batchNormalization2.py` | 층(relu) → BatchNormalization keras 방식, 파라미터 · 훈련 불가 512 개 확인 |
| `keras2/keras81_batchNormalization3.py` | use_bias=False → BatchNormalization → Activation('relu') 2015 논문 방식 + 세 방식 비교 |
| `ml/m11_1_and.py` | Perceptron · LinearSVC 로 AND 게이트 → 1.0 |
| `ml/m11_2_or.py` | OR 게이트 → 1.0 |
| `ml/m11_3_xor.py` | XOR 게이트 → 0.5, 선형 모델로 해결 안 됨 |
| `ml/m11_4_xor_unresolved.py` | keras Dense(1, sigmoid) 단층 → XOR 해결 안 됨 |
| `ml/m14_5_xor_resolved1.py` | 은닉층 Dense(10, relu) 다층 퍼셉트론 → XOR 1.0 |
| `ml/m14_5_xor_resolved2.py` | DecisionTreeClassifier → XOR 1.0 |
| `docs/Day56.md` | Day56 학습 기록 신규 작성 |
| `docs/Day54.md` | 하단 nav에 Day56 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(keras2/ · ml/ Day56) / 진행도(56일, 70%) 갱신 (수정) |

---

## 💻 핵심 개념

### BatchNormalization 위치

```
keras 방식  : Conv2D(activation='relu')              -> BatchNormalization
논문 방식   : Conv2D(activation=None, use_bias=False) -> BatchNormalization -> Activation('relu')

BN 파라미터 : 채널 수 × 4  (gamma · beta 훈련 가능 / moving_mean · moving_variance 훈련 불가)
use_bias=False : BN 의 beta 가 bias 역할을 대신한다
```

### XOR 문제

```
x1 x2 | AND OR XOR
 0  0 |  0   0   0
 0  1 |  0   1   1
 1  0 |  0   1   1
 1  1 |  1   1   0

선형 모델 (Perceptron · LinearSVC · Dense(1, sigmoid)) : AND · OR 1.0 / XOR 최대 0.75
다층 퍼셉트론 (은닉층 + relu)                         : XOR 1.0
DecisionTreeClassifier                                 : XOR 1.0
```

---

## 성과

- BatchNormalization 의 위치(없음 · relu 뒤 · relu 앞) 를 같은 조건으로 비교해 파라미터 수 변화와 loss 차이를 확인했다
- 단층 퍼셉트론이 XOR 을 풀지 못하는 이유(선형 분리 불가능) 를 sklearn · keras 양쪽에서 확인했다
- 은닉층 + 비선형 활성화 함수, 결정 트리로 XOR 을 정확도 1.0 으로 해결했다

---

## 💡 주요 학습 포인트

1. **BatchNormalization**: 배치 단위로 층 출력을 평균 0 · 분산 1 로 정규화 + gamma · beta 로 조절
2. **BN 파라미터**: 채널당 4 개, 그중 moving_mean · moving_variance 는 훈련 불가 (Non-trainable)
3. **논문 방식**: 층(bias 없음) → BN → `Activation('relu')`, `use_bias=False`
4. **Activation 층**: 활성화 함수만 따로 분리한 층 (파라미터 0)
5. **시드 고정**: `random.seed` · `np.random.seed` · `tf.random.set_seed` 로 비교 조건 맞추기
6. **Perceptron · LinearSVC**: 직선 하나로 나누는 선형 모델 → AND · OR 는 풀고 XOR 은 못 푼다
7. **XOR 문제**: 선형 분리 불가능 → 단층 퍼셉트론의 한계
8. **다층 퍼셉트론**: 은닉층 + 비선형 활성화(relu) 로 XOR 해결, 활성화 없이 쌓으면 여전히 직선 하나
9. **DecisionTreeClassifier**: 영역을 여러 번 쪼개 XOR 해결 (compile · epochs 없음)

---

[⬅️ Day54](Day54.md) · [🏠 전체 목차](../README.md)
