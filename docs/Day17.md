# Day17 - 이미지 증폭(augmentation)으로 훈련 데이터 늘리기, 클래스 불균형 맞추기

**학습 기간:** 2026-09-22

> Day16에서 사진 한 장을 `datagen.flow()`로 변형해 본 것을 **데이터셋 전체**로 넓혔다. 원본에서 일부를 랜덤으로 뽑아 변형한 뒤 **원본 + 변형본을 합쳐 훈련 데이터를 늘리는 흐름**을 fashion_mnist / mnist / cifar10 / cifar100에 적용했다. 마지막으로 man_woman에서 **적은 쪽(여자)만 골라 증폭해 남녀 장수를 1 : 1로 맞추고**, 증폭 데이터가 섞일 때 검증 데이터를 어디서 떼야 하는지 익혔다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| np.tile | 한 장을 여러 장으로 복사한다. 복사만 하면 전부 똑같아 변형이 필요하다 |
| flow(x, y) 결과 | `(x 묶음, y 묶음)` 튜플. `[0]`이 x, `[1]`이 y |
| batch_size = 전체 장수 | `.next()` 한 번에 전부 변형된다 |
| 증폭 흐름 | 인덱스 랜덤 추출 → `flow`로 변형 → `np.concatenate`로 원본에 합치기 |
| shuffle=False | 변형된 x와 원래 y의 순서(짝)를 유지한다 |
| rescale 끄기 | 원본과 증폭본을 합친 뒤 한 번에 스케일링해야 두 덩어리의 범위가 같다 |
| 증폭은 train만 | test(와 val)는 실제로 들어올 데이터 그대로 둬야 평가가 의미 있다 |
| randint vs choice | `randint`는 중복 허용, `choice(replace=False)`는 중복 불가(원본보다 많이 못 뽑는다) |
| 흑백 vs 컬러 | 흑백은 `(N,28,28)` → 4차원 reshape 필요, 컬러는 처음부터 4차원 |
| 한쪽만 증폭 | `np.where(y == 1)`로 적은 클래스만 골라 증폭 → 클래스 비율 맞추기 |
| augment_size 계산 | `많은 쪽 장수 - 적은 쪽 장수`로 정하면 정확히 1 : 1 |
| validation_split | 섞지 않고 **맨 뒤** 비율만큼 뗀다. 증폭본을 뒤에 붙였다면 쓰면 안 된다 |
| val은 증폭 전에 | `train_test_split`으로 먼저 떼고 train만 증폭, `validation_data=(x_val, y_val)` |

---

## 📖 핵심 학습 내용

### 1. 한 장을 100장으로 증폭 (keras50_flow2_next)

- keras50_flow1은 사진 한 장을 `batch_size=1`로 넣고 `next`를 5번 불러 5장을 꺼냈다
- 여기서는 fashion_mnist 한 장을 **`np.tile`로 100장 복사**한 뒤 한 번에 변형했다
  - `np.tile(x_train[0], 100)` → `(28, 2800)` : 2차원을 그대로 넣으면 가로로 이어 붙는다
  - `x_train[0].reshape(28*28)`로 한 줄로 편 뒤 tile → `.reshape(-1, 28, 28, 1)` → `(100, 28, 28, 1)`
  - **복사만 한 100장은 완전히 똑같다** → 그대로 쓰면 같은 사진을 외우게 되므로 변형을 거쳐야 한다
- `flow(x, y)`에 y를 같이 넣으면 결과가 **튜플**이다
  - 정답이 필요 없어 `np.zeros(100)`으로 자리만 채운 가짜 y를 넣었다
  - `xy_data.shape` → `AttributeError: 'tuple' object has no attribute 'shape'`
  - `len(xy_data)` → 2, `xy_data[0].shape` → `(100, 28, 28, 1)`, `xy_data[1].shape` → `(100,)`
- `batch_size=100`이면 `.next()` 한 번에 100장이 모두 나오고, 각각 다르게 변형되어 있다
- `plt.subplot(7, 7, i+1)`로 49장을 격자로 그려 확인했다

### 2. 데이터셋 증폭의 흐름 (keras51_augment1 ~ 4)

```
1) 원본에서 augment_size 만큼 인덱스를 랜덤으로 뽑는다   np.random.randint / np.random.choice
2) 뽑은 이미지를 복사한다                               x_train[randidx].copy()
3) flow 에 통과시켜 변형한다                            datagen.flow(...).next()[0]
4) 원본 + 변형본을 합친다                               np.concatenate
5) 합친 뒤 스케일링 -> reshape -> 원핫 -> 훈련
```

- `.copy()` : 새 메모리에 담아 원본 `x_train`이 함께 바뀌는 것을 막는다
- `shuffle=False` : 섞이면 변형된 x와 `y_augmented`의 짝이 어긋난다
- `.next()[0]` : 튜플에서 x만 받는다. y는 변형되지 않으므로 `y_augmented`를 그대로 쓴다
- **datagen의 `rescale`을 끈다**
  - 증폭본만 0~1로 나오고 원본은 0~255면 합친 데이터의 범위가 뒤섞인다
  - 합친 뒤 한 번에 `/255.` (또는 `(x-127.5)/127.5`) 하고, **x_test도 같은 식으로** 바꾼다
- **증폭은 train에만** 한다. test는 실제로 들어올 데이터 그대로여야 한다
- 모델은 Day15에서 함수형으로 바꾼 DNN(keras41_dnn1~4 구조)을 그대로 쓰고, 데이터만 늘려 비교했다

### 3. 흑백과 컬러의 차이

| | fashion_mnist / mnist | cifar10 / cifar100 |
|---|---|---|
| 원본 x | `(60000, 28, 28)` 3차원 | `(50000, 32, 32, 3)` 4차원 |
| flow 전 reshape | `(40000, 28, 28, 1)`로 필요 | 필요 없음 |
| 원본 y | `(60000,)` 1차원 | `(50000, 1)` 2차원 |
| 원핫 전 reshape | `reshape(-1, 1)` 필요 | 바로 넣는다 |
| 스케일링 | `/255.` → 0 ~ 1 | `(x-127.5)/127.5` → -1 ~ 1 |
| 증폭 후 train | 60000 + 40000 = **100000** | 50000 + 40000 = **90000** |

- mnist는 증폭 전과 결과가 거의 같았다 (0.9784 → 0.9783)
  - 숫자는 좌우로 뒤집으면 모양이 달라진다 → 옷(fashion)처럼 뒤집어도 같은 대상인 데이터와 달리 `horizontal_flip`이 도움이 안 될 수 있다

### 4. 뽑는 방법 두 가지 (keras51_augment4)

| | `np.random.randint(n, size=k)` | `np.random.choice(n, size=k, replace=False)` |
|---|---|---|
| 중복 | 같은 번호가 여러 번 나올 수 있다 | 서로 다른 번호만 |
| 원본보다 많이 뽑기 | 가능 | **불가** (k ≤ n) |

- cifar100은 클래스가 100개라 한 종류당 500장뿐이다 → 증폭으로 늘리는 효과가 컸다 (0.2972 → 0.3326)

### 5. 한쪽 클래스만 증폭해 비율 맞추기 (keras51_augment5_man_woman_여자만)

- man_woman train은 **남 14142 / 여 7591**로 남자가 약 2배 → 이대로면 "애매하면 남자"로 기울기 쉽다
- **적은 쪽(여자)만 골라서** 증폭한다
  - `np.where(y_train == 1)[0]` : 조건을 만족하는 **위치(인덱스)**만 모은다
  - `pick = woman_idx[randidx]` : 전체 인덱스가 아니라 **여자 목록 안에서** 고른다
- **augment_size를 숫자로 고정하지 않고 계산한다** : `len(man_idx) - len(woman_idx)` → 정확히 1 : 1
- npy가 keras46_03에서 `rescale=1./255`로 읽어 저장한 것이라 **이미 0~1**이다
  - datagen에도 `rescale`을 쓰지 않고, 합친 뒤에도 나누지 않는다 (두 번 나누면 값이 0 근처로 뭉개진다)

### 6. 증폭 데이터가 있을 때 검증 데이터 떼기

- **`validation_split`은 섞지 않고 맨 뒤 비율만큼 뗀다**
  - 증폭본을 `np.concatenate`로 뒤에 붙였다면, 검증 데이터가 **전부 증폭한 여자 사진**이 된다
  - 훈련에서는 여자가 다시 줄어 균형이 깨지고, val_acc는 "여자를 여자라고 맞히는 비율"만 재게 된다 → es / mcp 기준이 틀어진다
- **합친 뒤에 `train_test_split`으로 나눠도** 같은 사진의 원본과 증폭본이 train / val에 나뉘어 들어가 val_acc가 부풀려진다
- 그래서 **증폭하기 전에** 원본 train에서 val을 먼저 떼고, 남은 train의 여자 사진만 증폭했다

| 단계 | 남 | 여 |
|---|---:|---:|
| 원본 train | 14142 | 7591 |
| val 분리 후 train (`stratify`) | 11313 | 6073 |
| val (원본 그대로) | 2829 | 1518 |
| 여자 5240장 증폭 후 train | 11313 | **11313** |

- 훈련은 `validation_data=(x_val, y_val)`로 떼어 둔 원본 val을 넣는다
- mnist / cifar(augment1~4)는 전체에서 랜덤으로 뽑아 증폭해 클래스가 한쪽으로 쏠리지 않았다. 한쪽만 증폭할 때 특히 주의해야 한다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras50_flow2_next.py` | fashion_mnist 한 장을 `np.tile`로 100장 복사 → `flow(x, y).next()`로 한 번에 변형, 튜플 구조 확인 |
| `keras51_augment1_fashion.py` | fashion_mnist 4만 장 증폭 → 10만 장, 함수형 DNN, acc 0.879 |
| `keras51_augment2_mnist.py` | mnist 4만 장 증폭 → 10만 장, 함수형 DNN, acc 0.9783 |
| `keras51_augment3_cifar10.py` | cifar10 4만 장 증폭 → 9만 장, -1~1 스케일링, acc 0.5433 |
| `keras51_augment4_cifar100.py` | cifar100 4만 장 `choice(replace=False)` 증폭, BatchNormalization DNN, acc 0.3326 |
| `keras51_augment5_man_woman_여자만.py` | 여자만 5240장 증폭해 1 : 1, 증폭 전 `train_test_split`으로 val 분리, acc 0.9082 |
| `keras50_flow1.py` | 증폭 목적 / `flow` vs `flow_from_directory` / 옵션별 설명 주석 보완 (수정) |
| `docs/Day17.md` | Day17 학습 기록 신규 작성 |
| `docs/Day16.md` | 하단 nav에 Day17 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조 / 진행도(17일, 21%) 갱신 (수정) |

---

## 📊 실행 결과

### 증폭 전후 비교 (test acc)

| 데이터 | 모델 | 증폭 전 | 증폭 후 | train 장수 | 소요 시간 |
|---|---|:---:|:---:|:---:|---:|
| fashion_mnist | 함수형 DNN | 0.8788 | **0.879** | 60000 → 100000 | 154.43초 |
| mnist | 함수형 DNN | 0.9784 | 0.9783 | 60000 → 100000 | 595.48초 |
| cifar10 | 함수형 DNN | 0.519 | **0.5433** | 50000 → 90000 | 306.04초 |
| cifar100 | 함수형 DNN + BatchNormalization | 0.2972 | **0.3326** | 50000 → 90000 | 889.23초 |
| man_woman | CNN (keras47_03 구조) | 0.908 | **0.9082** | 17386 → 22626 (남녀 1 : 1) | 427.09초 |

**읽는 법:** 컬러 데이터(cifar10 / cifar100)에서 증폭 효과가 가장 컸다. mnist는 거의 변화가 없었다. man_woman은 test acc는 비슷하지만, 남녀 장수를 맞춰 한쪽으로 기우는 것을 막은 상태의 결과다.

---

## 💻 핵심 개념

### 한 장을 여러 장으로 (np.tile + flow)

```python
augment_size = 100

xy_data = datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),  # (100, 28, 28, 1)
    np.zeros(augment_size),     # 정답이 필요 없어 자리만 채운 가짜 y
    batch_size=augment_size,    # .next() 한 번에 100장
    shuffle=False,
).next()

print(type(xy_data))    # <class 'tuple'>
print(xy_data[0].shape) # (100, 28, 28, 1) -> x
print(xy_data[1].shape) # (100,)           -> y
```

### 데이터셋 증폭 (흑백)

```python
datagen = ImageDataGenerator(
    # rescale=1./255,           # 끈다 -> 합친 뒤 한 번에 스케일링
    horizontal_flip=True,
    width_shift_range=0.1,
    rotation_range=15,
    fill_mode='nearest',
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
x_augmented = x_augmented.reshape(-1, 28, 28, 1)    # 흑백은 4차원으로 (컬러는 생략)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,              # x 와 y 의 짝 유지
).next()[0]                     # [0] = x

x_train = x_train.reshape(60000, 28, 28, 1)
x_train = np.concatenate((x_train, x_augmented)) / 255.    # 합친 뒤 스케일링
y_train = np.concatenate((y_train, y_augmented))
x_test = x_test.reshape(10000, 28, 28, 1) / 255.           # test 도 같은 식으로
```

### 중복 없이 뽑기

```python
randidx = np.random.randint(50000, size=40000)                 # 중복 허용
randidx = np.random.choice(50000, size=40000, replace=False)   # 중복 불가 (원본보다 많이 못 뽑는다)
```

### 한쪽 클래스만 증폭 + 증폭 전에 val 분리

```python
# 1) 증폭 전에 val 을 먼저 뗀다
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y_train,
)

# 2) 적은 쪽(여자)만 골라 부족한 장수만큼 증폭
man_idx = np.where(y_train == 0)[0]
woman_idx = np.where(y_train == 1)[0]
augment_size = len(man_idx) - len(woman_idx)       # 11313 - 6073 = 5240

randidx = np.random.randint(len(woman_idx), size=augment_size)
pick = woman_idx[randidx]                          # 여자 목록 안에서 고른다

x_augmented = x_train[pick].copy()
y_augmented = y_train[pick].copy()
x_augmented = datagen.flow(x_augmented, y_augmented,
                           batch_size=augment_size, shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_augmented))   # 이미 0~1 이라 나누지 않는다
y_train = np.concatenate((y_train, y_augmented))

# 3) 떼어 둔 원본 val 로 검증
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_data=(x_val, y_val),
          callbacks=[es, mcp])
```

---

## 성과

- 사진 한 장 증폭(Day16)을 데이터셋 전체 증폭으로 확장해 네 데이터셋에 적용
- fashion_mnist / mnist 10만 장, cifar10 / cifar100 9만 장으로 훈련 데이터를 늘려 증폭 전후 비교
- cifar10 0.519 → 0.5433, cifar100 0.2972 → 0.3326으로 컬러 데이터에서 증폭 효과 확인
- `randint`와 `choice(replace=False)`의 차이를 직접 써 보며 확인
- man_woman에서 여자만 5240장 증폭해 남녀 1 : 1로 맞추고 test acc 0.9082
- 증폭 데이터가 있을 때 `validation_split` 대신 증폭 전 `train_test_split` + `validation_data`로 검증 데이터를 떼는 흐름 완성

---

## 💡 주요 학습 포인트

1. **복사만 한 데이터는 쓰지 않는다**: `np.tile`로 늘린 100장은 전부 똑같다 → 변형이 필요하다
2. **flow(x, y)는 튜플을 돌려준다**: `[0]`이 x, `[1]`이 y. `.shape`는 튜플에 없다
3. **batch_size를 전체 장수로**: `.next()` 한 번에 전부 변형된다
4. **shuffle=False로 짝을 지킨다**: 섞이면 변형된 x와 y가 어긋난다
5. **`.copy()`로 원본을 보호한다**: 뽑은 배열을 바꿔도 `x_train`은 그대로
6. **rescale은 끄고 합친 뒤 한 번에**: 원본과 증폭본의 범위를 같게, test도 같은 식으로
7. **이미 스케일링된 데이터는 다시 나누지 않는다**: npy가 0~1이면 rescale도 `/255.`도 빼야 한다
8. **증폭은 train에만**: test와 val은 실제로 들어올 데이터 그대로
9. **흑백은 4차원으로 reshape해서 flow에 넣는다**: 컬러는 처음부터 4차원
10. **`choice(replace=False)`는 원본보다 많이 못 뽑는다**: 더 많이 필요하면 `randint`
11. **데이터 특성에 맞는 옵션을 고른다**: 숫자는 좌우반전하면 모양이 달라져 효과가 없었다
12. **적은 클래스만 증폭해 비율을 맞출 수 있다**: `np.where`로 인덱스를 골라 그 안에서 뽑는다
13. **augment_size는 계산으로**: `많은 쪽 - 적은 쪽`이면 정확히 1 : 1
14. **validation_split은 맨 뒤를 뗀다**: 증폭본을 뒤에 붙였다면 검증 데이터가 증폭본으로만 채워진다
15. **val은 증폭 전에 뗀다**: 합친 뒤 나누면 같은 사진의 원본과 증폭본이 train / val에 나뉘어 val_acc가 부풀려진다

---

[⬅️ Day16](Day16.md) · [🏠 전체 목차](../README.md)
