# Day16 - 내 폴더 이미지로 분류하기 (softmax 확장), npy 저장/불러오기 분리, 사진 한 장 예측, 이미지 증폭

**학습 기간:** 2026-09-21

> Day15에서 배운 **ImageDataGenerator + npy 저장**을 horse-human / rps / man_woman 세 데이터에 적용했다. 2종 분류를 **softmax로 푸는 방법**을 배워 3종(가위바위보)까지 그대로 확장했고, **저장 → 훈련 → 예측**을 파일로 나누는 흐름을 익혔다. 마지막으로 사진 한 장을 모델에 넣어 예측하고, `datagen.flow()`로 **이미지 증폭**을 확인했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 2종을 softmax로 | `class_mode='categorical'` + `Dense(2, softmax)` + `categorical_crossentropy` + `np.argmax` |
| softmax의 장점 | 클래스가 늘어도 출력 칸 수만 바꾸면 된다 (sigmoid 1칸은 3종으로 못 늘린다) |
| class_indices | 클래스 번호는 폴더 이름 알파벳순. 외우지 말고 `xy_train.class_indices`로 확인한다 |
| train/test 폴더가 없을 때 | 상위 폴더를 한 번에 읽고 `train_test_split` + `stratify`로 직접 나눈다 |
| 파일 역할 분리 | 46 저장 → 47 훈련·모델 저장 → 49 예측. 느린 작업(이미지 읽기)을 한 번만 한다 |
| DirectoryIterator 인덱싱 | 앞의 `[0]`은 몇 번째 묶음, 뒤의 `[0] [1]`은 x냐 y냐 |
| load_img / img_to_array | 폴더가 아닌 **사진 한 장**을 predict 형태로 만들 때 쓴다 |
| np.expand_dims | 모델은 항상 4차원을 받는다. `(100,100,3)` → `(1,100,100,3)` |
| 예측 전 스케일링 | `img_to_array`는 0~255. 훈련이 0~1이었다면 예측할 사진도 `/255.` 해야 한다 |
| model.save / load_model | 구조 + 가중치 + 컴파일 설정을 한 파일로 저장하고 한 줄로 불러온다 |
| datagen.flow() | 배열 하나를 넣어 변형된 이미지를 계속 만들어 낸다 (증폭) |

---

## 📖 핵심 학습 내용

### 1. horse-human 2종 분류를 softmax로 (keras46_01)

- `_data/image/horse-human/horses` 500장 / `humans` 527장
- **train / test 폴더가 없다** → 상위 폴더 하나를 읽어 전부 받은 뒤 `train_test_split`으로 8:2로 나눴다
  - `stratify=np.argmax(y, axis=1)` : 원핫은 그대로 못 넣으므로 클래스 번호로 바꿔 비율을 맞춘다
- **2종인데 sigmoid가 아니라 softmax로 풀었다**

| | sigmoid | softmax |
|---|---|---|
| `class_mode` | `'binary'` | `'categorical'` |
| y 모양 | (N,) | (N, 2) |
| 출력층 | `Dense(1, 'sigmoid')` | `Dense(2, 'softmax')` |
| loss | `binary_crossentropy` | `categorical_crossentropy` |
| 예측 후처리 | `np.round` | `np.argmax(axis=1)` |

- **클래스 번호는 폴더 이름 알파벳순**으로 케라스가 정한다 → `print(xy_train.class_indices)`로 확인 (`{'horses': 0, 'humans': 1}`)
- 읽기 → 분리 → npy 저장 → 훈련 → 평가를 한 파일에서 했다
- 결과 : test acc 0.9854 (val_acc는 13 epoch에서 1.0 도달)

### 2. rps 3종 분류 (keras46_02)

- `paper` 840 / `rock` 840 / `scissors` 369 (총 2049장)
- **2종에서 3종으로 바뀔 때 고친 곳은 출력층 한 줄뿐이다** : `Dense(2, softmax)` → `Dense(3, softmax)`
  - `class_mode='categorical'`, loss, `np.argmax` 는 그대로
  - sigmoid 1칸 방식이었다면 이렇게 늘릴 수 없다 → **softmax를 쓰는 이유**
- 장수가 고르지 않아(scissors가 절반 이하) `stratify`로 비율을 맞춰 나눴다
- 원본이 300×300 PNG(RGBA)지만 `target_size=(100,100)`, `color_mode='rgb'`로 맞춰 읽는다 (투명 채널은 버려진다)
- **잘린 이미지가 섞여 있으면 읽다가 멈춘다**
  - `OSError: image file is truncated`
  - `from PIL import ImageFile` / `ImageFile.LOAD_TRUNCATED_IMAGES = True` 로 채워서 읽도록 허용
- 결과 : test acc 1.0

### 3. npy 저장 → 훈련 파일 분리 (keras47_01 ~ 03)

- keras46이 저장한 npy를 불러와 훈련만 한다
- 한 파일 안에서 **두 방법의 시간을 같이 재서 비교**했다
  - 방법 1 : ImageDataGenerator로 폴더 읽기
  - 방법 2 : `np.load`
  - `flow_from_directory()` 자체는 파일 목록만 훑는다. **실제로 읽는 시점은 `xy_train[0]`을 꺼낼 때**라 시간은 거기까지 재야 한다
  - 시간만 재고 `del`로 지운다 (man_woman은 x 하나가 3.3GB라 두 벌을 들고 있으면 메모리가 두 배)
- keras47_03은 훈련이 끝난 모델을 **`model.save()`로 고정된 이름으로 저장**한다
  - ModelCheckpoint 파일명에는 날짜가 들어가 실행할 때마다 달라진다 → 다른 파일에서 불러 쓰기 불편
  - `es`에 `restore_best_weights=True`가 있으므로 저장 시점 가중치는 가장 좋았던 상태다

### 4. man_woman - 저장 / 훈련 / 예측을 세 파일로 (keras46_03, keras47_03, keras49_02)

- `man` 17,678장 / `woman` 9,489장 (총 27,167장) - 지금까지 중 가장 큰 데이터
- **역할을 나눈 이유** : 이미지 읽기(약 30초)를 한 번만 하고, 훈련은 여러 번 반복하기 위해

| 파일 | 하는 일 |
|---|---|
| keras46_03 | 이미지를 읽어 train/test로 나눈 뒤 npy 저장 (훈련 없음) |
| keras47_03 | npy 불러와 훈련 + 모델 저장 |
| keras49_02 | 모델과 npy 불러와 사진 한 장 예측 (훈련 없음) |

- 용량 계산을 해 두면 돌리기 전에 감이 온다 : 100×100×3 float32 = 한 장 120KB → 27,167장 ≈ 3.3GB
  - `train_test_split`이 복사본을 만드는 순간 잠깐 2배까지 쓴다
- 남 65% / 여 35%로 기울어져 있어 `stratify=y`로 비율을 맞췄다
  - y가 0/1이라 원핫 변환 없이 바로 넣을 수 있다 (softmax였다면 `np.argmax` 필요)

### 5. 사진 한 장을 예측 형태로 만들기 (keras48_img_to_array)

- 지금까지는 ImageDataGenerator가 **폴더 전체**를 읽어 x, y를 만들어 줬다
- 내 사진 한 장은 정답(y)도 없고 폴더 구조도 없다 → 직접 만들어야 한다
```
load_img       : 사진 한 장을 target_size 로 열기 -> PIL.Image
img_to_array   : 숫자로 변환 -> (100, 100, 3) numpy
np.expand_dims : 축 추가 -> (1, 100, 100, 3)
```
- **모델은 언제나 (장수, 가로, 세로, 채널) 4차원을 받는다.** 한 장만 넣어도 '1장짜리 묶음'이어야 한다
- `img_to_array` 결과는 **0~255 그대로**다. 훈련 때 `rescale=1./255`를 썼으므로 예측 쪽에서 `/255.`를 해 줘야 한다
  - 이 줄을 빼면 모델이 훈련 중 본 적 없는 큰 숫자가 들어가 엉뚱한 값이 나온다

### 6. 저장한 모델로 사진 예측 (keras49_01, keras49_02)

- 훈련 없이 `load_model`로 불러오기만 한다

| | 설명 |
|---|---|
| `load_model` | 구조 + 가중치 + 컴파일 설정까지 한 파일에서 불러온다 |
| `load_weights` | 같은 구조의 모델을 먼저 만든 뒤 가중치만 덮어쓴다 |

- **사진을 넣기 전에 test 데이터로 모델 실력을 먼저 확인한다**
  - 불러온 모델의 acc가 0.5 근처면 무엇을 넣어도 반반이 나와 예측이 의미가 없다
  - 그때는 예측 코드가 아니라 훈련 쪽을 손봐야 한다
- 예측값 읽는 법 (sigmoid 1칸)
  - cat_dog : `cats 0 / dogs 1` → 값이 1에 가까우면 개상
  - man_woman : `man 0 / woman 1` → 값이 1에 가까우면 여자
  - 나머지 한쪽은 `1 - 값` (두 확률의 합은 1)

### 7. 이미지 증폭 (keras50_flow1)

- keras48을 베이스로, 사진 한 장을 **ImageDataGenerator로 변형**해 본다
- 지금까지는 `rescale`만 쓰고 나머지 옵션은 원본을 왜곡하므로 꺼 뒀는데, 여기서는 일부러 켠다
  - `horizontal_flip=True` (좌우반전), `width_shift_range=0.1` (평행 이동)
  - `rotation_range=15` (회전), `fill_mode='nearest'` (이동 후 빈 자리 채우기)
- **`flow()`** : 폴더가 아니라 **배열**을 넣는 방식 (`flow_from_directory`는 폴더)
  - 결과는 `NumpyArrayIterator` - 꺼낼 때마다 새로 변형된 이미지가 나온다
  - `it.next()` 는 Python 3.10 까지, 3.11 이후는 **`next(it)`**
  - `next(it).shape` → `(1, 100, 100, 3)`
- `plt.subplots(nrows=1, ncols=5)` 로 변형된 5장을 나란히 그려 확인

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras46_01_save_npy_horse.py` | horse-human 2종을 softmax로 - 읽기 / 분리 / npy 저장 / 훈련, acc 0.9854 |
| `keras46_02_save_npy_rps.py` | rps 3종 - `Dense(3, softmax)`, 잘린 PNG 처리(`LOAD_TRUNCATED_IMAGES`), acc 1.0 |
| `keras46_03_save_npy_man_woman.py` | man_woman 27,167장을 읽어 npy 저장만 (sigmoid 방식) |
| `keras47_01_load_npy_horse.py` | horse npy 불러와 훈련 - IDG 5.09초 vs npy 0.06초, acc 0.9709 |
| `keras47_02_load_npy_rps.py` | rps npy 불러와 훈련 - IDG 4.84초 vs npy 0.12초, acc 0.9976 |
| `keras47_03_load_npy_man_woman.py` | man_woman npy 불러와 훈련 + `model.save` - IDG 28.19초 vs npy 0.99초 |
| `keras48_img_to_array.py` | 사진 한 장 - `load_img` / `img_to_array` / `np.expand_dims` → npy 저장 |
| `keras49_01_내가고양이상인가.py` | cat_dog 모델 불러와 사진 예측 - test acc 0.787, 개상 94.61% |
| `keras49_02_내가남자게여자게.py` | man_woman 모델 불러와 사진 예측 - test acc 0.908, 남자 55.24% |
| `keras50_flow1.py` | `datagen.flow()` 로 사진 한 장을 증폭해 5장 출력 |
| `keras44_ImageDataGenerator3_CatDog.py` | learning_rate 관련 참고 주석 추가 (수정) |
| `keras45_04_catdog_load_npy.py` | learning_rate 관련 참고 주석 추가 (수정) |
| `.gitignore` | npy 폴더 규칙을 `_data/*_npy/` 로 정리 (수정) |
| `docs/Day16.md` | Day16 학습 기록 신규 작성 |
| `docs/Day15.md` | 하단 nav에 Day16 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조 / 진행도(16일, 20%) 갱신 (수정) |

---

## 📊 실행 결과

### 데이터셋별 결과

| 데이터 | 클래스 | 장수 | 출력층 | test acc |
|---|:---:|:---:|---|:---:|
| horse-human | 2 (horses/humans) | 1,027 | `Dense(2, softmax)` | 0.9854 |
| rps | 3 (paper/rock/scissors) | 2,049 | `Dense(3, softmax)` | 1.0 |
| man_woman | 2 (man/woman) | 27,167 | `Dense(1, sigmoid)` | 0.908 |
| cat_dog (Day15) | 2 (cats/dogs) | 10,028 | `Dense(1, sigmoid)` | 0.787 |

### ImageDataGenerator vs npy 로드

| 데이터 | 장수 | ImageDataGenerator | npy 로드 |
|---|:---:|---:|---:|
| horse-human | 1,027 | 5.09초 | **0.06초** |
| rps | 2,049 | 4.84초 | **0.12초** |
| man_woman | 27,167 | 28.19초 | **0.99초** |

**읽는 법:** 장수가 많을수록 차이가 커진다. ImageDataGenerator 시간은 같은 데이터라도 실행할 때마다 달라지는데(윈도우가 읽은 파일을 캐싱한다), npy는 항상 일정하다.

### 사진 한 장 예측

| 모델 | 결과 |
|---|---|
| cat_dog | 고양이상 5.39% / 개상 94.61% |
| man_woman | 남자 55.24% / 여자 44.76% |

---

## 💻 핵심 개념

### 2종을 softmax로 풀기

```python
xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),
    batch_size=1027,            # 전체 장수 -> 묶음이 1개
    class_mode='categorical',   # 원핫 -> y 가 (N, 2)
    color_mode='rgb',
    shuffle=True,
)
print(xy_train.class_indices)   # {'horses': 0, 'humans': 1}

model.add(Dense(2, activation='softmax'))       # 3종이면 Dense(3, softmax)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

y_predict_arg = np.argmax(y_predict, axis=1)    # sigmoid 였다면 np.round
y_test_arg = np.argmax(y_test, axis=1)
```

### train / test 폴더가 없을 때

```python
x = xy_train[0][0]      # [0] 을 꺼낼 때 실제로 이미지를 읽는다
y = xy_train[0][1]      # 앞 [0] = 묶음 번호 / 뒤 [0] [1] = x 냐 y 냐

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=42,
    shuffle=True,
    stratify=np.argmax(y, axis=1),  # 원핫은 클래스 번호로 바꿔서 넣는다 (binary 면 stratify=y)
)
```

### 잘린 이미지가 있을 때

```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # OSError: image file is truncated 방지
```

### npy 저장과 불러오기

```python
##### 저장 (keras46) #####
np_path = './_data/horse_npy/'
np.save(np_path + 'keras46_01_x_train.npy', arr=x_train)

##### 불러오기 (keras47) #####
x_train = np.load(np_path + 'keras46_01_x_train.npy')
```

### 훈련한 모델 저장 / 불러오기

```python
##### 저장 (keras47_03) #####
model.save(path + 'keras47_03_man_woman.keras')   # 구조 + 가중치 + 컴파일 설정

##### 불러오기 (keras49_02) #####
from tensorflow.keras.models import load_model
model = load_model('./_save/keras47/keras47_03_man_woman.keras')
```

### 사진 한 장을 predict 형태로

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img = load_img(path + 'image.png', target_size=(100,100))   # 훈련 때와 같은 크기
arr = img_to_array(img)             # (100, 100, 3), 값은 0 ~ 255
arr = np.expand_dims(arr, axis=0)   # (1, 100, 100, 3) -> 1장짜리 묶음

x_me = np.load(np_path + 'keras48_man.npy')
x_me = x_me/255.                    # 훈련이 0~1 이었으므로 예측도 맞춰 준다
```

### sigmoid 예측값 읽기

```python
y_me = model.predict(x_me)      # (1, 1) 짜리 확률
woman = float(y_me[0][0])       # 클래스 번호 1 쪽(woman)일 확률
man = 1 - woman                 # 나머지가 man

print('남자 :', round(man * 100, 2), '%')
print('여자 :', round(woman * 100, 2), '%')
```

### 이미지 증폭 (flow)

```python
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,       # 좌우반전
    width_shift_range=0.1,      # 평행 이동
    rotation_range=15,          # 회전
    fill_mode='nearest',        # 이동 후 빈 자리 채우기
)

it = datagen.flow(arr, batch_size=1)    # 폴더가 아니라 배열을 넣는다

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):
    batch = next(it)            # Python 3.11 이후. 3.10 까지는 it.next()
    batch = batch.reshape(100,100,3)
    ax[i].imshow(batch)
plt.show()
```

---

## 성과

- horse-human 2종을 softmax로 풀어 acc 0.9854, rps 3종으로 확장해 acc 1.0 달성
- 2종 → 3종 확장에서 바꾼 곳이 출력층 한 줄뿐이라는 것을 직접 확인
- man_woman 27,167장을 저장 / 훈련 / 예측 세 파일로 나눠 처리 (test acc 0.908)
- ImageDataGenerator와 npy 로드 시간을 세 데이터에서 비교 (최대 28배 차이)
- `load_img` / `img_to_array` / `np.expand_dims` 로 사진 한 장을 예측 형태로 만드는 법 습득
- 저장한 모델을 `load_model` 로 불러와 훈련 없이 예측하는 흐름 완성
- `datagen.flow()` 로 이미지 증폭 결과를 눈으로 확인

---

## 💡 주요 학습 포인트

1. **2종도 softmax로 풀 수 있다**: `categorical` + `Dense(2, softmax)` + `argmax`
2. **softmax는 클래스가 늘어도 출력 칸 수만 바꾸면 된다**: 2종 → 3종 확장이 한 줄
3. **클래스 번호는 폴더 이름 알파벳순**: `class_indices`로 확인하는 습관을 들인다
4. **폴더 이름이 바뀌면 번호도 바뀐다**: 예측값 해석도 같이 뒤집힌다
5. **train/test 폴더가 없으면 직접 나눈다**: `train_test_split` + `stratify`
6. **stratify는 원핫을 못 받는다**: `np.argmax(y, axis=1)` 로 번호를 넘긴다
7. **`[0]`을 꺼낼 때 이미지를 읽는다**: 시간 측정도 그 뒤에서 해야 맞다
8. **느린 작업은 한 번만**: 읽기(46) / 훈련(47) / 예측(49) 로 파일을 나눈다
9. **모델은 항상 4차원을 받는다**: 한 장도 `np.expand_dims` 로 묶음을 만든다
10. **예측할 사진도 훈련 때와 같은 범위로**: `img_to_array` 는 0~255 → `/255.` 필요
11. **`model.save` 는 고정된 이름으로**: mcp 파일명은 날짜가 붙어 매번 달라진다
12. **예측 전에 모델 실력부터 확인**: acc 0.5 근처 모델은 어떤 사진이든 반반이 나온다
13. **잘린 이미지 한 장이 전체를 멈춘다**: `LOAD_TRUNCATED_IMAGES = True`
14. **`flow`는 배열, `flow_from_directory`는 폴더**: 증폭은 배열 하나로도 할 수 있다

---

[⬅️ Day15](Day15.md) · [🏠 전체 목차](../README.md)
