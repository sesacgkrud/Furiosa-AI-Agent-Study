# Day10 - 스케일러 4종 비교와 모델 저장·불러오기(save/load)

**학습 기간:** 2026-09-11

> 스케일러 4종을 10개 데이터셋에 전부 적용해 비교하고, **모델 저장/불러오기(save·load)** 를 저장 시점별로 실습했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 스케일러 4종 | MinMax `(x-Min)/(Max-Min)` / Standard `(x-평균)/표준편차` / MaxAbs `x/절댓값 최대` / Robust `(x-중앙값)/IQR` |
| 0~1 로 수렴하는 것 | MinMax · MaxAbs 뿐. Standard · Robust 는 범위가 정해지지 않는다 |
| 정답 스케일러 | 없다. 데이터마다 다르다 (covtype·따릉이 Robust / cancer·bike Standard) |
| 1회 실행의 한계 | 같은 코드 재실행에서 california 0.6695 -> 1.0736 (시드 미고정) |
| model.save | `fit` **전** 저장 = 구조만 / `fit` **후** 저장 = 가중치 + 옵티마이저 |
| load_model | 훈련까지 끝난 파일은 `compile` 없이 바로 `evaluate` 가 된다 |
| path 함정 | 폴더까지만 적고 반드시 `/` 로 끝낸다 |

---

## 📖 핵심 학습 내용

### 스케일러 3종 추가 (StandardScaler / MaxAbsScaler / RobustScaler)
- Day09의 `MinMaxScaler`에 이어 스케일러 3종을 추가로 적용하고 **10개 데이터셋 전부에서 4종을 비교**
- 각 스케일러의 공식과 결과 범위가 다르다
  - `StandardScaler` = `(원값 - 평균) / 표준편차` -> 평균 0, 표준편차 1 (0~1로 묶이지 않는다)
  - `MaxAbsScaler` = `원값 / |최댓값|` -> -1 ~ 1
  - `RobustScaler` = `(원값 - 중앙값) / IQR` -> 중앙값 0, 이상치의 영향을 덜 받는다
- `MinMaxScaler` / `MaxAbsScaler`와 달리 `StandardScaler` / `RobustScaler`는 **0~1로 수렴하지 않는다**
  - california + RobustScaler는 `Min -7.2267 / Max 1455.8382`가 정상 출력 - 이상치가 그대로 남기 때문
  - 그래서 스케일러를 바꾸면 `print('Min :', ...)` 주석도 같이 고쳐야 한다
- **데이터마다 가장 잘 맞는 스케일러가 다르다** - 모든 데이터에 통하는 정답 스케일러는 없다
  - covtype / 따릉이 -> `RobustScaler`, cancer / kaggle bike -> `StandardScaler`, california / boston / wine / digits -> `MinMaxScaler`
- **한 번의 실행 결과로 우열을 단정할 수 없다**
  - california를 같은 코드로 재실행하니 `RobustScaler` loss가 0.6695 -> 1.0736으로 벌어졌다
  - 이 편차가 스케일러 간 차이(0.504 ~ 0.670)보다 크므로, 비교하려면 seed를 고정하거나 여러 번 돌려 평균을 봐야 한다
- 스케일링은 **연산 속도를 바꾸지 않는다** - 소요 시간이 변하는 건 `EarlyStopping`이 걸리는 시점이 달라져서다
  - covtype: MinMax 1110초 / Standard 866초 / MaxAbs 1091초 / Robust 664초
  - wine: MinMax 116초 / Standard 115초 / MaxAbs 11초 / Robust 115초

### 모델 저장과 불러오기 (model.save / load_model)
- **`model.save(경로)`** - 모델을 파일로 저장, **`load_model(경로)`** - 저장한 파일을 그대로 불러오기
- 확장자 두 가지: `.h5`(예전 방식) / `.keras`(최근 방식)
- **저장 시점에 따라 파일에 담기는 내용이 다르다** - 오늘 실습의 핵심
  - `fit` **전**에 저장 -> 모델 **구조**만 저장 (`Total params 6,001`)
  - `fit` **후**에 저장 -> 구조 + **훈련된 가중치 + 옵티마이저 상태** (`Total params 18,005`, `Optimizer params 12,004`)
- 훈련 전에 저장한 파일은 불러온 뒤 `compile` + `fit`을 다시 해야 한다 (구조만 있으므로)
- 훈련 후에 저장한 파일은 **`compile` 없이 바로 `evaluate`가 된다** - 컴파일 정보(`loss='mse'`, `optimizer='adam'`)까지 저장되기 때문
- `EarlyStopping(restore_best_weights=True)`로 되돌린 **최적 가중치**가 저장된다
- **`path`는 폴더까지만 적고 반드시 `/`로 끝낸다**
  - `'./_save/keras29'`처럼 슬래시를 빼면 뒤의 파일명과 그대로 이어붙어 `_save/keras29keras29_1_save_model.keras`라는 엉뚱한 파일이 생긴다
  - 만들어둔 `_save/keras29/` 폴더는 빈 채로 남아 있어서 눈치채기 어렵다
- 저장한 모델을 불러와 평가하면 **loss가 소수점 끝까지 똑같이 재현된다**
  - `evaluate`는 학습이 아니라 계산이라 무작위 요소가 없고, `random_state=100`으로 분할이 고정, 스케일러 변환도 고정, 가중치는 파일에 저장된 값이기 때문
  - 단, 저장 파일은 **마지막 실행 것으로 덮어써지므로** 저장 파일(keras29_3) 실행 **직후**에 불러오기 파일(keras29_4)을 돌려야 짝이 맞는다

### 코드 점검·정리
- 스케일러만 바꾸고 주석을 그대로 두면 **주석이 거짓말을 한다** - Min/Max 주석을 실제 출력값으로 전부 교정
- 훈련 loss가 평가 loss보다 높게 나올 수 있다 - RobustScaler + california는 **최대 이상치(1455.8)가 x_train 쪽**에 있고 x_test 최대는 586.4라서
- 불러오기 전용 파일(keras29_2, keras29_4)에서는 `hist`가 없어 그래프 블록이 동작하지 않는다 (빈 창 + `No artists with labels found` 경고)

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras28_Scaler01_california.py` | California 회귀 - Standard 0.5426 / MaxAbs 0.5590 / Robust 0.6695 추가 측정. 재실행 시 Robust가 1.0736으로 벌어져 **seed 미고정 편차**를 확인하고 주의사항 기록. Min/Max 주석을 RobustScaler 실제 값으로 교정 (수정) |
| `keras28_Scaler02_diabetes.py` | Diabetes 회귀 - Standard 3279 / MaxAbs 3042 / Robust 3233. 미적용(2874)이 가장 좋아 **이미 정규화된 데이터에는 효과가 없음**을 재확인 (수정) |
| `keras28_Scaler03_boston.py` | Boston 회귀 - Standard 20.86 / MaxAbs 20.24 / Robust 21.35. CRIM(0 ~ 89) / TAX(187 ~ 711) / B(0 ~ 396)처럼 컬럼 범위가 제각각이라 4종 모두 미적용(22.8)보다 개선 (수정) |
| `keras28_Scaler04_dacon_ddareung.py` | 따릉이 회귀 - Standard r2 0.6575 / MaxAbs r2 0.7047 / **Robust r2 0.7460 (loss 1870.78, RMSE 43.25)로 최고 기록** (수정) |
| `keras28_Scaler05_kaggle_bike.py` | Kaggle Bike 회귀 - **Standard 21298 (r2 0.3179)가 최고**, MaxAbs 21775 / Robust 22059. 편차가 작아 '좋아졌다'고 결론 내리면 안 되는 케이스 (수정) |
| `keras28_Scaler06_cancer.py` | Breast Cancer 이진 분류 - **Standard acc_score 0.9825로 최고**, MaxAbs / Robust 모두 0.9649. test 171개라 숫자 하나로 단정하지 말 것을 주석에 기록 (수정) |
| `keras28_Scaler07_santander.py` | Santander 다중 분류 - Standard 0.9109 / MaxAbs 0.9131 / Robust 0.9116. var_0~var_199가 이미 비슷한 크기라 4종 모두 변화 없음. 제출 파일 새로 생성 (수정) |
| `keras28_Scaler08_wine.py` | Wine 다중 분류 - Standard / MaxAbs / Robust 모두 acc_score 0.9630 (MinMax만 0.9815). 소요 시간 116 / 115 / 11 / 115초로 들쭉날쭉한 이유가 EarlyStopping 시점 차이임을 기록 (수정) |
| `keras28_Scaler09_fetch_covtype.py` | Covtype 다중 분류 - **Robust acc_score 0.9426으로 최고 기록 갱신** (MinMax 0.9361 / Standard 0.9361 / MaxAbs 0.9338). 소요 시간도 664초로 가장 짧음 (수정) |
| `keras28_Scaler10_digits.py` | Digits 다중 분류 - Standard 0.9444 / MaxAbs 0.9694 / Robust 0.9374. 픽셀 0~16으로 단위가 이미 같은 **대조군**이라 스케일러를 바꿔도 개선이 없음 (수정) |
| `keras29_1_save_model.py` | `model.save()` 입문 - 훈련 **전**에 저장해서 구조만 담기는 것을 확인 (`Total params 6,001`). `.h5` / `.keras` 두 방식 비교, `exit()`로 훈련 전에 종료 |
| `keras29_2_load_model.py` | `load_model()` 입문 - keras29_1이 저장한 구조를 불러와 `compile` + `fit` 수행 (loss 2.4101). `#2. 모델 구성`을 통째로 주석 처리 |
| `keras29_3_save_model2.py` | `model.save()`를 `fit` **아래**로 옮겨 훈련된 가중치까지 저장 (loss 1.7271) |
| `keras29_4_load_model2.py` | 훈련된 모델을 불러와 `compile`/`fit` 없이 바로 `evaluate` - **keras29_3과 동일한 loss 1.7271162271499634 재현** |
| `_save/keras29/keras29_1_save_model.keras` | 훈련 전 저장 결과물 (구조만, 42KB) |
| `_save/keras29/keras29_3_save_model.keras` | 훈련 후 저장 결과물 (가중치 + 옵티마이저 상태, 97KB) |
| `_data/kaggle_santander/submit/submit_0910_1724_scaler.csv` | 스케일링 적용본 Santander 제출 파일 |

---

## 📊 실행 결과 - 스케일러 4종 비교 (10개 데이터셋)

### 회귀 - loss (낮을수록 좋음)

| 데이터셋 | 미적용 | MinMax | Standard | MaxAbs | Robust | 최고 |
|---|---|---|---|---|---|---|
| California | 0.635 | **0.504** | 0.543 | 0.559 | 0.670 | MinMax |
| Diabetes | **2874** | 2971 | 3279 | 3042 | 3233 | 미적용 |
| Boston | 22.82 | **18.74** | 20.86 | 20.24 | 21.35 | MinMax |
| 따릉이 | 3192.95 | 1995.12 | 2522.89 | 2175.08 | **1870.78** | Robust |
| Kaggle Bike | 21894 | 22418 | **21298** | 21775 | 22059 | Standard |

따릉이 r2: MinMax 0.7292 / Standard 0.6575 / MaxAbs 0.7047 / **Robust 0.7460**
Kaggle Bike r2: MinMax 0.2820 / **Standard 0.3179** / MaxAbs 0.3026 / Robust 0.2936

### 분류 - accuracy_score (높을수록 좋음)

| 데이터셋 | 미적용 | MinMax | Standard | MaxAbs | Robust | 최고 |
|---|---|---|---|---|---|---|
| Cancer | 0.9240 | 0.9708 | **0.9825** | 0.9649 | 0.9649 | Standard |
| Santander | 0.9113 | **0.9146** | 0.9109 | 0.9131 | 0.9116 | 차이 없음 |
| Wine | 0.9630 | **0.9815** | 0.9630 | 0.9630 | 0.9630 | MinMax |
| Covtype | 0.8612 | 0.9361 | 0.9361 | 0.9338 | **0.9426** | Robust |
| Digits | 0.9652 | **0.9736** | 0.9444 | 0.9694 | 0.9374 | 대조군 |

**읽는 법:** 표의 '최고'는 이번 1회 실행 기준이다. seed를 고정하지 않아 california처럼 같은 코드가 0.6695 / 1.0736으로 갈리는 경우가 있으므로, 값이 근소하게 갈리는 데이터(santander, wine, digits)는 우열을 단정하지 않는다.

---

## 성과

- 10개 데이터셋 × 스케일러 4종 = **40개 조합을 모두 실행하고 결과를 파일 하단에 기록**
- covtype을 `RobustScaler`로 **acc_score 0.9426까지 끌어올려 Day09 최고 기록(0.9361)을 갱신**하고, 소요 시간도 1110초 -> 664초로 단축
- 따릉이에서 `RobustScaler`로 loss 1870.78 / r2 0.7460 달성 (Day09 MinMax 1995.12 / 0.7292 대비 개선)
- 스케일러마다 결과 범위가 다르다는 것을 `Min/Max` 출력으로 직접 확인하고 낡은 주석을 전부 교정
- 같은 코드를 재실행해 **seed 미고정 편차가 스케일러 간 차이보다 클 수 있다**는 것을 수치로 확인
- `model.save()` / `load_model()`을 저장 시점(훈련 전/후)으로 나눠 4단계로 실습하고, 불러온 모델이 **소수점 끝까지 같은 loss를 재현**하는 것까지 검증
- `path` 슬래시 누락으로 파일이 엉뚱한 이름으로 저장되던 문제를 찾아 `_save/keras29/` 아래로 정리

---

## 💻 핵심 개념

### 스케일러 4종 - 공식과 결과 범위

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler

# MinMaxScaler  : (원값 - Min) / (Max - Min)   -> 0 ~ 1
# StandardScaler: (원값 - 평균) / 표준편차       -> 평균 0, 표준편차 1 (범위 안 정해짐)
# MaxAbsScaler  : 원값 / |최댓값|                -> -1 ~ 1
# RobustScaler  : (원값 - 중앙값) / IQR          -> 중앙값 0, 이상치 영향 적음

# 비교할 때는 스케일러 줄만 바꾸고 나머지는 절대 건드리지 않는다 (A/B 테스트)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train) # fit + transform 을 한 줄로
x_test  = scaler.transform(x_test)      # test 는 transform 만 (데이터 누수 방지)

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : -7.2267 Max : 1455.8382
print('Min :', np.min(x_test),  'Max :', np.max(x_test))  # Min : -7.6732 Max : 586.3726
# Standard / Robust 는 0~1 로 수렴하지 않는다 -> 이 출력이 정상
```

### model.save() - 훈련 '전'에 저장하면 구조만 담긴다

```python
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(50))
model.add(Dense(1))

model.summary()          # Total params 6,001

path = './_save/keras29/'                       # 폴더까지만! 반드시 '/' 로 끝낸다
# model.save(path + 'keras29_1_save_model.h5')    # 예전 방식
model.save(path + 'keras29_1_save_model.keras')   # 최근 방식

exit()                   # 훈련은 하지 않고 여기서 종료
```

### path 슬래시 함정

```python
# path = './_save/keras29'                      # ❌ 슬래시 누락
# -> path + 'keras29_1_save_model.keras'
# -> './_save/keras29keras29_1_save_model.keras'  (폴더 밖에 엉뚱한 이름으로 저장)
# -> 만들어둔 _save/keras29/ 폴더는 빈 채로 남아서 눈치채기 어렵다

path = './_save/keras29/'                       # ✅
# -> './_save/keras29/keras29_1_save_model.keras'
```

### load_model() - 구조만 불러왔으면 compile + fit 을 다시 한다

```python
from tensorflow.keras.models import Sequential, load_model

# #2. 모델 구성  <- 직접 쌓지 않고 파일에서 가져오므로 통째로 주석 처리
# model = Sequential()
# model.add(Dense(100, input_dim=8))

model = load_model(path + 'keras29_1_save_model.keras')
model.summary()          # 저장할 때와 같은 Total params 6,001

model.compile(loss='mse', optimizer='adam')   # 구조만 있으므로 컴파일부터 다시
model.fit(x_train, y_train, epochs=10000, batch_size=32,
          validation_split=0.2, callbacks=[es])
```

### 훈련 '뒤'에 저장하면 가중치까지 담긴다

```python
hist = model.fit(x_train, y_train, epochs=10000, batch_size=32,
                 validation_split=0.2, callbacks=[es])

# keras29_1 과 달리 save 가 fit '아래'에 있다
model.save(path + 'keras29_3_save_model.keras')
# -> 구조 + 훈련된 가중치 + 옵티마이저 상태까지 저장 (6,001 -> 18,005)
# -> EarlyStopping(restore_best_weights=True) 로 되돌린 '최적 가중치'가 저장된다
```

### 불러와서 바로 평가 - compile 이 필요 없다

```python
model = load_model(path + 'keras29_3_save_model.keras')
model.summary()
# Total params      : 18,005
# Trainable params  :  6,001   <- 가중치
# Optimizer params  : 12,004   <- 옵티마이저 상태까지 저장돼 있다

#3. 컴파일, 훈련  <- 훈련이 끝난 모델이라 통째로 주석 처리
# model.compile(loss='mse', optimizer='adam')
# model.fit(...)

loss = model.evaluate(x_test, y_test)   # compile 을 안 했는데도 평가가 된다
print("loss :", loss)                   # 저장 파일에 컴파일 정보까지 들어있기 때문
```

### 저장 -> 불러오기가 제대로 됐는지 확인하는 방법

```python
# keras29_3 (훈련 후 저장)   loss : 1.7271162271499634
# keras29_4 (불러와서 평가)  loss : 1.7271162271499634   <- 소수점 끝까지 같아야 정상

# 같은 값이 나오는 이유
#   evaluate 는 학습이 아니라 계산이라 무작위 요소가 없다
#   + random_state=100 으로 분할이 고정
#   + 스케일러 변환도 고정
#   + 가중치는 파일에 저장된 값 그대로
#   => 입력이 같으면 출력도 항상 같다

# 주의: 저장 파일은 마지막 실행 것으로 덮어써진다
#       keras29_3 을 돌린 '직후'에 keras29_4 를 돌려야 짝이 맞는다
#       (keras29_3 은 seed 미고정이라 돌릴 때마다 1.1605 / 1.7271 처럼 값이 달라진다)
```

---

## 💡 주요 학습 포인트

1. **스케일러 4종의 공식**: MinMax `(x-Min)/(Max-Min)` / Standard `(x-평균)/표준편차` / MaxAbs `x/|최댓값|` / Robust `(x-중앙값)/IQR`
2. **0~1로 수렴하는 건 MinMax·MaxAbs뿐**: Standard / Robust는 Min이 음수, Max가 1을 훌쩍 넘어도 정상이다
3. **정답 스케일러는 없다**: covtype·따릉이는 Robust, cancer·bike는 Standard, california·boston은 MinMax가 좋았다
4. **1회 실행으로 단정 금지**: 같은 코드 재실행에서 california가 0.6695 -> 1.0736. seed 미고정 편차가 스케일러 간 차이보다 컸다
5. **소요 시간 변화는 EarlyStopping 탓**: 스케일링이 연산을 빠르게/느리게 만드는 게 아니라 `patience`가 걸리는 시점이 달라진 것
6. **저장 시점이 저장 내용을 결정한다**: `fit` 전 저장 = 구조만(6,001) / `fit` 후 저장 = 가중치 + 옵티마이저(18,005)
7. **`compile` 없이 `evaluate`가 되는 이유**: `.keras` 파일에 컴파일 정보(loss, optimizer)까지 저장되기 때문
8. **`path`는 `/`로 끝낸다**: 빼면 `_save/keras29keras29_1_save_model.keras`처럼 파일명이 이어붙어 버린다
9. **불러온 모델은 같은 값을 재현한다**: `evaluate`에는 무작위 요소가 없다. 값이 다르면 저장/불러오기의 짝이 어긋난 것
10. **주석도 코드와 함께 갱신한다**: 스케일러를 바꿨는데 Min/Max 주석이 예전 값이면 다음에 볼 때 원인 파악을 방해한다

---

[⬅️ Day09](Day09.md) · [🏠 전체 목차](../README.md) · [Day11 ➡️](Day11.md)
