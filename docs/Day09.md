# Day09 - 데이터 스케일링(MinMaxScaler)과 모델 구조 확인

**학습 기간:** 2026-09-10

> 컬럼마다 값 범위가 달라 학습이 안 되던 문제를 **MinMaxScaler** 로 해결하고, 10개 데이터셋에 일괄 적용했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| model.summary() | 파라미터 수 = `(입력 × 출력) + 출력(bias)` |
| input_shape | `input_dim=4` = `input_shape=(4,)`. 맨 앞 데이터 개수(n)는 빼고 적는다 |
| MinMaxScaler | `(원값 - Min) / (Max - Min)` -> 0 ~ 1 로 수렴 |
| fit vs transform | `fit` 은 Min/Max 를 **찾아 저장**만, `transform` 이 실제로 값을 바꾼다 |
| 철칙 | `fit` 은 **x_train 에만**. test/val/제출용은 `transform` 만 (데이터 누수 방지) |
| x_test 가 0~1 을 넘는 것 | 정상! train 에 없던 값이 test 에 있다는 뜻 |
| 적용 대상 | x_train / x_test / x_val / 제출용 test_csv **전부** |
| A/B 테스트 | 한 번에 변수 하나만 바꾼다 |

---

## 📖 핵심 학습 내용

### model.summary() - 파라미터 개수 세기
- 파라미터 = `(입력 노드 수 × 출력 노드 수) + 출력 노드 수(bias)` - **bias까지 포함**해서 센다
- 층마다 노드 하나당 bias가 1개씩 붙기 때문에 마지막에 출력 노드 수만큼 더해진다

### input_dim -> input_shape
- `input_dim=4` 는 2차원 데이터에서만 쓸 수 있고, `input_shape=(4,)` 는 다차원 입력까지 표현 가능
- `(n, 4)` -> `(4,)` / `(n, 100, 3)` -> `(100, 3)` - **맨 앞의 데이터 개수(n)는 빼고** 적는다

### 이진 분류를 다중 분류 방식으로 풀기 (Santander)
- 출력층 1개 + `sigmoid` + `binary_crossentropy` -> 출력층 2개 + `softmax` + `categorical_crossentropy`
- `to_categorical(y, num_classes=num_classes)` 로 클래스 개수를 명시
- y가 이미 One-Hot이면 `stratify=y` 가 아니라 **`stratify=np.argmax(y, axis=1)`** 로 클래스 번호를 넘긴다

### 데이터 스케일링(Scaling)
- 컬럼마다 값의 범위가 다르면 **값이 큰 컬럼 쪽으로만 gradient가 크게 튀어서** loss 그래프가 톱니 모양이 된다
- **MinMaxScaler** : `(원값 - Min) / (Max - Min)` -> 모든 값을 0 ~ 1 로 수렴
- Day06에서 loss 곡선이 우하향하지 않던 원인이 바로 이것이었다

### fit 과 transform 은 하는 일이 다르다
| 함수 | 하는 일 |
|---|---|
| `scaler.fit(x_train)` | 컬럼별 Min/Max 를 **찾아서 scaler 안에 저장**한다. 데이터는 아직 안 바뀐다 |
| `scaler.transform(x_train)` | 저장해둔 Min/Max 를 공식에 대입해 **실제로 값을 0~1로 바꾼다** |
| `scaler.transform(x_test)` | Min/Max 를 **새로 구하지 않고** x_train 의 값을 그대로 대입한다 |

### 스케일링의 철칙
- **`fit` 은 x_train 에만, 나머지는 `transform` 만**
  - x_test로 `fit` 하면 평가 데이터의 Min/Max 가 변환 기준에 반영된다 -> **데이터 누수(Data Leakage)**
- 그래서 **x_test 값이 0~1 을 벗어나는 것은 정상** - train 에 없던 값이 test 에 있다는 뜻이고, fit을 제대로 했다는 증거
- 적용 대상은 **모델에 입력으로 들어가는 모든 데이터** : `x_train` / `x_test` / `x_val` / 제출용 `test_csv`
  - 하나라도 빠뜨리면 모델은 0~1로 학습했는데 입력만 원본 단위로 들어가 결과가 통째로 망가진다

### 스케일링 누락 3대 버그 (직접 찾아서 수정)
1. **`x_val` 누락** -> val_loss가 엉터리 -> EarlyStopping이 엉뚱한 시점에 멈춤 -> `restore_best_weights` 가 나쁜 가중치를 복원
2. **제출용 `test_csv` 누락** -> x_test로 잰 acc는 맞지만 제출 파일만 엉터리 예측
3. **`model.evaluate(x, y)`** -> 스케일링 안 된 원본을 넣어 loss 89923 (게다가 x에는 train이 포함되어 평가로도 부적합)

### 실험을 제대로 하는 법
- **A/B 테스트** - 성능 비교는 한 번에 변수 하나만 바꾼다
  - 베이스 파일을 복사한 뒤 스케일러 블록만 추가하고 모델 구조 / `random_state` / epochs / batch_size는 그대로
  - 결과 주석을 `==========` 구분선으로 나눠 before / after를 같은 파일에 기록
- **효과가 없는 것도 결과다**
  - 잘 듣는 데이터 : 컬럼 간 값 범위 차이가 큰 경우 (covtype, california, boston)
  - 안 듣는 데이터 : 이미 컬럼 단위가 고른 경우 (diabetes, digits, santander)
- **표본이 작으면 단정하지 않는다** - wine은 test가 54개라 0.963 -> 0.981 이 **맞힌 개수 1개 차이**
  - seed를 고정하지 않으면 가중치 초기화가 매번 달라지므로 1회 실행 결과로 결론을 내리면 안 된다
- **소요 시간 변화는 EarlyStopping 때문** - 연산이 빨라진 게 아니라 `patience` 가 걸리는 시점이 달라진 것
  - wine 16초 -> 116초 (val_loss가 더 오래 개선됨) / covtype 1463초 -> 1110초 (더 빨리 수렴)

### 제출 파일 관리
- 실습마다 파일명을 다르게 준다 : `submit_날짜_시간_실습이름.csv`
- 파일명을 재사용하면 이전 실습의 제출 결과가 덮어써져 캐글 점수를 비교할 수 없다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras24_kaggle_santander_categorical.py` | Santander 이진 분류를 **softmax 다중 분류 방식**으로 재구성 (200,000×200). `to_categorical(y, num_classes=2)`, `stratify=np.argmax(y, axis=1)`, 500-250-125-60-30-2 구조, patience=20, batch_size=32 (loss: 0.2437, acc: 0.9113, acc_score: 0.9113). `[수정]` keras22(sigmoid)가 만든 제출 파일을 덮어쓰던 파일명을 실습별 고유 이름으로 변경, `sample_submission.csv` 중복 로드 제거 |
| `keras25_summary.py` | `model.summary()`로 파라미터 개수 확인. 3-4-3-1 구조로 층별 param이 6 / 16 / 15 / 4 (총 41개)가 나오는 과정을 bias 포함해서 계산 |
| `keras26_input_shape.py` | Iris 다중 분류에서 `input_dim=4`를 `input_shape=(4,)`로 교체 (keras23_softmax1 베이스). 30-20-10-3 구조 (loss: 0.0730, acc: 0.978, accuracy_score: 0.9778, 8초) |
| `keras27_Scaler01_california.py` | 스케일링 입문. `MinMaxScaler` 알고리즘 `(원값 - Min) / (Max - Min)` 확인, 은닉층에 `relu` 추가. 우하향하지 않던 loss 그래프의 원인이 컬럼별 스케일 차이였음을 확인하고 `model.evaluate(x, y)`의 문제점 정리. 여기서는 split 전에 전체 `x`로 `fit` 했는데, 이것이 데이터 누수임을 확인하고 keras28에서 `fit(x_train)`으로 교정 |
| `keras28_Scaler01_california.py` | California 회귀 (keras20 베이스) - **loss 0.635 -> 0.504 (약 20% 개선)**. `[수정]` 스케일링 안 된 `model.evaluate(x, y)` 제거 |
| `keras28_Scaler02_diabetes.py` | Diabetes 회귀 (keras20 베이스) - loss 2874 -> 2971. sklearn이 이미 정규화해 배포하는 데이터라 효과 없음을 확인. `[수정]` 안 쓰는 `x_val` split이 x_train 크기를 결정하므로 지우면 안 되는 이유 주석 추가 |
| `keras28_Scaler03_boston.py` | Boston 회귀 (keras20 베이스) - **loss 22.8 -> 18.7 (약 18% 개선)**. CRIM/TAX/B 등 컬럼 범위가 제각각이라 효과 있음 |
| `keras28_Scaler04_dacon_ddareung.py` | Dacon 따릉이 회귀 (keras20 베이스) - `validation_data`용 `x_val`에 스케일링이 빠져 r2 -1.16 / RMSE 126으로 결과가 망가짐. `[수정]` `x_val = scaler.transform(x_val)` 추가 후 재실행하여 **loss 3192 -> 1995 (약 37% 개선), r2 0.7292, RMSE 44.67** 확보 |
| `keras28_Scaler05_kaggle_bike.py` | Kaggle Bike 회귀 (keras20 베이스) - loss 21894 -> 22418. 컬럼 범위 차이가 작아 변화 없음. `[수정]` 출력층 `relu`는 문제지만 A/B 비교를 위해 그대로 두는 이유 주석 추가 |
| `keras28_Scaler06_cancer.py` | Breast Cancer 이진 분류 (keras21 베이스) - acc_score 0.9240 -> 0.9708. mean area(143 ~ 2501)와 mean smoothness(0.05 ~ 0.16)가 섞여 있어 효과 기대. 다만 test 171개라 8개 차이 |
| `keras28_Scaler07_santander.py` | Kaggle Santander 다중 분류 (keras24 베이스) - acc 0.9113 -> 0.9146. var_0~var_199가 이미 비슷한 크기라 변화 없음. `[수정]` 제출용 `test_csv` 스케일링 추가, 제출 파일명이 이전 파일을 덮어쓰던 문제 수정, `sample_submission.csv` 중복 로드 제거 |
| `keras28_Scaler08_wine.py` | Wine 다중 분류 (keras23 베이스) - accuracy_score 0.9630 -> 0.9815. test 54개라 1개 차이로 신뢰하기 어려움 |
| `keras28_Scaler09_fetch_covtype.py` | Covtype 다중 분류 (keras23 베이스) - **accuracy_score 0.8612 -> 0.9361, 목표 0.93 통과**. Elevation(1859~3858)과 0/1 컬럼이 섞여 있어 이번 실습에서 효과가 가장 확실했던 데이터 |
| `keras28_Scaler10_digits.py` | Digits 다중 분류 (keras23 베이스) - accuracy_score 0.9652 -> 0.9736, 소요 시간 34초로 동일. 모든 컬럼이 픽셀 밝기 0~16으로 단위가 같아 변화가 없는 대조군. `x_test` Max가 2.67로 1을 넘는 것이 정상인 이유 주석 추가 |
| `keras20_EarlyStopping1_california.py` | 스케일링 전 비교 기준이 되는 실행 결과 주석 추가 (수정) |
| `keras22_sigmoid_santander.py` | sigmoid 이진 분류 재실행 결과 주석 추가 - loss 0.2389, acc 0.9103, acc_score 0.9123 (수정) |
| `keras23_softmax1_OneHot_iris.py` | 주석 오타 `OneHot Encording` -> `OneHot Encoding` 수정 (수정) |
| `_data/kaggle_santander/submit/submit_0910_1042.csv` | Santander 제출 파일 (sigmoid 확률값 출력) |

---

## 성과
- 회귀 5개 + 이진 분류 1개 + 다중 분류 4개, **총 10개 데이터셋에 MinMaxScaler를 일괄 적용해 성능 변화를 측정**
- 스케일링만으로 covtype 정확도 0.861 -> 0.936을 달성해 목표치(0.93)를 통과
- california 20%, boston 18%, 따릉이 37% loss 개선을 확인
- 따릉이는 `x_val` 스케일링을 빠뜨렸을 때 r2 -1.16 -> 한 줄 수정 후 r2 0.729로, 누락 하나가 결과 전체를 망가뜨린다는 것을 수치로 확인
- 효과가 있는 데이터와 없는 데이터를 구분하고, **왜 효과가 없는지**(이미 컬럼 단위가 고름)를 데이터 특성으로 설명
- `fit`은 train에만 한다는 원칙을 적용하고, keras27(전체 x에 fit)과 keras28(train에만 fit)의 차이로 데이터 누수 개념을 이해
- 스케일링 누락 버그 3가지를 직접 찾아 수정하고 각 코드에 `[수정]` 표시와 원인 설명을 남김
- 성능 비교에서 표본 크기와 seed 미고정 때문에 생기는 신뢰도 한계를 인식 (wine 1개 차이, cancer 8개 차이)
- `model.summary()`로 파라미터 개수를 직접 계산하고 `input_shape` 표기법을 습득

---

## 💻 핵심 개념

### model.summary() - 파라미터 개수 세기

```python
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))
model.summary()

# 파라미터 = (입력 노드 × 출력 노드) + 출력 노드(bias)
# Dense(3, input_dim=1) : (1 × 3) + 3 =  6
# Dense(4)              : (3 × 4) + 4 = 16
# Dense(3)              : (4 × 3) + 3 = 15
# Dense(1)              : (3 × 1) + 1 =  4
#                              Total  = 41
# bias 는 노드마다 1개씩 붙으므로 출력 노드 수만큼 더해준다
```

### input_dim vs input_shape

```python
# model.add(Dense(30, input_dim=4, activation='relu'))     # 1차원 입력만 가능
model.add(Dense(30, input_shape=(4,), activation='relu'))  # 다차원까지 표현 가능
```

### input_shape 표기법 - 맨 앞 데이터 개수(n)는 빼고 적는다

```python
# (n, 4)             -> (4,)
# (n, 100, 3)        -> (100, 3)
# (n, 100, 100, 3)   -> (100, 100, 3)
# 맨 앞 데이터 개수(n)는 빼고 적는다
```

### 이진 분류를 softmax 로 풀기

```python
# sigmoid  : 출력 1개 / binary_crossentropy       / y 그대로
# softmax  : 출력 2개 / categorical_crossentropy  / y 를 One-Hot

num_classes = len(np.unique(y))                        # 2
y = to_categorical(y, num_classes=num_classes)         # (200000,) -> (200000, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=777,
    stratify=np.argmax(y, axis=1),   # y 가 One-Hot 이면 클래스 번호로 되돌려서 넘긴다
)
model.add(Dense(num_classes, activation='softmax'))
```

### MinMaxScaler - 왜 필요한가

```python
'''
california 는 컬럼마다 값의 범위가 완전히 다르다.
  MedInc(소득) 0~15 / AveRooms 몇 개 / Population(인구) 수천 / AveOccup 는 1000 넘는 이상치까지
값이 큰 컬럼 하나 때문에 gradient 가 그쪽으로만 크게 튀어서
loss 가 내려가다 갑자기 솟구치는 톱니 모양 그래프가 된다.

MinMaxScaler : (원값 - Min) / (Max - Min)  -> 0 ~ 1 로 수렴
'''
```

### 스케일링의 철칙 : fit 은 x_train 에만!

```python
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(...)   # ① 먼저 나눈다

scaler = MinMaxScaler()

scaler.fit(x_train)                    # ② x_train 의 컬럼별 Min/Max 를 찾아서 저장 (데이터는 아직 안 바뀜)
x_train = scaler.transform(x_train)    # ③ 저장된 Min/Max 를 공식에 대입해서 실제로 0~1 로 변환
x_test  = scaler.transform(x_test)     #    Min/Max 를 새로 구하지 않고 x_train 의 값을 그대로 대입
x_val   = scaler.transform(x_val)      #    validation_data 를 쓴다면 val 도 반드시!
test_csv_scaled = scaler.transform(test_csv)   # 제출용 데이터도 반드시!

# scaler.fit(x_test) 를 하면 x_test 의 Min/Max 가 변환 기준이 되어
# 아직 보면 안 되는 평가 데이터 정보가 미리 새어 들어간다 (데이터 누수)

print('Min :', np.min(x_train), 'Max :', np.max(x_train))  # Min : 0.0  Max : 1.0
print('Min :', np.min(x_test),  'Max :', np.max(x_test))   # Min : -0.0012  Max : 1.0
# x_test 가 0~1 을 벗어나는 것은 정상!
# train 에 없던 더 큰 값이 test 에 있다는 뜻 -> fit 을 train 에만 했다는 증거다
# (digits 는 가장자리 픽셀 때문에 Max 가 2.67 까지 나온다)
```

### 스케일링 누락 3대 버그 (① x_val / ② test_csv / ③ evaluate)

```python
# ① x_val 누락
#    모델은 0~1 로 학습 -> validation_data 에는 원본이 들어감
#    -> val_loss 가 엉터리 -> EarlyStopping 이 엉뚱하게 멈춤
#    -> restore_best_weights=True 가 '가짜 최저점'의 나쁜 가중치를 복원
#    결과: r2 -1.16 (평균값만 찍는 모델보다 못함)

# ② 제출용 test_csv 누락
#    x_test 로 잰 acc 는 맞지만 제출 파일만 엉터리 예측이 된다
y_submit = model.predict(test_csv)          # ❌
y_submit = model.predict(test_csv_scaled)   # ✅

# ③ evaluate 에 원본 x 를 넣기
loss = model.evaluate(x_test, y_test)   # ✅ 0.5035
# loss = model.evaluate(x, y)           # ❌ 89923 - 스케일 단위가 안 맞아서 터진 값
#                                       #    게다가 x 에는 x_train 이 들어있어 평가로도 부적합
```

### A/B 테스트 - 한 번에 변수 하나만

```python
# 베이스 파일을 복사 -> 스케일러 블록만 추가
# 모델 구조 / random_state / epochs / batch_size 는 절대 건드리지 않는다
# 같이 바꾸면 무엇 때문에 성능이 변했는지 알 수 없다
```

### 결과 기록 형식

```python
# loss : 0.63498455286026                     <- 적용 전
# ========== ========== ========== ==========  <- MinMaxScaler 적용 후
# loss : 0.503523051738739                    <- 적용 후
```


---

## 📊 실행 결과 - 스케일링 적용 전/후 (10개 데이터셋)

| 파일 | 데이터 | 적용 전 | 적용 후 | 판단 |
|---|---|---|---|---|
| Scaler01 | California | loss 0.635 | **0.504** | 개선 - 컬럼 범위 차이 큼 |
| Scaler02 | Diabetes | loss 2874 | 2971 | 변화 없음 - 이미 정규화된 데이터 |
| Scaler03 | Boston | loss 22.8 | **18.7** | 개선 - CRIM/TAX/B 범위 제각각 |
| Scaler04 | 따릉이 | loss 3192 | **1995** | 개선(약 37%) - x_val 수정 후 재측정, r2 0.729 |
| Scaler05 | Kaggle Bike | loss 21894 | 22418 | 변화 없음 - 범위 차이 작음 |
| Scaler06 | Cancer | acc 0.9240 | 0.9708 | 개선으로 보이나 test 171개 (8개 차이) |
| Scaler07 | Santander | acc 0.9113 | 0.9146 | 변화 없음 - 이미 익명화·정규화된 컬럼 |
| Scaler08 | Wine | acc 0.9630 | 0.9815 | test 54개, **1개 차이**라 판단 보류 |
| Scaler09 | Covtype | acc 0.8612 | **0.9361** | 효과 확실 - test 17만개, 목표 0.93 통과 |
| Scaler10 | Digits | acc 0.9652 | 0.9736 | 변화 없음 - 픽셀 0~16 단위 통일 (대조군) |

---

## 💡 주요 학습 포인트
1. **파라미터 개수**: `(입력 노드 × 출력 노드) + 출력 노드(bias)` - `model.summary()`로 층별 확인
2. **`input_shape=(4,)`**: `input_dim=4`와 같지만 다차원 입력까지 표현 가능, 맨 앞 데이터 개수(n)는 빼고 적는다
3. **이진 분류도 softmax로 가능**: 출력 2개 + `categorical_crossentropy` + `to_categorical` - 결과는 sigmoid와 비슷
4. **One-Hot된 y로 stratify**: `stratify=y`가 아니라 `stratify=np.argmax(y, axis=1)`
5. **MinMaxScaler 공식**: `(원값 - Min) / (Max - Min)` -> 0~1 수렴. 컬럼별 gradient 쏠림을 막는다
6. **`fit`과 `transform`의 차이**: `fit`은 Min/Max를 찾아 저장만 하고, `transform`이 그 값을 공식에 대입해 실제로 변환한다
7. **`fit`은 x_train에만**: x_test로 `fit`하면 평가 데이터의 Min/Max가 기준에 반영된다 (데이터 누수). test는 `transform`만
8. **x_test가 0~1을 벗어나는 건 정상**: train에 없던 값이 test에 있다는 뜻 - 오히려 제대로 했다는 증거
9. **스케일링은 입력 전부에 적용**: `x_train` / `x_test` / `x_val` / 제출용 `test_csv` - 하나만 빠져도 결과가 망가진다
10. **A/B 테스트는 변수 하나만**: 스케일러 외에 모델 구조·seed·epochs를 같이 바꾸면 원인을 알 수 없다
11. **효과가 없는 것도 결과다**: 이미 컬럼 단위가 고른 데이터(diabetes, digits, santander)는 스케일링해도 안 좋아지는 게 정상
12. **표본이 작으면 단정하지 말 것**: wine 0.963 -> 0.981은 맞힌 개수 1개 차이. seed 미고정이면 다시 돌려 확인해야 한다
13. **소요 시간 변화는 EarlyStopping 때문**: 스케일링이 느려서가 아니라 `patience`가 걸리는 시점이 달라진 것
14. **제출 파일명은 실습마다 다르게**: 파일명을 재사용하면 이전 실습의 제출 결과가 덮어써져 비교가 불가능해진다

---

[⬅️ Day08](Day08.md) · [🏠 전체 목차](../README.md) · [Day10 ➡️](Day10.md)
