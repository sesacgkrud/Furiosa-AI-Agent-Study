# Day06 - Validation(검증 데이터), 과적합 시각화 및 EarlyStopping

**학습 기간:** 2026-09-07

> 검증(validation) 데이터를 떼어내 **val_loss** 를 보고, loss 곡선을 그려 과적합 시점을 눈으로 확인했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| verbose | 훈련 로그 출력량 조절 (0=생략 / 1=기본 / 2=진행바 없음) |
| validation | train 중 일부를 떼어 매 epoch 성적을 재는 데이터. **반드시 x_train 에서** 떼어낸다 |
| [방법 1] validation_data | `train_test_split` 을 2번 써서 x_val 을 직접 만든다 |
| [방법 2] validation_split | `fit` 이 x_train 뒤에서 순서대로 떼어간다 (섞지 않음) |
| 과적합 판독 | loss ↓ / val_loss ↑ 로 벌어지는 지점부터 과적합 |
| EarlyStopping | val_loss 가 patience 만큼 안 좋아지면 중단 + 최적 가중치 복원 |
| 활성화 함수 | relu 없이 Dense 만 쌓으면 층을 늘려도 직선 모델 1개와 같다 |

---

## 📖 핵심 학습 내용

### verbose - 훈련 로그 출력 제어
- `0` 생략 / `1` 기본(진행바) / `2` 진행바 생략 / 그 외 epoch 번호만
- 출력하는 데도 시간이 들기 때문에 데이터가 크면 0이나 2로 줄인다

### Validation(검증) 데이터
- **train = 공부 / val = 모의고사 / test = 수능** - val은 매 epoch 성적을 재는 데이터
- 검증 데이터는 반드시 `x_train` 안에서 떼어낸다 (`x_test` 를 val로 쓰면 데이터 누수)
- val_loss는 가중치 갱신에 쓰이지 않고, 과적합을 미리 알아채는 기준이 된다

### 검증 데이터를 만드는 2가지 방법
- **[방법 1] `validation_data=(x_val, y_val)`** - `train_test_split()` 을 2번 써서 x_val을 직접 만든다
- **[방법 2] `validation_split=0.2`** - `fit()` 이 `x_train` 에서 알아서 떼어간다
- 방법 2의 함정
  - `x_train` 의 **맨 뒤에서부터 섞지 않고** 순서대로 자른다 (정렬된 데이터면 val이 한쪽으로 치우친다)
  - 비율 기준은 전체가 아니라 `x_train`
  - `validation_data` 와 같이 쓰면 `validation_data` 가 우선한다

### History로 과적합 판독
- `hist = model.fit(...)` 의 반환값에 epoch별 기록이 들어 있다 -> `hist.history['loss']`, `['val_loss']`
- Matplotlib으로 두 곡선을 같이 그린다
- `loss ↓ / val_loss ↓` = 정상 학습, **`loss ↓ / val_loss ↑` = 과적합 시작**

### EarlyStopping
- 과적합 지점에서 훈련을 자동으로 중단하는 콜백
- `monitor='val_loss'` / `mode='min'` / `patience` (참고 기다리는 횟수) / `restore_best_weights=True`
- `callbacks=[es]` 에 넣어야 실제로 동작한다

### 그 외
- Keras 내장 `boston_housing.load_data()` - 이미 train/test로 나뉘어 반환된다
- `time.time()` 으로 훈련 소요 시간 측정
- 활성화 함수 없이 Dense만 쌓으면 **선형 × 선형 = 선형** -> 층을 늘려도 직선 모델 1개와 같다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras15_verbose.py` | `verbose` 옵션별 훈련 로그 출력 차이 비교 |
| `keras16_validation1.py` | `x_train`/`x_val`/`x_test`를 수동으로 나누고 `validation_data` 전달 (6/2/2 분할) |
| `keras16_validation2.py` | 1~16 데이터를 Numpy 슬라이싱으로 8/4/4 분할 실습 |
| `keras16_validation3_train_test.py` | `train_test_split()`을 2번 사용해 train/val/test 분할 실습 |
| `keras16_validation4_split.py` | `validation_split=0.33`으로 검증 데이터 자동 분리 |
| `keras17_val1_california.py` | California Housing + `validation_split` (방법 2) |
| `keras17_val2_diabetes.py` | Diabetes + `train_test_split` 2회 + `validation_data` (방법 1) |
| `keras17_val3_boston.py` | Boston Housing(Keras 내장 `boston_housing.load_data()`) + `validation_split` (방법 2) |
| `keras17_val4_dacon_ddareung.py` | Dacon 따릉이 + `validation_data` (방법 1), r2/mse/RMSE 평가 |
| `keras17_val5_kaggle_bike.py` | Kaggle Bike Sharing + `validation_split` (방법 2), r2/mse/RMSE 평가 |
| `keras18_time.py` | `time.time()`으로 훈련 시작/종료 시간을 재서 소요 시간 출력 |
| `keras19_overfit1_california.py` | California Housing loss/val_loss 곡선 시각화 + `relu` 적용 (val_loss가 어느 시점부터 상승하는 것을 보고 과적합 구간 확인) |
| `keras19_overfit2_diabetes.py` | Diabetes loss/val_loss 곡선 시각화 (초반 급락 구간을 `[10:]`로 잘라내고 표시) |
| `keras19_overfit3_boston.py` | Boston Housing loss/val_loss 곡선 시각화 (`[10:]` 구간) |
| `keras19_overfit4_dacon_ddareung.py` | 따릉이 loss/val_loss 곡선 시각화 + r2/mse/RMSE 평가 |
| `keras19_overfit5_kaggle_bike.py` | Kaggle Bike loss/val_loss 곡선 시각화 + r2/mse/RMSE 평가 |
| `keras20_EarlyStopping1_california.py` | EarlyStopping 콜백 적용 (`patience=100`, `restore_best_weights=True`) |
| `keras14_kaggle_bike1.py` | 하이퍼파라미터 튜닝 기록 구분선 추가 (수정) |

---

## 성과
- 훈련/검증/평가 데이터의 역할을 명확히 구분하고 데이터 누수 없이 분할하는 방법 습득
- `validation_data`와 `validation_split` 두 방식을 5개 데이터셋(California, Diabetes, Boston, 따릉이, Kaggle Bike)에 모두 적용
- 훈련 과정을 그래프로 시각화하여 과적합 시점을 눈으로 확인하는 능력 확보
- EarlyStopping으로 최적 지점의 가중치를 자동으로 되돌리는 방법 습득
- 스케일링 적용 전/후 비교로 loss 곡선이 우하향하지 않던 원인 규명 (1 epoch loss 1,144.9 → 0.75)

---

## 💻 핵심 개념

```python
# verbose 옵션
model.fit(x_train, y_train, epochs=100, batch_size=4, verbose=0)
# verbose=0 : 훈련 과정 출력 생략 (자원 절약)
# verbose=1 : 기본값 (progress bar + 로그)
# verbose=2 : progress bar 생략
# 그 외      : epoch 번호만 표시
```

### [방법 1] train_test_split 2번 -> x_val 직접 생성

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(   # x_test가 아니라 x_train을 다시 자른다
    x_train, y_train, train_size=0.8, random_state=42)
model.fit(x_train, y_train, validation_data=(x_val, y_val))
```

### [방법 2] validation_split -> fit이 알아서 분리

```python
model.fit(x_train, y_train, validation_split=0.2)
# 주의 1) 비율 기준은 전체가 아니라 x_train
# 주의 2) x_train의 맨 뒤에서부터 "섞지 않고" 순서대로 자름
# 주의 3) validation_data와 같이 쓰면 validation_data가 우선
```

### Keras 내장 Boston 데이터 (이미 train/test 분리되어 있음)

```python
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()  # (404,13) (102,13)
```

### 훈련 소요 시간 측정

```python
import time
start_time = time.time()
model.fit(x_train, y_train, epochs=2, batch_size=16)
end_time = time.time()
print("소요 시간 :", round(end_time - start_time, 2), "초")
```

### History 객체로 loss 곡선 그리기

```python
hist = model.fit(x_train, y_train, epochs=500, validation_split=0.2)
print(hist.history)              # {'loss': [...], 'val_loss': [...]}
print(hist.history['loss'])      # epoch별 훈련 손실
print(hist.history['val_loss'])  # epoch별 검증 손실

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 깨짐 해결
plt.rcParams['axes.unicode_minus'] = False      # 음수 부호 깨짐 방지
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.xlabel('epochs')
plt.ylabel('loss')      # xlabel을 두 번 쓰면 x축 라벨이 덮어써진다
plt.grid()
plt.show()
```

### 과적합 판독

```python
# loss 감소 , val_loss 감소  -> 아직 잘 학습되는 중
# loss 감소 , val_loss 증가  -> 과적합 시작 (훈련할수록 오히려 나빠짐)
```

### EarlyStopping - 과적합 지점에서 자동 중단

```python
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',          # 무엇을 기준으로 볼 것인가
    mode='min',                  # val_loss는 작을수록 좋으므로 min
    patience=100,                # 개선이 없어도 참고 기다리는 횟수
    restore_best_weights=True,   # 지나쳤더라도 최솟값 시점의 가중치로 되돌림
)
model.fit(x_train, y_train, epochs=99999999999, batch_size=32,
          validation_split=0.2, callbacks=[es])

# 활성화 함수 없이 Dense만 쌓으면 선형 x 선형 = 선형 -> 직선 모델 1개와 동일
model.add(Dense(100, input_dim=8, activation='relu'))
```


---

## 💡 주요 학습 포인트
1. **verbose**: 훈련 로그 출력량 조절로 자원 절약 (0=생략, 1=기본, 2=progress bar 생략)
2. **train/val/test 역할**: train=공부, val=모의고사(가중치 갱신 X), test=수능(마지막 1회)
3. **데이터 누수 방지**: val은 반드시 x_train에서 떼어낸다 (x_test를 val로 쓰면 성능이 부풀려짐)
4. **validation_data vs validation_split**: 전자는 섞어서 자르고 직접 관리, 후자는 한 줄이지만 맨 뒤에서 순서대로 자름
5. **History 시각화**: `hist = model.fit(...)` -> `hist.history['loss']`, `['val_loss']`를 그래프로 확인
6. **과적합 판독**: 두 곡선이 벌어지는(val_loss가 상승하는) 지점이 훈련을 멈춰야 할 시점
7. **EarlyStopping**: patience만큼 기다렸다가 중단하고, restore_best_weights로 최적 가중치 복원
8. **활성화 함수의 필요성**: relu 없이 Dense만 쌓으면 층을 늘려도 선형 모델 1개와 같다
9. **훈련 시간 측정**: `time.time()`으로 시작/종료 시각을 재서 모델별 소요 시간 비교

---

[⬅️ Day05](Day05.md) · [🏠 전체 목차](../README.md) · [Day07 ➡️](Day07.md)
