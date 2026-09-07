# Furiosa AI Agent - 학습 진행 현황

## 📚 Day01 - Keras 기초 및 Deep Learning 구조
**학습 기간:** 2026-08-31

### 핵심 학습 내용
- TensorFlow/Keras Sequential 모델 기초
- Dense Layer를 이용한 신경망 구성
- MSE 손실함수 및 Adam 최적화 알고리즘
- Deep Learning 구조(다층 네트워크)로 모델 성능 향상
- Batch Processing을 통한 훈련 방식
- 모델 컴파일, 훈련, 평가, 예측의 4가지 단계

### 학습 파일
- `keras01.py`: 기본 신경망 구성 (1-1-1 구조)
- `keras02.py`: 선형 회귀 모델 (1-1-1 구조)
- `keras03.py`: epochs 조정을 통한 손실값 개선 (900 epochs)
- `keras04_deep1.py`: 4층 Deep Learning (500-300-1 구조)
- `keras05_deep2.py`: 3층 최적화 네트워크 (1000-300-1 구조)
- `keras06_batch.py`: Batch Processing 개념 이해

### 성과
- 기본 신경망 구성 완료
- 손실값 최적화 (0.3238)
- Deep Learning 개념 이해

---

## 📚 Day02 - 다중 입력/출력 신경망 및 데이터 분할
**학습 기간:** 2026-09-01

### 핵심 학습 내용
- Numpy 다차원 배열(Matrix, Tensor) 이해 및 shape 개념
- 다중 입력(Multi-Input) 신경망 구성
- 데이터 전치(Transpose) - `x.T` 및 `transpose()` 메서드
- 다중 출력(Multi-Output) 신경망 구성
- Train/Test 데이터 분할 (검증 개념 입문)
- MLP(Multi-Layer Perceptron) 고급 구조

### 학습 파일
- `keras07_matrix.py`: Numpy 다차원 배열 기초 (1D, 2D, 3D, 5D 데이터 구조)
- `keras08_mlp1_1.py`: 다중 입력 신경망 (2입력-1출력, shape: (5,2))
- `keras08_mlp1_2.py`: 데이터 전치 적용 (2입력-1출력, transpose 사용)
- `keras08_mlp2_1.py`: 3입력-1출력 MLP (10->5->2->1 구조, loss: 5.54e-05)
- `keras08_mlp2_2.py`: 3입력-2출력 MLP (20->10->5->2 구조)
- `keras08_mlp3_1.py`: 1입력-3출력 MLP (10->5->3 구조)
- `keras08_mlp4.py`: 완성된 MLP (정확도 높음)
- `keras09_train_test1.py`: Train/Test 분할 (검증 방식 학습)

### 성과
- 다중 입력/출력 신경망 완성
- 다차원 데이터 처리 능력 확보
- Train/Test 분할 개념 이해
- MLP 고급 구조 마스터

### 핵심 개념
```python
# 데이터 전치
x.T  # numpy array 전치
x.transpose()  # 메서드를 이용한 전치

# 다중 입력/출력
Dense(n_output, input_dim=n_input)  # n_input 개의 입력, n_output 개의 출력

# Train/Test 분할
x_train, x_test = x[:7], x[7:]
y_train, y_test = y[:7], y[7:]
model.fit(x_train, y_train, ...)  # 훈련
model.evaluate(x_test, y_test)    # 검증
```

---

## 📚 Day03 - 데이터 분할 고급 및 실제 데이터셋 활용
**학습 기간:** 2026-09-02

### 핵심 학습 내용
- Numpy 슬라이싱을 이용한 Train/Test 분할 (수동 방식)
- Scikit-learn의 `train_test_split()` 함수 활용
- `random_state` 파라미터를 통한 난수 재현성 확보
- `train_size`, `test_size` 비율 설정 (7:3, 8:2 등)
- `shuffle` 파라미터를 통한 데이터 섞기
- Matplotlib을 이용한 데이터 시각화 (scatter plot, line plot)
- 실제 머신러닝 데이터셋 활용 (California Housing, Diabetes, Boston Housing)
- 회귀 모델(Regression Model) 성능 평가

### 학습 파일
- `keras09_train_test2.py`: Numpy 슬라이싱을 이용한 7:3 분할
- `keras09_train_test3.py`: Scikit-learn train_test_split() 기본 활용
- `keras10_scatter1.py`: train_test_split() + matplotlib 산점도 그리기
- `keras10_scatter2.py`: 예측 결과를 그래프로 시각화
- `keras11_1_california.py`: California Housing 데이터셋 활용 (8개 특성, 20640 샘플)
- `keras11_2_diabetes.py`: Diabetes 데이터셋 활용 (10개 특성, 442 샘플)
- `keras11_3_boston_tf.py`: Boston Housing 데이터셋 활용 (13개 특성, 506 샘플)
- `keras12_R2_RMSE_boston.py`: Boston Housing으로 회귀 모델 평가

### 성과
- 실제 데이터셋으로 모델 훈련 완료
- 데이터 분할 방법 마스터 (Numpy vs Scikit-learn)
- 모델 예측 결과 시각화 능력 획득

### 핵심 개념
```python
# Numpy 슬라이싱
x_train, x_test = x[:7], x[7:]

# Scikit-learn train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,      # 70% 훈련 데이터
    test_size=0.3,       # 30% 테스트 데이터
    shuffle=True,        # 데이터 섞기
    random_state=1234    # 난수 고정 (재현성)
)

# Matplotlib 시각화
plt.scatter(x, y)              # 원본 데이터 점찍기
plt.plot(x, result, color='red')  # 예측 결과 선그래프
plt.show()
```

---

## 📚 Day04 - 회귀 모델 평가 메트릭 및 실제 데이터셋 활용
**학습 기간:** 2026-09-03

### 핵심 학습 내용
- R2 Score (결정계수, Coefficient of Determination) 이해 및 적용
- RMSE (Root Mean Squared Error, 평균제곱근오차) 계산 방법
- MSE (Mean Squared Error, 평균제곱오차)와 RMSE의 관계
- 회귀 모델 성능 평가 메트릭의 실전 활용
- Scikit-learn의 `r2_score()`, `mean_squared_error()` 함수 사용법
- 다양한 공개 데이터셋에 평가 메트릭 적용 (Boston, California, Diabetes)
- Pandas를 이용한 CSV 데이터 로드 및 기본 EDA (Exploratory Data Analysis)
- 데이터프레임(DataFrame) 형태 확인 및 결측치(Missing Value) 분석
- Dacon 공개 데이터셋 활용 (따릉이 자전거 대여 데이터)

### 학습 파일
- `keras12_R2_RMSE_01_boston.py`: Boston Housing 데이터셋으로 R2, RMSE, MSE 평가 (loss: 23.06, r2: 0.72, RMSE: 5.23)
- `keras12_R2_RMSE_02_california.py`: California Housing 데이터셋으로 R2 평가 (r2: 0.56, 기준 0.55 이상)
- `keras12_R2_RMSE_03_diabetes.py`: Diabetes 데이터셋으로 R2 평가 (r2: 0.46, 기준 0.62 이상)
- `keras13_ddareung.py`: Dacon 따릉이 자전거 대여 데이터셋 EDA 및 전처리 (1459 훈련 샘플, 715 테스트 샘플, 9-10개 특성, 결측치 분석)

### 성과
- 회귀 모델 성능 평가 메트릭 완전 이해
- 평가 메트릭을 이용한 모델 성능 비교 능력 습득
- 실제 데이터셋의 구조 분석 및 EDA 능력 향상
- Pandas 데이터프레임 처리 능력 습득
- 결측치 처리 전준비 (dropna() 함수 학습)

### 핵심 개념
```python
# R2 Score 계산
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

r2 = r2_score(y_test, y_predict)  # 1에 가까울수록 좋음 (0 ~ 1)

# RMSE 계산
rmse = np.sqrt(mean_squared_error(y_test, y_predict))

# MSE 계산
mse = mean_squared_error(y_test, y_predict)

# 평가 메트릭 비교
# R2 Score: 모델 설명력 (0.7 이상 좋음)
# RMSE: 예측 오류 크기 (작을수록 좋음)
# MSE: RMSE의 제곱 (손실함수로도 사용)

# Pandas CSV 로드 및 EDA
import pandas as pd

df = pd.read_csv('data.csv', index_col=0)
print(df.shape)    # (행, 열) 확인
print(df.info())   # 데이터타입 및 결측치 확인
print(df.columns)  # 컬럼명 확인

# 결측치 처리
df_clean = df.dropna()  # 결측치 있는 행 삭제
```

---

## 📚 Day05 - Kaggle 실제 데이터셋 및 대규모 회귀 모델 개발
**학습 기간:** 2026-09-04

### 핵심 학습 내용
- 대규모 실제 데이터셋 처리 (Kaggle Bike Sharing Demand: 10,886 샘플, 8개 특성)
- 결측치 처리 방법 비교 (dropna() vs fillna() - 평균값 대체)
- 제출 파일(submission.csv) 생성 및 형식 이해
- 모델 하이퍼파라미터 튜닝 (Dense 레이어 수, epochs, batch_size, activation)
- 회귀 모델 성능 최적화 (r2, MSE, RMSE 지표를 통한 반복 개선)
- 다중 회귀 분석 (multi-feature regression) 실전 경험

### 학습 파일
- `keras13_ddareung01.py`: Dacon 따릉이 데이터 활용 (dropna()로 결측치 삭제, r2: 0.7157, RMSE: 46.1)
- `keras13_ddareung02_submit.py`: 따릉이 데이터 + 결측치 대체 (fillna() 평균값, 제출 파일 생성, r2: 0.6854, RMSE: 43.05)
- `keras14_kaggle_bike1.py`: Kaggle 자전거 공유 데이터셋 (10,886 훈련 샘플, 6,493 테스트 샘플, r2: 0.3211, RMSE: 147.91)

### 성과
- 대규모 실제 데이터셋으로 회귀 모델 구축 완료
- 결측치 처리 방법 실전 적용
- Kaggle 형식의 submission.csv 생성 및 제출 프로세스 이해
- 모델 성능 튜닝을 통한 반복적 개선 능력 습득

### 핵심 개념
```python
# 결측치 처리 방법 비교
train_df = train_df.dropna()                    # 방법1: 결측치 있는 행 삭제
test_df = test_df.fillna(test_df.mean())        # 방법2: 평균값으로 대체

# 제출 파일 생성
submission['count'] = model.predict(test_df)
submission.to_csv(path + "submit/" + "submit.csv")

# 하이퍼파라미터 튜닝 반복
# Dense 레이어 수: 2~4개 조정
# epochs: 1000~2000 조정
# batch_size: 16~32 조정
# activation: relu vs sigmoid vs none 비교
```

---

## 📚 Day06 - Validation(검증 데이터), 과적합 시각화 및 EarlyStopping
**학습 기간:** 2026-09-07

### 핵심 학습 내용
- `verbose` 옵션으로 훈련 로그 출력 제어 (0: 생략, 1: 기본, 2: progress bar 생략, 그 외: epoch만 표시)
- Validation(검증) 데이터의 개념과 역할 이해 (train=공부 / val=모의고사 / test=수능)
- 검증 데이터는 반드시 `x_train` 안에서 떼어내야 함 (`x_test`를 val로 쓰면 데이터 누수)
- **[방법 1]** `train_test_split()`을 2번 사용해 `x_val`을 직접 만들고 `validation_data=(x_val, y_val)`로 전달
- **[방법 2]** `validation_split=0.2` 한 줄로 `fit()`이 `x_train`에서 알아서 검증 데이터를 분리
- 두 방법의 차이와 함정: `validation_split`은 `x_train`의 **맨 뒤에서부터 섞지 않고** 순서대로 자름 (정렬된 데이터면 val이 한쪽으로 치우침)
- `validation_data`와 `validation_split`을 함께 쓰면 `validation_data`가 우선 적용됨
- `validation_split`의 비율 기준은 전체 데이터가 아니라 `x_train`
- Keras 내장 데이터셋 `boston_housing.load_data()` 활용 (이미 train/test로 분리되어 반환)
- `time` 모듈을 이용한 훈련 소요 시간 측정
- `hist = model.fit(...)`의 반환값(History 객체)과 `hist.history` 딕셔너리 구조 이해
- `hist.history['loss']`, `hist.history['val_loss']`를 Matplotlib으로 시각화하여 과적합(Overfitting) 판독
- 과적합 판독 기준: `loss ↓ / val_loss ↓` = 정상 학습, `loss ↓ / val_loss ↑` = 과적합 시작
- **EarlyStopping** 콜백으로 과적합 지점에서 훈련 자동 중단 (`monitor`, `mode`, `patience`, `restore_best_weights`)
- 데이터 스케일링의 필요성 확인: `StandardScaler` 적용 전/후 loss 곡선 비교
- 활성화 함수 없이 Dense를 여러 층 쌓으면 결국 선형 모델 1개와 동일함을 이해

### 학습 파일
- `keras15_verbose.py`: `verbose` 옵션별 훈련 로그 출력 차이 비교
- `keras16_validation1.py`: `x_train`/`x_val`/`x_test`를 수동으로 나누고 `validation_data` 전달 (6/2/2 분할)
- `keras16_validation2.py`: 1~16 데이터를 Numpy 슬라이싱으로 8/4/4 분할 실습
- `keras16_validation3_train_test.py`: `train_test_split()`을 2번 사용해 train/val/test 분할 실습
- `keras16_validation4_split.py`: `validation_split=0.33`으로 검증 데이터 자동 분리
- `keras17_val1_california.py`: California Housing + `validation_split` (방법 2)
- `keras17_val2_diabetes.py`: Diabetes + `train_test_split` 2회 + `validation_data` (방법 1)
- `keras17_val3_boston.py`: Boston Housing(Keras 내장 `boston_housing.load_data()`) + `validation_split` (방법 2)
- `keras17_val4_dacon_ddareung.py`: Dacon 따릉이 + `validation_data` (방법 1), r2/mse/RMSE 평가
- `keras17_val5_kaggle_bike.py`: Kaggle Bike Sharing + `validation_split` (방법 2), r2/mse/RMSE 평가
- `keras18_time.py`: `time.time()`으로 훈련 시작/종료 시간을 재서 소요 시간 출력
- `keras19_overfit1_california.py`: California Housing loss/val_loss 곡선 시각화 + `StandardScaler`/`relu` 적용 (train loss 0.75 → 0.025, val_loss 최저 0.281@49 epoch 이후 상승 = 과적합)
- `keras19_overfit2_diabetes.py`: Diabetes loss/val_loss 곡선 시각화 (초반 급락 구간을 `[10:]`로 잘라내고 표시)
- `keras19_overfit3_boston.py`: Boston Housing loss/val_loss 곡선 시각화 (`[10:]` 구간)
- `keras19_overfit4_dacon_ddareung.py`: 따릉이 loss/val_loss 곡선 시각화 + r2/mse/RMSE 평가
- `keras19_overfit5_kaggle_bike.py`: Kaggle Bike loss/val_loss 곡선 시각화 + r2/mse/RMSE 평가
- `keras20_EarlyStopping1_california.py`: EarlyStopping 콜백 적용 (`patience=100`, `restore_best_weights=True`)
- `keras14_kaggle_bike1.py`: 하이퍼파라미터 튜닝 기록 구분선 추가 (수정)

### 성과
- 훈련/검증/평가 데이터의 역할을 명확히 구분하고 데이터 누수 없이 분할하는 방법 습득
- `validation_data`와 `validation_split` 두 방식을 5개 데이터셋(California, Diabetes, Boston, 따릉이, Kaggle Bike)에 모두 적용
- 훈련 과정을 그래프로 시각화하여 과적합 시점을 눈으로 확인하는 능력 확보
- EarlyStopping으로 최적 지점의 가중치를 자동으로 되돌리는 방법 습득
- 스케일링 적용 전/후 비교로 loss 곡선이 우하향하지 않던 원인 규명 (1 epoch loss 1,144.9 → 0.75)

### 핵심 개념
```python
# verbose 옵션
model.fit(x_train, y_train, epochs=100, batch_size=4, verbose=0)
# verbose=0 : 훈련 과정 출력 생략 (자원 절약)
# verbose=1 : 기본값 (progress bar + 로그)
# verbose=2 : progress bar 생략
# 그 외      : epoch 번호만 표시

# [방법 1] train_test_split 2번 -> x_val 직접 생성
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(   # x_test가 아니라 x_train을 다시 자른다
    x_train, y_train, train_size=0.8, random_state=42)
model.fit(x_train, y_train, validation_data=(x_val, y_val))

# [방법 2] validation_split -> fit이 알아서 분리
model.fit(x_train, y_train, validation_split=0.2)
# 주의 1) 비율 기준은 전체가 아니라 x_train
# 주의 2) x_train의 맨 뒤에서부터 "섞지 않고" 순서대로 자름
# 주의 3) validation_data와 같이 쓰면 validation_data가 우선

# Keras 내장 Boston 데이터 (이미 train/test 분리되어 있음)
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()  # (404,13) (102,13)

# 훈련 소요 시간 측정
import time
start_time = time.time()
model.fit(x_train, y_train, epochs=2, batch_size=16)
end_time = time.time()
print("소요 시간 :", round(end_time - start_time, 2), "초")

# History 객체로 loss 곡선 그리기
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

# 과적합 판독
# loss 감소 , val_loss 감소  -> 아직 잘 학습되는 중
# loss 감소 , val_loss 증가  -> 과적합 시작 (훈련할수록 오히려 나빠짐)

# EarlyStopping - 과적합 지점에서 자동 중단
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',          # 무엇을 기준으로 볼 것인가
    mode='min',                  # val_loss는 작을수록 좋으므로 min
    patience=100,                # 개선이 없어도 참고 기다리는 횟수
    restore_best_weights=True,   # 지나쳤더라도 최솟값 시점의 가중치로 되돌림
)
model.fit(x_train, y_train, epochs=99999999999, batch_size=32,
          validation_split=0.2, callbacks=[es])

# 스케일링 - loss 곡선이 우하향하지 않을 때
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)   # fit은 train으로만! (test 정보를 미리 보면 안 됨)
x_test  = scaler.transform(x_test)        # test는 train의 기준으로 변환만
# 활성화 함수 없이 Dense만 쌓으면 선형 x 선형 = 선형 -> 직선 모델 1개와 동일
model.add(Dense(100, input_dim=8, activation='relu'))
```

---

## 🎯 학습 진행도

| 단계 | 학습 내용 | 완료 | 주요 내용 |
|------|---------|------|---------|
| Day01 | Keras 기초 | ✅ | Sequential, Dense, Batch Processing, Deep Learning |
| Day02 | 다중 입력/출력 | ✅ | Matrix, Transpose, Multi-Input/Output, Train/Test 분할 |
| Day03 | 데이터 분할 고급 & 실제 데이터셋 | ✅ | train_test_split, Matplotlib 시각화, 공개 데이터셋 활용 |
| Day04 | 회귀 모델 평가 메트릭 & EDA | ✅ | R2 Score, RMSE, MSE, Pandas EDA, 결측치 분석 |
| Day05 | Kaggle 실제 데이터셋 & 제출 | ✅ | 대규모 데이터 처리, 결측치 처리 방법, 하이퍼파라미터 튜닝 |
| Day06 | Validation & 과적합 시각화 | ✅ | verbose, validation_data/split, History 시각화, EarlyStopping |

---

## 📁 디렉토리 구조

```
C:\furiosa_study\
├── keras/
│   ├── keras01.py ~ keras06_batch.py                           (Day01)
│   ├── keras07_matrix.py ~ keras09_train_test1.py              (Day02)
│   ├── keras09_train_test2.py ~ keras12_R2_RMSE_boston.py      (Day03)
│   ├── keras12_R2_RMSE_01_boston.py ~ keras13_ddareung.py      (Day04)
│   ├── keras13_ddareung01.py ~ keras14_kaggle_bike1.py         (Day05)
│   ├── keras15_verbose.py ~ keras20_EarlyStopping1_...py       (Day06)
│   └── ...
├── _data/
│   ├── ddareung/         (Dacon 따릉이 데이터)
│   └── kaggle_bike/      (Kaggle Bike Sharing 데이터)
├── README.md
└── .vscode/
```

---

## 💡 주요 학습 포인트

### Day01
1. **모델 구성**: Sequential() → add(Dense()) → compile() → fit()
2. **손실함수**: MSE (Mean Squared Error)
3. **최적화**: Adam optimizer
4. **성능 개선**: 레이어 깊이 증가, epochs 조정, 뉴런 수 증가

### Day02
1. **데이터 형태**: (sample, features) 형태 유지
2. **Transpose**: 데이터를 올바른 형태로 변환
3. **다중 출력**: 최종 Dense layer 출력값 = 원하는 출력 개수
4. **Train/Test**: 전체 데이터를 훈련/검증 데이터로 분할

### Day03
1. **데이터 분할**: Numpy 슬라이싱 vs train_test_split() 비교
2. **random_state**: 난수 고정으로 재현성 확보 (매번 동일한 분할 결과)
3. **shuffle=True**: 데이터를 무작위로 섞어서 공정한 훈련/테스트 분할
4. **Matplotlib 시각화**: scatter plot으로 원본 데이터, plot으로 예측값 표시
5. **공개 데이터셋**: sklearn의 California Housing, Diabetes, Boston Housing 등 활용

### Day04
1. **R2 Score**: 모델이 데이터의 분산을 얼마나 설명하는지 (0 ~ 1)
2. **RMSE vs MSE**: RMSE는 MSE에 루트를 씌운 값 (원래 단위로 표현)
3. **평가 메트릭 활용**: R2는 설명력, RMSE는 오류 크기
4. **EDA의 중요성**: 데이터의 형태, 결측치, 특성을 먼저 파악
5. **Pandas 데이터 처리**: DataFrame으로 대용량 데이터 효율적으로 처리

### Day05
1. **결측치 처리 방법 비교**: dropna()는 행 제거, fillna()는 평균값 대체 선택
2. **대규모 데이터셋 처리**: Kaggle 형식 데이터셋 (10,886 훈련, 6,493 테스트 샘플)
3. **제출 파일 생성**: submission.csv 형식 이해 및 생성 방법
4. **하이퍼파라미터 튜닝**: Dense 레이어 수, epochs, batch_size, activation 반복 조정
5. **모델 성능 최적화**: R2, RMSE를 지표로 반복 개선 (트레이드오프 이해)
6. **다중 회귀 분석 실전**: 8개 특성을 입력으로 하는 실제 예측 모델 구축

### Day06
1. **verbose**: 훈련 로그 출력량 조절로 자원 절약 (0=생략, 1=기본, 2=progress bar 생략)
2. **train/val/test 역할**: train=공부, val=모의고사(가중치 갱신 X), test=수능(마지막 1회)
3. **데이터 누수 방지**: val은 반드시 x_train에서 떼어낸다 (x_test를 val로 쓰면 성능이 부풀려짐)
4. **validation_data vs validation_split**: 전자는 섞어서 자르고 직접 관리, 후자는 한 줄이지만 맨 뒤에서 순서대로 자름
5. **History 시각화**: `hist = model.fit(...)` -> `hist.history['loss']`, `['val_loss']`를 그래프로 확인
6. **과적합 판독**: 두 곡선이 벌어지는(val_loss가 상승하는) 지점이 훈련을 멈춰야 할 시점
7. **EarlyStopping**: patience만큼 기다렸다가 중단하고, restore_best_weights로 최적 가중치 복원
8. **스케일링의 효과**: 컬럼별 값 범위가 다르면 gradient가 폭주 -> StandardScaler로 loss 곡선이 매끄러워짐
9. **활성화 함수의 필요성**: relu 없이 Dense만 쌓으면 층을 늘려도 선형 모델 1개와 같다
10. **훈련 시간 측정**: `time.time()`으로 시작/종료 시각을 재서 모델별 소요 시간 비교

---

## 📊 학습 진행도 (완료율)

**기간:** 2026-08-31 ~ 2026-12-24 (토요일, 일요일, 법정 공휴일 제외: 추석 3일, 한글날 1일, 대체공휴일 1일)  
**총 수업일:** 80일 (640시간)  
**완료:** 6일 (Day01 ~ Day06)  
**완료 시간:** 48시간  
**완료율: 7.5%**

---

**마지막 업데이트:** 2026-09-07  
**총 학습 시간:** Day01, Day02, Day03, Day04, Day05, Day06 완료
