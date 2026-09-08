# Day04 - 회귀 모델 평가 메트릭 및 실제 데이터셋 활용

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

### 💡 주요 학습 포인트
1. **R2 Score**: 모델이 데이터의 분산을 얼마나 설명하는지 (0 ~ 1)
2. **RMSE vs MSE**: RMSE는 MSE에 루트를 씌운 값 (원래 단위로 표현)
3. **평가 메트릭 활용**: R2는 설명력, RMSE는 오류 크기
4. **EDA의 중요성**: 데이터의 형태, 결측치, 특성을 먼저 파악
5. **Pandas 데이터 처리**: DataFrame으로 대용량 데이터 효율적으로 처리

---

[⬅️ Day03](Day03.md) · [🏠 전체 목차](../README.md) · [Day05 ➡️](Day05.md)
