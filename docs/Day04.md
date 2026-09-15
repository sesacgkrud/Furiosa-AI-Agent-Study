# Day04 - 회귀 모델 평가 메트릭 및 실제 데이터셋 활용

**학습 기간:** 2026-09-03

> loss 만으로는 알 수 없는 성능을 **r2 / mse / RMSE** 로 재는 법을 배우고, pandas로 CSV 데이터를 처음 다뤘다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| r2 (결정계수) | **1에 가까울수록 좋다.** 데이터가 달라도 서로 비교할 수 있다 |
| mse | 오차를 제곱해서 평균 낸 값. `loss='mse'` 로 훈련했다면 evaluate의 loss와 같다 |
| RMSE | mse에 루트를 씌운 값 -> **정답과 단위가 같아져서** "평균 몇 만큼 틀렸는지"로 읽힌다 |
| pandas 로드 | `pd.read_csv(path, index_col=0)` - 첫 컬럼을 인덱스로 뺀다 |
| EDA | `.shape` / `.info()` / `.columns` / `.isna().sum()` 로 먼저 데이터를 본다 |
| 결측치 | `.dropna()` 로 빈 칸이 있는 행을 지운다 |

---

## 📖 핵심 학습 내용

### 회귀 평가 메트릭 3종
- **r2 (결정계수)** - 모델이 데이터의 분산을 얼마나 설명하는지. 1에 가까울수록 좋다
  - loss(mse)는 데이터마다 단위가 달라 서로 비교할 수 없지만, r2는 비교가 된다
- **mse** - 오차를 제곱해서 평균 낸 값. `loss='mse'` 로 훈련했다면 `evaluate` 의 loss와 같은 값이 나온다
- **RMSE** - mse에 루트를 씌운 값. 정답과 단위가 같아져서 "평균 몇 만큼 틀렸는지"로 읽을 수 있다
- 세 지표를 같이 보면 loss 숫자만으로는 안 보이던 성능이 드러난다 (diabetes는 loss 2900인데 r2 0.46)

### pandas로 CSV 데이터 다루기
- `pd.read_csv(path, index_col=0)` - `index_col=0` 으로 id 컬럼을 인덱스로 빼서 컬럼 수를 맞춘다
- `.shape` (행, 열) / `.info()` (자료형 + 결측치) / `.columns` (컬럼명) / `.describe()` (통계)
- `.isna().sum()` 또는 `.isnull().sum()` - 컬럼별 결측치 개수

### 결측치(Missing Value)
- 빈 칸이 있으면 훈련이 되지 않으므로 먼저 처리해야 한다
- `.dropna()` - 빈 칸이 있는 행을 통째로 지운다 (따릉이 1459행 -> 1328행)

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras12_R2_RMSE_01_boston.py` | Boston Housing으로 r2 / mse / RMSE 전부 계산 (loss 23.06, r2 0.72, RMSE 5.23) - 한 줄씩 주석을 단 기준 파일 |
| `keras12_R2_RMSE_02_california.py` | California Housing r2 평가 (r2 0.56, 기준 0.55 이상) |
| `keras12_R2_RMSE_03_diabetes.py` | Diabetes r2 평가 (r2 0.46, 기준 0.62 이상) |
| `keras13_ddareung.py` | Dacon 따릉이 데이터 EDA 및 전처리 (1459 훈련 / 715 테스트, 9~10개 특성, 결측치 분석) |

---

## 💻 핵심 개념

### r2 / mse / RMSE

```python
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

r2 = r2_score(y_test, y_predict)                       # 1에 가까울수록 좋음
mse = mean_squared_error(y_test, y_predict)            # loss='mse' 와 같은 값
rmse = np.sqrt(mean_squared_error(y_test, y_predict))  # mse 에 루트 -> 정답과 같은 단위

# r2   : 모델 설명력 (0.7 이상이면 좋은 편)
# RMSE : 예측 오차 크기 (작을수록 좋음)
# mse  : RMSE 의 제곱 (손실함수로도 사용)
```

### Pandas CSV 로드 및 EDA

```python
import pandas as pd

df = pd.read_csv('data.csv', index_col=0)   # 첫 컬럼을 인덱스로
print(df.shape)          # (행, 열)
print(df.info())         # 자료형 + 결측치
print(df.columns)        # 컬럼명
print(df.isna().sum())   # 컬럼별 결측치 개수

df_clean = df.dropna()   # 결측치 있는 행 삭제
```

---

## 성과
- 회귀 모델 성능 평가 메트릭 완전 이해
- 평가 메트릭을 이용한 모델 성능 비교 능력 습득
- 실제 데이터셋의 구조 분석 및 EDA 능력 향상
- Pandas 데이터프레임 처리 능력 습득
- 결측치 처리 준비 (`dropna()` 함수 학습)

---

## 💡 주요 학습 포인트
1. **r2 Score**: 모델이 데이터의 분산을 얼마나 설명하는지 (1에 가까울수록 좋음)
2. **RMSE vs MSE**: RMSE는 MSE에 루트를 씌운 값 (원래 단위로 표현)
3. **평가 메트릭 활용**: r2는 설명력, RMSE는 오차 크기
4. **EDA의 중요성**: 데이터의 형태, 결측치, 특성을 먼저 파악
5. **Pandas 데이터 처리**: DataFrame으로 대용량 데이터 효율적으로 처리

---

[⬅️ Day03](Day03.md) · [🏠 전체 목차](../README.md) · [Day05 ➡️](Day05.md)
