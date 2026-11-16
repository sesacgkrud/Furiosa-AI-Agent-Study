# Day52 - 결측치 처리 (interpolate 보간 · fillna · ffill · bfill) & sklearn SimpleImputer · KNNImputer · IterativeImputer & pandas loc · iloc 행 · 열 선택

**학습 기간:** 2026-11-16

> 결측치(NaN) 처리 방법(삭제 · 임의의 값 · 보간 · 모델 예측 · 트리 모델 · 스케일링) 을 정리하고, 날짜 인덱스 시계열에 `interpolate` 보간을 적용했다 (pd01).
>
> pandas 로 결측치를 확인(`isnull` · `info`) · 삭제(`dropna`) · 채우기(`fillna` 평균 · 중위 · 0 · 777, `ffill` · `bfill`) 했다 (pd02).
>
> sklearn `SimpleImputer` (mean · median · most_frequent · constant · lambda), `KNNImputer` (가까운 행 평균), `IterativeImputer` (XGBRegressor 로 예측) 로 결측치를 채워 비교했다 (pd03 ~ pd05).
>
> 주식 종목 DataFrame 으로 `df[컬럼]` · `loc` (이름) · `iloc` (위치) 의 차이와 범위 선택을 확인했다 (pd06).

---

## 🎯 한눈에 보기

### 결측치 처리 방법

| 방법 | 코드 | 특징 |
|---|---|---|
| 삭제 | `dropna()` / `dropna(axis=1)` | NaN 이 있는 행 / 열 삭제 |
| 평균 · 중위 | `fillna(data.mean())` · `SimpleImputer(strategy="mean" / "median")` | 평균은 이상치에 약하다 |
| 0 · 특정값 | `fillna(0)` · `fillna(777)` · `strategy="constant", fill_value=777` | 조건을 보고 넣는 게 낫다 |
| 앞값 · 뒷값 | `ffill()` · `bfill()` | 시계열, 첫 행 / 마지막 행은 채울 값이 없다 |
| 최빈값 | `strategy="most_frequent"` | 가장 많이 나온 값 |
| 보간 | `interpolate()` | 앞뒤 값을 직선으로 이어 채움 |
| KNN | `KNNImputer(n_neighbors=2)` | 비슷한 행 n 개의 평균 |
| 모델 예측 | `IterativeImputer(estimator=XGBRegressor())` | 다른 컬럼으로 결측 컬럼을 예측 |

### df[ ] · loc · iloc

| 방법 | 기준 | 예 |
|---|---|---|
| `df["종목명"]["045"]` | 컬럼 먼저 (열 → 행) | 아모레 |
| `df.loc["045", "종가"]` | 인덱스 이름 · 컬럼 이름 | 6000 |
| `df.iloc[3, 2]` | 정수 위치 (0 부터) | 6000 |

---

## 📖 핵심 학습 내용

### 1. 결측치 처리 정리 · interpolate 보간 (pd01)

```python
dates = pd.to_datetime(dates, dayfirst=True)    # "11/11/2026" → 2026-11-11 (일/월/년)

ts = pd.Series([0, 2, np.nan, np.nan, 8, 10, np.nan, 14, 16 ], index=dates)
ts = ts.interpolate()
# 2026-11-13     4.0   ← 2 와 8 사이
# 2026-11-14     6.0
# 2026-11-17    12.0   ← 10 과 14 사이
```

- 결측치 처리 : 1. 삭제 2. 임의의 값 (0 · 평균 · 중위 · ffill · bfill · 특정값 · 최빈값) 3. interpolate 4. 모델 `.predict` 5. 부스팅(트리) 계열 모델은 결측치 · 이상치 영향을 덜 받음 6. RobustScaler 등 스케일링
- `pd.to_datetime(..., dayfirst=True)` : 일이 앞에 오는 날짜 문자열 → `DatetimeIndex` (dtype `datetime64[us]`)
- `interpolate()` : 중간 NaN 은 linear 로, 마지막 NaN 은 이전 값으로 채워지고, 첫 번째 NaN 은 채워지지 않는다

### 2. pandas 결측치 확인 · 삭제 · 채우기 (pd02)

```python
data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]
                     ]).T
data.columns = ["x1", "x2", "x3", "x4"]

print(data.isnull().sum())      # x1 1 / x2 2 / x3 0 / x4 3
print(data.info())              # Non-Null Count : 4 / 3 / 5 / 2

print(data.dropna())            # NaN 이 없는 3 번 행만 남는다

data2 = data.fillna(data.mean())    # 컬럼별 평균 (x1 6.5, x2 4.67, x3 6, x4 6)
data3 = data.fillna(data.median())  # 컬럼별 중위 (x1 7, x2 4, x3 6, x4 6)
data4 = data.fillna(0)
data4 = data.fillna(777)
data5 = data.ffill()            # 앞 행 값 → 첫 행 x4 는 NaN
data6 = data.bfill()            # 뒤 행 값 → 마지막 행 x2 · x4 는 NaN

# 특정 칼럼만
data["x1"] = data["x1"].fillna(data["x1"].mean())   # 6.5
data["x2"] = data["x2"].ffill()
data["x4"] = data["x4"].fillna(data["x4"].median()) # 6.0
```

- 리스트 4 개를 `.T` 로 전치해 5 행 4 열 (각 리스트가 컬럼)
- `dropna()` 기본은 행 삭제 (`axis=0`), `axis=1` 은 열 삭제
- `data.mean()` · `data.median()` 은 컬럼별 값 Series → `fillna` 가 컬럼마다 자기 값으로 채운다
- 컬럼마다 다른 방법으로 채울 수 있다 (x1 평균, x2 ffill, x4 중위)

### 3. SimpleImputer (pd03_imputer1 · imputer2)

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = SimpleImputer()                       # default = mean
imputer3 = SimpleImputer(strategy="median")
imputer = SimpleImputer(strategy="most_frequent")
imputer2 = SimpleImputer(strategy="constant", fill_value=777)
imputer3 = SimpleImputer(strategy=lambda x : np.min(x))
data2 = imputer.fit_transform(data)
```

| strategy | x1 | x2 | x4 |
|---|---|---|---|
| mean (기본) | 6.5 | 4.67 | 6 |
| median | 7 | 4 | 6 |
| most_frequent (6 행 데이터) | 8 | 4 | 8 |
| constant 777 | 777 | 777 | 777 |
| lambda np.min (6 행 데이터) | 2 | 2 | 4 |

- fit (채울 값 계산) → transform (채우기), 결과는 numpy 배열
- `IterativeImputer` 는 실험 기능이라 `enable_iterative_imputer` 를 먼저 import
- strategy 에 함수(lambda) 를 넣으면 컬럼마다 NaN 을 뺀 값으로 계산한 결과로 채운다
- imputer2 부터는 최빈값이 보이도록 6 번째 행 `[8, 4, 12, 8]` 을 추가

### 4. KNNImputer (pd04)

```python
imputer = KNNImputer(n_neighbors=2)     # default 5
data2 = imputer.fit_transform(data)
# [[ 2.  2.  2.  6.]
#  [ 4.  4.  4.  4.]
#  [ 6.  6.  6.  6.]
#  [ 8.  8.  8.  8.]
#  [10.  6. 10.  8.]
#  [ 8.  4. 12.  8.]]
```

- NaN 이 있는 행과 가장 가까운 행 n_neighbors 개의 평균으로 채운다 (거리는 둘 다 값이 있는 컬럼끼리)
- 컬럼 전체 값 하나가 아니라 행마다 다른 값으로 채워진다 (x2 : 6, 6 / x4 : 6, 6, 8)

### 5. IterativeImputer + XGBRegressor (pd05)

```python
xgb = XGBRegressor(max_depth=5, learning_rate=0.1, random_state=333)

imputer = IterativeImputer(
    estimator=xgb,
    max_iter=10,        # 최대 10번
    tol=0.001,          # default
    random_state=333,
)
data2 = imputer.fit_transform(data)
print("실제 반복 횟수", imputer.n_iter_)    # 3
```

- 결측 컬럼을 y, 나머지 컬럼을 x 로 모델이 예측해서 채우고, 이를 반복해 값을 다듬는다 (결측치 처리 4. 모델 .predict)
- `estimator` 기본값은 BayesianRidge → XGBRegressor 로 지정
- `tol` 보다 변화가 작아지면 `max_iter` 전에 멈춘다 → `n_iter_` = 3

| 행 | 컬럼 | 채워진 값 |
|---|---|---|
| 0 | x4 | 7.99865 |
| 1 | x1 | 9.98037 |
| 2 | x2 · x4 | 2.01480 · 7.99865 |
| 4 | x2 · x4 | 7.97928 · 4.01579 |

### 6. df[ ] · loc · iloc (pd06_loc_iloc1)

```python
df = pd.DataFrame(data=data, index=index, columns=columns)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500

print(df["종목명"])             # ★ Pandas 열행 ← 컬럼이 기준
print(df["종목명"]["045"])      # 아모레

print(df.loc["045"])            # 아모레 행 (이름)
print(df.iloc[3, 2])            # 6000   [행 위치, 열 위치]
print(df.iloc[3]["종가"])       # 6000
print(df.loc["045", "종가"])    # 6000   [행 이름, 열 이름]
print(df.iloc[3].iloc[2])       # 6000
print(df.loc["045"].iloc[2])    # 6000
```

| 시도 | 결과 |
|---|---|
| `df[0]` · `df["031"]` · `df["045"]["종목명"]` | KeyError (df[ ] 는 컬럼 이름) |
| `df.iloc["045"]` | TypeError (iloc 은 정수만) |
| `df.loc[3]` | KeyError (loc 은 이름만) |
| `df.iloc[3][2]` · `df.loc["045"][2]` | KeyError (pandas 3 에서 Series[정수] 는 이름으로 찾음 → `.iloc[2]`) |
| `df.iloc[3, "종가"]` | ValueError |

- loc : 인덱스(이름) 기준 / iloc : 행 번호(int location) 기준 — "int loc" 로 외우기

### 7. 범위 선택 (pd06_loc_iloc2_range)

```python
print(df.iloc[3:5, 1])                  # 아모레 · 네이버 시가 (5 미포함)
print(df.iloc[3:5]["시가"])
print(df.loc["045":"023"]["시가"])      # "023" 포함
print(df.loc["045":"023", "시가"])
# 045    3500
# 023     100

print(df.iloc[3:5].iloc[1])             # 네이버 행
print(df.loc["045":"023"].iloc[1])      # 네이버 행
```

- `iloc[3:5]` 는 끝 미포함, `loc["045":"023"]` 는 끝 포함
- `iloc[3:5, "시가"]` → ValueError, `iloc["045":"023", 1]` → TypeError, `loc["045":"023"].loc["시가"]` → KeyError (행 이름으로 찾음)

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `pandas/pd01_bogan1_interpolate.py` | 결측치 처리 방법 정리, `to_datetime(dayfirst=True)` 날짜 인덱스 Series 에 `interpolate` 보간 |
| `pandas/pd02_bogan2_pandas.py` | `isnull` · `info` 확인, `dropna`, `fillna` (평균 · 중위 · 0 · 777), `ffill` · `bfill`, 컬럼별 채우기 |
| `pandas/pd03_bogan3_imputer1.py` | `SimpleImputer` mean (기본) · median |
| `pandas/pd03_bogan3_imputer2.py` | `SimpleImputer` most_frequent · constant (777) · lambda np.min |
| `pandas/pd04_bogan4_KNNImputer.py` | `KNNImputer(n_neighbors=2)` |
| `pandas/pd05_bogan5_IterativeImputer.py` | `IterativeImputer(estimator=XGBRegressor)`, max_iter · tol · n_iter_ |
| `pandas/pd06_loc_iloc1.py` | `df[컬럼]` · `loc` · `iloc` 로 행 · 값 선택, 에러 경우 비교 |
| `pandas/pd06_loc_iloc2_range.py` | `iloc[3:5]` · `loc["045":"023"]` 범위 선택 |
| `docs/Day52.md` | Day52 학습 기록 신규 작성 |
| `docs/Day51.md` | 하단 nav에 Day52 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(pandas/ Day52) / 진행도(52일, 65%) 갱신 (수정) |

---

## 💻 핵심 개념

### pandas 결측치

```
data.isnull().sum()        : 컬럼별 NaN 개수
data.dropna(axis=0 / 1)    : NaN 있는 행 / 열 삭제
data.fillna(값 또는 Series) : NaN 채우기 (Series 면 컬럼별 값)
data.ffill() / bfill()     : 앞 행 / 뒤 행 값으로
ts.interpolate()           : 앞뒤 값 직선 보간 (첫 NaN 은 그대로)
```

### sklearn Imputer

```
SimpleImputer(strategy="mean" | "median" | "most_frequent" | "constant" | 함수)
KNNImputer(n_neighbors=5)                    : 가까운 행 평균
IterativeImputer(estimator, max_iter, tol)   : 모델로 반복 예측, n_iter_ = 실제 반복 수
  (from sklearn.experimental import enable_iterative_imputer 먼저)
fit_transform 결과 = numpy 배열
```

### df[ ] · loc · iloc

```
df["열"]["행이름"]       : 열 → 행 (Pandas 열행)
df.loc["행이름", "열"]   : 이름 기준, 범위 끝 포함
df.iloc[행위치, 열위치]  : 정수 위치 기준, 범위 끝 미포함
Series.loc["이름"] / Series.iloc[위치]
```

---

## 성과

- 결측치 처리 방법 6 가지를 정리하고 interpolate 보간을 시계열에 적용했다
- pandas `dropna` · `fillna` · `ffill` · `bfill` 로 결측치를 처리하고 컬럼별로 다른 방법을 적용했다
- SimpleImputer 5 가지 strategy, KNNImputer, XGBRegressor 를 쓴 IterativeImputer 결과를 비교했다
- `df[ ]` · `loc` · `iloc` 의 기준 차이와 범위 선택, 에러가 나는 경우를 확인했다

---

## 💡 주요 학습 포인트

1. **interpolate**: 중간 NaN 은 직선 보간, 마지막은 이전 값, 첫 번째는 채워지지 않는다
2. **pd.to_datetime(dayfirst=True)**: "일/월/년" 문자열을 날짜로
3. **fillna(data.mean())**: 컬럼마다 자기 평균으로 채운다 (평균은 이상치 주의)
4. **ffill · bfill**: 시계열에서 앞 · 뒤 값으로, 맨 앞 · 맨 뒤는 남는다
5. **SimpleImputer**: mean · median · most_frequent · constant · 함수(lambda)
6. **KNNImputer**: 비슷한 행 n 개의 평균 → 행마다 다른 값
7. **IterativeImputer**: 다른 컬럼으로 결측 컬럼을 모델이 예측, tol 로 조기 종료 (n_iter_)
8. **loc vs iloc**: 이름 (끝 포함) vs 정수 위치 (끝 미포함), df[ ] 는 컬럼 기준

---

[⬅️ Day51](Day51.md) · [🏠 전체 목차](../README.md)
