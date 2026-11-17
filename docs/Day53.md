# Day53 - pandas 조건 행 선택 (불리언 인덱싱) & IQR 이상치 탐지 (np.percentile · boxplot) & 이상치 처리 (NaN 후 결측치 처리 · np.clip)

**학습 기간:** 2026-11-17

> 주식 종목 DataFrame 에서 `df["시가"] >= 1100` 같은 조건으로 행을 골라냈다 (pd06_loc_iloc3).
>
> `np.percentile` 로 1사분위 · 3사분위를 구해 IQR 범위 밖의 값을 이상치로 찾고, boxplot 과 상한 · 하한 선으로 확인했다 (pd07_outlier1).
>
> 컬럼이 2 개인 데이터에서 컬럼마다 이상치를 찾아 ① NaN 으로 바꾼 뒤 결측치 처리(interpolate · bfill) ② `np.clip` 으로 경계값까지 잘라내기, 두 방법으로 처리했다 (pd07_outlier2).

---

## 🎯 한눈에 보기

| 파일 | 내용 | 결과 |
|---|---|---|
| pd06_loc_iloc3_conditions | `df[df["시가"] >= 1100]` 조건 행 선택 | 현대 · LG · 아모레 3 개 행 |
| pd07_outlier1 | 1 차원 IQR 이상치 · boxplot | 범위 -5 ~ 19 → 이상치 -10, 50 |
| pd07_outlier2 | 2 컬럼 이상치 → NaN 후 결측치 처리 / np.clip | -10 → 2 · 50 → 12 · -70000 → 700 / -10 → -5 · 50 → 19 · -70000 → -400 |

---

## 📖 핵심 학습 내용

### 1. 조건으로 행 고르기 (pd06_loc_iloc3_conditions)

```python
data = [
    ["삼성", 1000, 2000],      # 시가 · 종가를 숫자로 (크기 비교용)
    ...
]
df = pd.DataFrame(data=data, index=index, columns=columns)

cond1 = df["시가"] >= 1100     # 행마다 True / False
print(df[cond1])
print(df[df["시가"] >= 1100])
#      종목명    시가    종가
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000

print(df.loc[df["시가"] >= 1100])
# print(df.iloc[df["시가"] >= 1100]) # ValueError: iLocation based boolean indexing cannot use an indexable as a mask

print(df[df["시가"] >= 1100]["종가"])   # 3000 / 500 / 6000
# print(df[df["시가"] >= 1100].loc["종가"]) # KeyError
```

- 비교 연산 결과는 bool Series → `df[조건]` 에 넣으면 True 인 행만 남는다 (불리언 인덱싱)
- `df[조건]` = `df.loc[조건]`, `iloc` 은 인덱스가 붙은 Series 조건을 받지 못해 ValueError
- 고른 결과에서 열은 `["종가"]` 로 (`.loc["종가"]` 는 행 이름으로 찾아 KeyError)
- 크기 비교를 하려고 이전 파일의 문자열 숫자("1000") 대신 정수(1000) 로 데이터를 만들었다

### 2. IQR 이상치 탐지 · boxplot (pd07_outlier1)

```python
aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])

def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data > upper_bound) | (data < lower_bound)), \
           iqr, lower_bound, upper_bound

outlier_loc, iqr, low, up = outlier(aaa)
print("이상치의 위치: ", outlier_loc)   # (array([ 0, 12]),)

plt.boxplot(aaa)
plt.axhline(up, color="pink", label="upper bound")
plt.axhline(low, color="pink", label ="lower bound")
plt.legend()
plt.show()
```

| 값 | 결과 |
|---|---|
| 1사분위 (25%) | 4.0 |
| 중위값 q2 (50%) | 7.0 |
| 3사분위 (75%) | 10.0 |
| IQR | 6.0 |
| 하한 · 상한 | -5.0 · 19.0 |
| 이상치 | -10 (0 번), 50 (12 번) |

- IQR = Q3 - Q1, 하한 = Q1 - 1.5 × IQR, 상한 = Q3 + 1.5 × IQR → 범위 밖이 이상치
- `np.where(조건1 | 조건2)` : 둘 중 하나라도 참인 위치를 튜플로
- 함수에서 값 4 개를 return 해 한 번에 나눠 받는다
- `plt.boxplot` : 상자 Q1 ~ Q3, 가운데 선 중위값, 바깥 점 이상치 / `plt.axhline` 으로 상한 · 하한 가로선

### 3. 2 컬럼 이상치 처리 (pd07_outlier2)

```python
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T   # (13, 2)

aaa_nan = aaa.astype(float)         # 정수 배열에는 np.nan 을 넣을 수 없다
lower = []
upper = []

for i in range(aaa.shape[1]):       # 컬럼마다 따로
    outlier_loc, iqr, low, up = outlier(aaa[:, i])
    lower.append(low)
    upper.append(up)
    aaa_nan[outlier_loc, i] = np.nan

# 방법 1) NaN 으로 바꾼 뒤 결측치 처리
df = pd.DataFrame(aaa_nan, columns=["x1", "x2"])
df = df.interpolate()
df = df.bfill()

# 방법 2) np.clip
aaa_clipped = np.clip(aaa, lower, upper)
```

| 컬럼 | Q1 / Q3 / IQR | 범위 | 이상치 |
|---|---|---|---|
| 0 (x1) | 4 / 10 / 6 | -5 ~ 19 | -10 (0 행), 50 (12 행) |
| 1 (x2) | 200 / 600 / 400 | -400 ~ 1200 | -70000 (6 행) |

| 이상치 | 방법 1 (NaN → interpolate · bfill) | 방법 2 (np.clip) |
|---|---|---|
| x1 -10 (0 행) | 2 (첫 행이라 bfill) | -5 (하한) |
| x2 -70000 (6 행) | 700 (600 과 800 사이 보간) | -400 (하한) |
| x1 50 (12 행) | 12 (마지막 행이라 이전 값) | 19 (상한) |

- 2 차원 배열을 통째로 넣으면 두 컬럼 값이 섞여 사분위가 계산된다 → `aaa[:, i]` 로 컬럼마다
- -30 은 음수지만 x2 하한 -400 보다 커서 이상치가 아니다
- `np.clip(배열, 최솟값, 최댓값)` : lower · upper 가 컬럼별 2 개짜리라 컬럼마다 자기 경계가 적용된다
- 방법 1 은 이상치를 주변 값으로 다시 채우고, 방법 2 는 값을 지우지 않고 경계까지만 줄인다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `pandas/pd06_loc_iloc3_conditions.py` | `df[df["시가"] >= 1100]` · `df.loc[조건]` 조건 행 선택, 종가 열 선택 |
| `pandas/pd07_outlier1.py` | `np.percentile` 사분위 · IQR 로 이상치 위치 찾기, `boxplot` · `axhline` |
| `pandas/pd07_outlier2.py` | 2 컬럼 데이터 컬럼별 이상치 → NaN 후 `interpolate` · `bfill` / `np.clip` |
| `docs/Day53.md` | Day53 학습 기록 신규 작성 |
| `docs/Day52.md` | 하단 nav에 Day53 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(pandas/ Day53) / 진행도(53일, 66%) 갱신 (수정) |

---

## 💻 핵심 개념

### 불리언 인덱싱

```
df["시가"] >= 1100          : 행마다 True / False (bool Series)
df[조건] = df.loc[조건]      : True 인 행만
df.iloc[조건]               : ValueError (iloc 은 정수 위치만)
df[조건]["종가"]            : 고른 행의 종가 열
```

### IQR 이상치

```
Q1, q2, Q3 = np.percentile(data, [25, 50, 75])
IQR = Q3 - Q1
하한 = Q1 - 1.5 * IQR,  상한 = Q3 + 1.5 * IQR
np.where((data > 상한) | (data < 하한))  : 이상치 위치
```

### 이상치 처리

```
1) 이상치 → np.nan → 결측치 처리 (interpolate · bfill · fillna · Imputer ...)
2) np.clip(data, 하한, 상한) → 범위 밖 값을 경계값으로
컬럼이 여러 개면 컬럼마다 따로 IQR 계산
```

---

## 성과

- 조건식으로 DataFrame 행을 골라내고 loc · iloc 에서의 차이를 확인했다
- IQR 로 이상치 위치를 찾는 함수를 만들고 boxplot 으로 확인했다
- 컬럼별로 이상치를 찾아 NaN 후 결측치 처리 · np.clip 두 방법으로 처리하고 결과를 비교했다

---

## 💡 주요 학습 포인트

1. **불리언 인덱싱**: `df[df["시가"] >= 1100]`, loc 은 되고 iloc 은 ValueError
2. **np.percentile**: 1사분위 · 중위값 · 3사분위를 한 번에
3. **IQR**: Q3 - Q1, 1.5 × IQR 바깥이 이상치
4. **boxplot · axhline**: 이상치와 경계선을 그림으로 확인
5. **컬럼별 이상치**: 컬럼마다 값의 범위가 달라 따로 계산
6. **이상치 → NaN → 결측치 처리**: Day52 의 interpolate · bfill 을 이어서 사용
7. **np.clip**: 범위 밖 값을 하한 · 상한으로 잘라낸다

---

[⬅️ Day52](Day52.md) · [🏠 전체 목차](../README.md)
