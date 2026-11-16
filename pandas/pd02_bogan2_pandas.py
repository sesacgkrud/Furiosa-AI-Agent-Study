# [목적]
# pandas 만으로 결측치(NaN) 를 확인 · 삭제 · 채우는 방법을 작은 DataFrame 으로 비교한다.
#
# [흐름]
#   0. 확인 : isnull() (칸마다 True / False), isnull().sum() (컬럼별 결측 개수), info() (Non-Null Count)
#   1. 삭제 : dropna() → NaN 이 하나라도 있는 행 삭제 (axis=0 기본) / axis=1 이면 열 삭제
#   2. 채우기 : fillna(값)
#      - data.mean() · data.median() 은 컬럼별 평균 · 중위값 Series → fillna 에 넣으면 컬럼마다 자기 값으로 채운다
#      - fillna(0) · fillna(777) : 정한 값으로
#      - ffill() : 바로 앞(위) 행의 값으로 채움 / bfill() : 바로 뒤(아래) 행의 값으로 채움 → 시계열에서 주로 사용
#   3. 특정 칼럼만 : data["x1"] = data["x1"].fillna(...) 처럼 컬럼마다 다른 방법으로 채울 수 있다
#
# [데이터]  리스트 4 개를 행으로 만든 뒤 .T 로 전치 → 5 행 4 열 (각 리스트가 하나의 컬럼이 된다)

import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]
                     ]).T

print(data)
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

data.columns = ["x1", "x2", "x3", "x4"]
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#0. 결측치 확인
print(data.isnull())
print(data.isnull().sum())
# x1    1
# x2    2
# x3    0
# x4    3
# dtype: int64

print(data.info())
# <class 'pandas.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64
# dtypes: float64(4)
# memory usage: 292.0 bytes
# None

#1. 결측치 삭제
print(data.dropna()) # nan 포함된 행 모두 삭제 (default = 행)
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0

# print(data.dropna(axis=0)) # 행 삭제
# print(data.dropna(axis=1)) # 열 삭제

#2-1. 특정값 - 평균
means = data.mean()
print(means)                # x1 6.5 / x2 4.666667 / x3 6.0 / x4 6.0
data2 = data.fillna(means)
print(data2)

#2-2. 특정값 - 중위값
med = data.median()
print(med)                  # x1 7.0 / x2 4.0 / x3 6.0 / x4 6.0
data3 = data.fillna(med)
print(data3)

#2-3-1. 특정값 - 0
data4 = data.fillna(0)
print(data4)

#2-3-2. 특정값 - 777
data4 = data.fillna(777)
print(data4)

#2-4. 특정값 - ffill (통상 마지막 값), (시계열)   ffill = forward fill : 앞(위) 행 값으로 채움
data5 = data.ffill()
print(data5)            # 가장 첫번째 행은 채울 값이 없어서 NaN

#2-5. 특정값 - bfill (통상 첫번째 값), (시계열)   bfill = backward fill : 뒤(아래) 행 값으로 채움
data6 = data.bfill()
print(data6)            # 가장 마지막 행은 채울 값이 없어서 NaN

#################### 특정 칼럼만 ####################
print("===========================================")
means = data["x1"].mean()
print(means)                # 6.5

med = data["x4"].median()
print(med)                  # 6.0

data["x1"] = data["x1"].fillna(means)
data["x2"] = data["x2"].ffill()     # x2 는 앞 값으로 (4 → 4, 8 → 8)
data["x4"] = data["x4"].fillna(med)

print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   6.5  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  6.0
