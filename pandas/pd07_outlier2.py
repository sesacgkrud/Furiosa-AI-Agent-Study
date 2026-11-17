# pd07_outlier1.py 베이스

# [목적]
# 컬럼이 2 개인 데이터에서 컬럼마다 IQR 로 이상치를 찾고, 두 가지 방법으로 처리한다.
#   방법 1) 이상치를 np.nan 으로 바꾼 뒤 결측치 처리 (interpolate → 첫 행은 bfill)
#   방법 2) np.clip 으로 이상치를 경계값(lower · upper bound) 으로 잘라내기
#
# [데이터]
#   리스트 2 개를 .T 로 전치 → (13, 2), 0 번 컬럼 / 1 번 컬럼
#
# [컬럼마다 따로 찾는 이유]
#   outlier 함수에 2 차원 배열을 통째로 넣으면 np.percentile 이 두 컬럼 값을 섞어서 사분위를 계산한다
#   → 컬럼마다 값의 범위(0 번은 -10 ~ 50, 1 번은 -70000 ~ 1000) 가 달라서 aaa[:, i] 로 한 컬럼씩 넣는다
#
# [결과]
#   0 번 컬럼 : 1사분위 4, 3사분위 10, IQR 6 → 범위 -5 ~ 19 → 이상치 -10 (0 행), 50 (12 행)
#   1 번 컬럼 : 1사분위 200, 3사분위 600, IQR 400 → 범위 -400 ~ 1200 → 이상치 -70000 (6 행)
#              (-30 은 음수지만 -400 보다 커서 이상치가 아니다)

import numpy as np
import pandas as pd

aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T

def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75])
    print("1사분위: ", quartile_1)
    print("q2: ", q2)
    print("3사분위: ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("IQR: ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data > upper_bound) | (data < lower_bound)), \
           iqr, lower_bound, upper_bound

## 1. 이상치를 찾아내고, 이상치를 처리해라! (np.nan 처리 후 결측치 처리)

########## 컬럼별 이상치 찾기 ##########
aaa_nan = aaa.astype(float)         # 정수 배열에는 np.nan 을 넣을 수 없어서 실수형으로 복사
lower = []                          # 컬럼별 하한 (np.clip 에서 사용)
upper = []                          # 컬럼별 상한

for i in range(aaa.shape[1]):       # aaa.shape[1] = 컬럼 개수 2
    print("========== {} 번 컬럼 ==========".format(i))
    outlier_loc, iqr, low, up = outlier(aaa[:, i])
    print("이상치의 위치: ", outlier_loc)
    print("이상치: ", aaa[outlier_loc[0], i])     # np.where 결과는 튜플 → [0] 으로 위치 배열만 꺼낸다
    lower.append(low)
    upper.append(up)

    aaa_nan[outlier_loc, i] = np.nan    # 이상치 자리를 NaN 으로

# ========== 0 번 컬럼 ==========
# 1사분위:  4.0
# q2:  7.0
# 3사분위:  10.0
# IQR:  6.0
# 이상치의 위치:  (array([ 0, 12]),)
# 이상치:  [-10  50]
# ========== 1 번 컬럼 ==========
# 1사분위:  200.0
# q2:  400.0
# 3사분위:  600.0
# IQR:  400.0
# 이상치의 위치:  (array([6]),)
# 이상치:  [-70000]

print(aaa_nan)
# [[  nan  100.]
#  [   2.  200.]
#  [   3.  -30.]
#  [   4.  400.]
#  [   5.  500.]
#  [   6.  600.]
#  [   7.   nan]
#  [   8.  800.]
#  [   9.  900.]
#  [  10. 1000.]
#  [  11.  210.]
#  [  12.  420.]
#  [  nan  350.]]

########## 방법 1) NaN 으로 바꾼 뒤 결측치 처리 ##########
df = pd.DataFrame(aaa_nan, columns=["x1", "x2"])
print(df.isnull().sum())            # x1 2 / x2 1

df = df.interpolate()               # 중간 NaN 은 앞뒤 값 직선 보간, 마지막 NaN 은 이전 값
df = df.bfill()                     # 첫 번째 NaN 은 interpolate 로 안 채워져서 뒤 값으로 채운다
print(df)
#       x1      x2
# 0    2.0   100.0     ← -10 → 2  (bfill)
# 1    2.0   200.0
# 2    3.0   -30.0
# 3    4.0   400.0
# 4    5.0   500.0
# 5    6.0   600.0
# 6    7.0   700.0     ← -70000 → 700  (600 과 800 사이 보간)
# 7    8.0   800.0
# 8    9.0   900.0
# 9   10.0  1000.0
# 10  11.0   210.0
# 11  12.0   420.0
# 12  12.0   350.0     ← 50 → 12  (마지막이라 이전 값)

############### 이거 사용하기 ###############
# aaa_clipped = np.clip(aaa, lower, upper)
# print(aaa_clipped)

########## 방법 2) np.clip 으로 경계값까지 잘라내기 ##########
# np.clip(배열, 최솟값, 최댓값) : 최솟값보다 작으면 최솟값, 최댓값보다 크면 최댓값으로 바꾼다
# lower · upper 가 컬럼별 2 개짜리라 (13, 2) 배열의 컬럼마다 자기 경계가 적용된다
aaa_clipped = np.clip(aaa, lower, upper)
print(aaa_clipped)
# [[  -5.  100.]     ← -10 → -5 (0 번 컬럼 하한)
#  [   2.  200.]
#  [   3.  -30.]
#  [   4.  400.]
#  [   5.  500.]
#  [   6.  600.]
#  [   7. -400.]     ← -70000 → -400 (1 번 컬럼 하한)
#  [   8.  800.]
#  [   9.  900.]
#  [  10. 1000.]
#  [  11.  210.]
#  [  12.  420.]
#  [  19.  350.]]    ← 50 → 19 (0 번 컬럼 상한)
