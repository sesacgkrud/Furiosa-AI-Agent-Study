# pd03_bogan3_imputer1.py 베이스

# [SimpleImputer strategy 추가]
#   - "most_frequent" : 컬럼별 최빈값 (x1 = 8, x2 = 4, x3 = 없음, x4 = 8)
#     → 최빈값이 잘 보이도록 데이터에 6 번째 행 [8, 4, 12, 8] 을 추가했다
#   - "constant" + fill_value=777 : 정한 값으로 채움
#   - strategy 에 함수(callable) 를 넣으면 컬럼마다 NaN 을 뺀 값 배열을 받아 그 결과로 채운다
#     lambda x : np.min(x) → 컬럼별 최솟값 (x1 = 2, x2 = 2, x4 = 4) / np.max 로 바꾸면 최댓값

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = pd.DataFrame([[2, np.nan, 6, 8, 10, 8],
                     [2, 4, np.nan, 8, np.nan, 4],
                     [2, 4, 6, 8, 10, 12],
                     [np.nan, 4, np.nan, 8, np.nan, 8]
                     ]).T

print(data)
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# 5   8.0  4.0  12.0  8.0

imputer = SimpleImputer(strategy="most_frequent")
data2 = imputer.fit_transform(data)
print(data2)
# [[ 2.  2.  2.  8.]
#  [ 8.  4.  4.  4.]
#  [ 6.  4.  6.  8.]
#  [ 8.  8.  8.  8.]
#  [10.  4. 10.  8.]
#  [ 8.  4. 12.  8.]]

imputer2 = SimpleImputer(strategy="constant", fill_value=777)
data3 = imputer2.fit_transform(data)
print(data3)
# [[  2.   2.   2. 777.]
#  [777.   4.   4.   4.]
#  [  6. 777.   6. 777.]
#  [  8.   8.   8.   8.]
#  [ 10. 777.  10. 777.]
#  [  8.   4.  12.   8.]]

# def 함수():
#     return 1

imputer3 = SimpleImputer(
                        # strategy=lambda x : np.max(x)
                        strategy=lambda x : np.min(x)
                        )
data4 = imputer3.fit_transform(data)
print(data4)
# [[ 2.  2.  2.  4.]
#  [ 2.  4.  4.  4.]
#  [ 6.  2.  6.  4.]
#  [ 8.  8.  8.  8.]
#  [10.  2. 10.  4.]
#  [ 8.  4. 12.  8.]]
