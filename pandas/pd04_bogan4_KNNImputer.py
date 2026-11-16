# pd03_bogan3_imputer2.py 베이스

# [KNNImputer]
#   NaN 이 있는 행과 가장 가까운(비슷한) 행 n_neighbors 개를 찾아, 그 행들 값의 평균으로 채운다
#   - 거리는 두 행 모두 값이 있는 컬럼끼리만 계산한다 (NaN 은 빼고 계산)
#   - n_neighbors 기본값 5 → 여기서는 2 로 가까운 2 개 행 평균
#   - 예) 1 행 x1 : 1 행 (NaN, 4, 4, 4) 과 가까운 0 행 (2) · 2 행 (6) 의 평균 → 4
#   - 평균 · 중위값처럼 컬럼 전체 값 하나로 채우지 않고, 행마다 비슷한 행을 보고 다르게 채운다

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

data.columns = ["x1", "x2", "x3", "x4"]
print(data)

# imputer = KNNImputer()
# imputer = KNNImputer(n_neighbors=5) # default
imputer = KNNImputer(n_neighbors=2)
data2 = imputer.fit_transform(data)
print(data2)

# [[ 2.  2.  2.  6.]
#  [ 4.  4.  4.  4.]
#  [ 6.  6.  6.  6.]
#  [ 8.  8.  8.  8.]
#  [10.  6. 10.  8.]
#  [ 8.  4. 12.  8.]]
