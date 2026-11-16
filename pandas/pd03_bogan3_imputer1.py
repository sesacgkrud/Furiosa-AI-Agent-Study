# pd02_bogan2_pandas.py 베이스

# [SimpleImputer]
#   sklearn 의 결측치 채우기 도구. scaler 처럼 fit (컬럼별 채울 값 계산) → transform (NaN 채우기)
#   - fit_transform 결과는 DataFrame 이 아니라 numpy 배열 (컬럼 이름이 없어진다)
#   - strategy : "mean" (기본값, 평균) / "median" (중위값) / "most_frequent" (최빈값) / "constant" (fill_value 값)
#   - 결과는 pd02 의 data.fillna(data.mean()) · data.fillna(data.median()) 과 같다
#
# KNNImputer · IterativeImputer 는 pd04 · pd05 에서 사용
# IterativeImputer 는 아직 실험 기능이라 enable_iterative_imputer 를 먼저 import 해야 불러올 수 있다

import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]
                     ]).T

print(data)

data.columns = ["x1", "x2", "x3", "x4"]
print(data)

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = SimpleImputer()           # default = mean
data2 = imputer.fit_transform(data)
print(data2)
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]

imputer2 = SimpleImputer(strategy="mean")
data3 = imputer2.fit_transform(data)
print(data3)                        # 기본값과 같은 평균 → data2 와 같다
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]

imputer3 = SimpleImputer(strategy="median")
data4 = imputer3.fit_transform(data)
print(data4)                        # 중위값 x1 7 / x2 4 / x3 6 / x4 6
# [[ 2.  2.  2.  6.]
#  [ 7.  4.  4.  4.]
#  [ 6.  4.  6.  6.]
#  [ 8.  8.  8.  8.]
#  [10.  4. 10.  6.]]
