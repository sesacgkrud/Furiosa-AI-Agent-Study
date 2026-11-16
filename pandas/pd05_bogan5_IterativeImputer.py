# pd03_bogan3_imputer2.py 베이스

# [IterativeImputer]
#   NaN 이 있는 컬럼을 y 로, 나머지 컬럼을 x 로 놓고 모델(estimator) 로 예측해 채운다 (결측치 처리 4. 모델 .predict)
#   - 처음에는 평균으로 임시로 채운 뒤, 컬럼마다 돌아가며 모델로 다시 예측 → 이것을 여러 번 반복해 값을 다듬는다
#   - estimator : 예측에 쓸 모델 (기본값 BayesianRidge) → 여기서는 XGBRegressor
#   - max_iter : 최대 반복 횟수 (기본값 10)
#   - tol : 반복 사이 값 변화가 이보다 작아지면 max_iter 전에 멈춘다 (기본값 0.001)
#   - n_iter_ : 실제로 반복한 횟수 → 3 번 만에 멈췄다
#   - random_state 를 XGBRegressor · IterativeImputer 모두 고정해 실행할 때마다 같은 결과가 나오게 했다

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

from xgboost import XGBRegressor
xgb = XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    random_state=333,
)

imputer = IterativeImputer(
    estimator=xgb,
    max_iter=10,        # 최대 10번
    tol=0.001,          # default
    random_state=333,
)

data2 = imputer.fit_transform(data)
print(data2)
# [[ 2.          2.          2.          7.99865484]
#  [ 9.98037434  4.          4.          4.        ]
#  [ 6.          2.01480126  6.          7.99865484]
#  [ 8.          8.          8.          8.        ]
#  [10.          7.97927809 10.          4.01578808]
#  [ 8.          4.         12.          8.        ]]

print("실제 반복 횟수", imputer.n_iter_)    # 실제 반복 횟수 3
