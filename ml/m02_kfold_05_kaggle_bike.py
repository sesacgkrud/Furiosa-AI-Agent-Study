# m02_kfold_01_california.py 베이스
# https://www.kaggle.com/competitions/bike-sharing-demand/data

# [목적]
# m02_01 의 KFold + cross_val_score 교차 검증을 Kaggle Bike Sharing (자전거 대여량 회귀) 에 적용한다
# - 데이터 : train.csv 를 pd.read_csv 로 불러와 x / y 를 나눈다 (keras 실습과 같은 방식)
#   → casual + registered = count 라서 두 컬럼은 정답을 그대로 알려 주는 셈이므로 x 에서 뺀다
# - 모델 : DecisionTreeRegressor → RandomForestRegressor (결정 트리 여러 개의 평균, m01 에서 사용)
#   → random_state 를 줘서 실행할 때마다 같은 점수가 나오게 했다
# - test.csv 는 정답(count) 이 없는 제출용이라 교차 검증에는 train.csv 만 쓴다

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

#1. 데이터
path = "./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)    # [10886 rows x 11 columns]

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   # (10886, 8)
y = train_csv['count']                                          # (10886,)
print(x.shape, y.shape)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델 구성
model = RandomForestRegressor(random_state=333)

#3. 훈련, 평가
scores = cross_val_score(model, x, y, cv=kfold)

print("r2 : ", scores, "\n평균 r2 : ", round(np.mean(scores), 2))
# r2 :  [0.29174901 0.27468772 0.29782    0.30663835 0.30233358]
# 평균 r2 :  0.29
# → 날씨 · 계절 컬럼 8 개만으로는 대여량을 30% 정도밖에 설명하지 못한다 (datetime 은 index 로 빠져 있다)
