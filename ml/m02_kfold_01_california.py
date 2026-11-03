# m02_kfold_00_iris.py 베이스

# [목적]
# m02_00 의 KFold + cross_val_score 교차 검증을 회귀 데이터(california) 에 적용한다
# - 분류 모델 DecisionTreeClassifier → 회귀 모델 DecisionTreeRegressor
# - 회귀 모델의 cross_val_score 는 fold 별 R2(결정계수) 를 반환한다 (model.score 와 같은 기준)

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)
print(x, y)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델 구성
model = DecisionTreeRegressor()

#3. 훈련, 평가
scores = cross_val_score(model, x, y, cv=kfold) # Regressor 기준 : R2(결정계수) 반환

print("r2 : ", scores, "\n평균 r2 : ", round(np.mean(scores), 2))
# r2 :  [0.6036485  0.60703136 0.57476646 0.61100558 0.63768908]
# 평균 r2 :  0.61
# → fold 별 점수는 실행할 때마다 조금씩 달라지지만 평균은 0.61 근처
