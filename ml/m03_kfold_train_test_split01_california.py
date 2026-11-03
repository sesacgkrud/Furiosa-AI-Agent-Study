# m03_kfold_train_test_split00_iris.py 베이스

# [목적]
# m03_00 의 train_test_split + KFold 교차 검증을 회귀 데이터(california) 에 적용한다
# - 분류 모델 DecisionTreeClassifier → 회귀 모델 DecisionTreeRegressor
# - 회귀라서 train_test_split 의 stratify 를 뺐다 (y 가 연속값이라 클래스 비율을 맞출 수 없다)
# - 평가 지표 accuracy_score → r2_score (cross_val_score 도 fold 별 R2 를 반환한다)

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델 구성
model = DecisionTreeRegressor()

#3. 훈련, 평가
scores = cross_val_score(model, x_train, y_train, cv=kfold)         # Regressor 기준 : fold 별 R2 5 개 반환

print("r2 : ", scores, "\n평균 r2 : ", round(np.mean(scores), 2))
# r2 :  [0.60071355 0.60918453 0.65959733 0.60238626 0.59000983]
# 평균 r2 :  0.61 (모델에 random_state 가 없어서 실행마다 조금씩 달라진다)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_test)
print(y_predict)
# [1.875 1.563 2.545 ... 0.613 3.5   0.871] -> 원값
# [1.345 2.605 2.08  ... 0.59  4.188 0.976] -> 예측값

r2 = r2_score(y_test, y_predict)
print("cross_val_predict r2 : ", round(r2, 4))
# cross_val_predict r2 :  0.5367 → x_test 4128 개만으로 다시 5 조각 훈련 · 예측해서 train 교차 검증(0.61) 보다 낮다
