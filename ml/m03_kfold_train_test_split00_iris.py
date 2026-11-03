# m02_kfold_00_iris.py 베이스

# [목적]
# m02_00 은 전체 x, y 로 교차 검증만 했다. 여기서는 train_test_split 으로 test 를 먼저 떼어 두고
# train 으로 교차 검증 → test 로 예측 · 평가하는 흐름을 만든다
# - train_test_split(train_size=0.8, stratify=y) : 150 개 → train 120 개 / test 30 개 (클래스 비율 유지)
# - StandardScaler : x_train 에만 fit_transform, x_test 는 transform 만 (데이터 누수 방지)
# - cross_val_score(model, x_train, y_train, cv=kfold) : train 120 개를 5 조각으로 교차 검증 → fold 별 acc
# - cross_val_predict(model, x_test, y_test, cv=kfold) : x_test 30 개를 다시 5 조각으로 나눠
#   24 개로 훈련 · 6 개 예측을 5 번 반복 → 30 개 전체의 예측값을 돌려준다
#   (위의 train 교차 검증 모델로 예측하는 것이 아니라 x_test 안에서 새로 훈련한다)
# - accuracy_score(y_test, y_predict) : 예측값과 실제값을 비교한 정답률

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델 구성
model = DecisionTreeClassifier()

#3. 훈련, 평가
scores = cross_val_score(model, x_train, y_train, cv=kfold)

print("acc : ", scores, "\n평균 acc : ", round(np.mean(scores), 2))
# acc :  [0.95833333 0.95833333 0.95833333 1.         0.95833333] 
# 평균 acc :  0.97 (모델에 random_state 가 없어서 실행마다 0.96 ~ 0.98 로 조금씩 달라진다)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_test)       # [2 0 2 1 1 1 2 0 2 0 0 0 0 2 2 2 0 2 0 1 2 0 1 1 2 0 1 1 1 1] -> 원값
print(y_predict)    # [2 0 2 1 1 1 1 0 2 0 0 0 0 2 2 2 0 1 0 1 1 0 1 1 2 0 1 1 1 1] -> 예측값

# accuracy_score 만들어 보기
acc = accuracy_score(y_test, y_predict)     # (실제값, 예측값) → 맞힌 개수 / 전체 개수
print("cross_val_predict acc : ", round(acc, 4))
# cross_val_predict acc :  0.9 → 위 예측값 기준 30 개 중 27 개 맞힘 (실행마다 0.9 ~ 0.93 으로 조금씩 달라진다)