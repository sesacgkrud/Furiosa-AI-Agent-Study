# [목적]
# KFold 로 데이터를 n 개 조각(fold)으로 나누고, cross_val_score 로 교차 검증(Cross Validation)을 해서
# iris 분류 모델(DecisionTreeClassifier) 의 성능을 fold 마다 확인한다
#
# 흐름
# - 지금까지는 train_test_split 으로 train / test 를 한 번만 나눴다
#   → 어떤 데이터가 test 로 뽑히느냐에 따라 점수가 달라지고, test 로 뽑힌 데이터는 훈련에 쓰이지 않는다
# - KFold(n_splits=5) : 전체 데이터를 5 조각으로 나눈다
#   → 1 조각을 검증용, 나머지 4 조각을 훈련용으로 쓰는 것을 5 번 반복 (모든 데이터가 한 번씩 검증에 쓰인다)
#   → iris 150 개 기준 : 훈련 120 개 / 검증 30 개 × 5 번
# - shuffle=True : 나누기 전에 데이터를 섞는다 (iris 는 클래스 순서대로 정렬되어 있어서 섞지 않으면 fold 마다 클래스가 치우친다)
# - random_state : 섞는 순서 고정 → 실행할 때마다 같은 fold 로 나뉜다
# - cross_val_score(model, x, y, cv=kfold) : fold 마다 fit (훈련) + score (평가) 를 자동으로 반복하고
#   점수 5 개를 배열로 돌려준다 (분류 모델 → accuracy)
# - np.mean(scores) : 5 번 점수의 평균 = 이 모델의 대표 성능

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x, y)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)     # 5 조각, 섞어서 나눔

#2. 모델 구성
model = DecisionTreeClassifier()

#3. 훈련, 평가
scores = cross_val_score(model, x, y, cv=kfold)     # Classifier 기준 : fold 별 accuracy 5 개 반환

print("acc : ", scores, "\n평균 acc : ", round(np.mean(scores), 2))
# acc :  [0.93333333 0.9        1.         0.96666667 0.9       ]
# 평균 acc :  0.94
# → 모델에 random_state 를 주지 않아서 실행할 때마다 조금씩 달라진다 (예 : 마지막 fold 0.9333, 평균 0.95)
#   KFold 의 random_state 는 데이터를 나누는 순서만 고정한다
