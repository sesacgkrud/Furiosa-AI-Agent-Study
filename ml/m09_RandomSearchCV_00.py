# m08_GridSearchCV_00.py 베이스

# [목적]
# m08_00 의 GridSearchCV 를 RandomizedSearchCV 로 바꿔, 42 개 조합 중 일부만 무작위로 뽑아 탐색한다 (iris + XGBClassifier)
# - GridSearchCV : param_grid 의 모든 조합을 교차 검증 (42 조합 × 5 fold = 210 번 훈련)
# - RandomizedSearchCV : param_distributions 에서 n_iter (기본값 10) 개 조합만 무작위로 뽑아 교차 검증 (10 × 5 = 50 번 훈련)
#   → 조합이 많을 때 시간을 크게 줄일 수 있지만, 최적 조합이 뽑히지 않으면 점수가 조금 낮을 수 있다
# - random_state 로 뽑히는 조합을 고정하고, n_iter 로 뽑을 개수를 정한다

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [                                                                                          # 총 42번
    {"n_estimators" : [100, 500], "max_depth" : [6, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},     # 18번
    {"max_depth" : [6, 8, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},                               # 12번
    {"min_child_weight" : [2, 3, 5, 10], "learning_rate" : [0.1, 0.01, 0.001]},                         # 12번
]

#2. 모델 구성
xgb = XGBClassifier()

model = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=parameters,     # 탐색할 하이퍼파라미터 조합 (이 중 일부만 무작위로 뽑는다)
                                        # param_grid=parameters,
                                        # → TypeError: RandomizedSearchCV.__init__() got an unexpected keyword argument 'param_grid'
                                        #   [원인] param_grid 는 GridSearchCV 의 인자 이름이다. RandomizedSearchCV 는 같은 역할의 인자 이름이 param_distributions
                                        #   [수정] param_grid → param_distributions (parameters 리스트는 그대로 넘기면 된다)
    cv=kfold,
    verbose=1,
    n_jobs=-1,
    refit=True,
    random_state=333,                   # 무작위로 뽑는 조합을 실행마다 같게 고정
    # n_iter=3                          # 뽑을 조합 개수 (기본값 10) → 3 이면 3 × 5 = 15 번 훈련
)

#3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

# print("최적의 매개변수 :", model.best_estimator_)
# print("최적의 파라미터 :", model.best_params_)

#4. 평가, 예측
print("model.score : ", model.score(x_test, y_test))
print("best_score : ", model.best_score_)
print("걸린시간 : ", round(end - start, 2), "초")

# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# model.score :  0.9
# best_score :  0.9333333333333332
# 걸린시간 :  3.89 초
# → RandomizedSearchCV 는 42 개 조합을 전부 돌지 않고 n_iter (기본값 10) 개만 무작위로 뽑아 교차 검증한다
#   → 10 조합 × 5 fold = 50 번 훈련 (GridSearchCV m08_00 은 42 조합 × 5 = 210 번)
# → 뽑힌 10 개 중 최적 파라미터 : {'min_child_weight': 2, 'learning_rate': 0.1} (best_params_ 출력 주석을 풀면 확인)
# → GridSearchCV 의 best_score 0.95 보다 조금 낮다 : 그 최적 조합(n_estimators=500 · max_depth=6) 이 무작위 10 개에 뽑히지 않았다