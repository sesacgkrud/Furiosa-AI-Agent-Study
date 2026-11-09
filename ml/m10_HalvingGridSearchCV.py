# m09_RandomSearchCV_00.py 베이스

# [목적]
# m09_00 의 RandomizedSearchCV 를 HalvingGridSearchCV 로 바꿔, digits + XGBClassifier 의 42 개 조합을
# "적은 데이터로 먼저 걸러내고, 살아남은 조합만 데이터를 늘려서" 다시 교차 검증하는 방식으로 탐색한다
# - HalvingGridSearchCV : GridSearchCV 처럼 param_grid 의 모든 조합에서 출발하지만, 단계(iter)마다
#   ① 적은 샘플 수(n_resources)로 남은 후보를 모두 교차 검증 → ② 점수 상위 1/factor 만 남기고
#   → ③ 샘플 수를 factor 배로 늘려 다시 교차 검증 을 반복한다
# - min_resources : 첫 단계(iter 0) 에서 쓰는 샘플 수
#                   (기본값 'exhaust' → 이 데이터에서는 2 × cv 개수 × 클래스 개수 = 2 × 5 × 10 = 100 으로 정해졌다)
# - max_resources : 단계에서 쓸 수 있는 최대 샘플 수 (기본값 'auto' → x_train 의 행 수 1437)
# - factor        : 단계마다 후보는 1/factor 로 줄이고, 샘플 수는 factor 배로 늘린다 (기본값 3)
# - 단계 수(n_iterations) 는 min_resources × factor^iter 가 max_resources 를 넘지 않을 때까지만 진행된다
#   → n_required_iterations : 후보가 1 개로 줄어들 때까지 필요한 단계 수
#   → n_possible_iterations : 샘플 수 범위(min ~ max) 안에서 실제로 가능한 단계 수
# - 아래 결과 주석에 min_resources · max_resources · factor 를 바꿔 가며 실행한 결과를 남겼다

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._search_successive_halving import HalvingGridSearchCV
# → HalvingGridSearchCV 는 실험적(experimental) 기능이라 from sklearn.model_selection import HalvingGridSearchCV 로 바로 쓰면 ImportError 가 난다
#   그래서 실제로 정의된 모듈(_search_successive_halving) 에서 직접 가져왔다
# from sklearn.experimental import enable_halving_search_cv
# → 공식 방법 : 이 줄을 먼저 import 하면 from sklearn.model_selection import HalvingGridSearchCV 로 가져올 수 있다
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

# print(x_train.shape) # (1437, 64)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [
    {"n_estimators" : [100, 500], "max_depth" : [6, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},
    {"max_depth" : [6, 8, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},
    {"min_child_weight" : [2, 3, 5, 10], "learning_rate" : [0.1, 0.01, 0.001]},
]

#2. 모델 구성
xgb = XGBClassifier()

model = HalvingGridSearchCV(
    estimator=xgb,
    param_grid=parameters,
    cv=kfold,
    verbose=1,
    # n_jobs=-1,
    refit=True,
    # n_iter=10,
    min_resources=50,                   # 첫 단계 샘플 수 (기본값으로 두면 100)
    max_resources=1000,                 # 단계에서 쓸 수 있는 최대 샘플 수 (기본값으로 두면 1437)
    factor=2,                           # 단계마다 후보 1/2 · 샘플 수 2 배 (기본값 3)
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

# ==============================
# 결과 1
# min_resources=100, # 2*5*10 = 2*cv*클래스 갯수
# max_resources=1437,
# factor=3 (기본값)
# ==============================

# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 100
# max_resources_: 1437
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 100
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 14
# n_resources: 300
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# ----------
# iter: 2
# n_candidates: 5
# n_resources: 900
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# model.score :  0.9444444444444444
# best_score :  0.9453755431409062
# 걸린시간 :  210.39 초
# → 후보 42 → 14 → 5 (1/3 씩, 올림), 샘플 수 100 → 300 → 900 (3 배씩)
#   다음 단계 샘플 수 2700 은 max_resources 1437 을 넘어서 iter 2 까지 3 단계만 진행 (n_possible_iterations: 3)
# → 후보를 1 개까지 줄이려면 4 단계가 필요하지만(n_required_iterations: 4) 3 단계에서 멈추고, 남은 5 개 중 최고를 고른다
# → 총 훈련 210 + 70 + 25 = 305 번 (GridSearchCV 처럼 1437 개 전체로 42 조합을 모두 돌리면 210 번이지만, 여기서는 큰 데이터로는 소수 후보만 훈련한다)


# ==============================
# 결과 2 (현재 코드 설정)
# min_resources=50,
# max_resources=1000,
# factor=2,
# ==============================

# n_iterations: 5
# n_required_iterations: 6
# n_possible_iterations: 5
# min_resources_: 50
# max_resources_: 1000
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 50
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 21
# n_resources: 100
# Fitting 5 folds for each of 21 candidates, totalling 105 fits
# ----------
# iter: 2
# n_candidates: 11
# n_resources: 200
# Fitting 5 folds for each of 11 candidates, totalling 55 fits
# ----------
# iter: 3
# n_candidates: 6
# n_resources: 400
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# ----------
# iter: 4
# n_candidates: 3
# n_resources: 800
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# model.score :  0.9416666666666667
# best_score :  0.933490566037736
# 걸린시간 :  142.75 초
# → 후보 42 → 21 → 11 → 6 → 3 (1/2 씩, 올림), 샘플 수 50 → 100 → 200 → 400 → 800 (2 배씩)
#   다음 단계 1600 은 max_resources 1000 을 넘어서 iter 4 까지 5 단계 진행
# → 총 훈련 210 + 105 + 55 + 30 + 15 = 415 번 (결과 1 은 305 번)
# → 같은 설정으로 다시 실행하니 단계 · 후보 수 · 샘플 수는 똑같고, 점수와 시간은 달랐다
#   model.score :  0.9444444444444444 / best_score :  0.9473034591194966 / 걸린시간 :  219.19 초
#   → HalvingGridSearchCV 는 단계마다 x_train 에서 n_resources 개 샘플을 무작위로 뽑는데, random_state 를 주지 않아 실행마다 뽑히는 샘플이 달라진다
#   → 그래서 한 번 실행한 결과만으로 결과 1 · 결과 2 의 점수 · 시간 우열을 정하기는 어렵다
