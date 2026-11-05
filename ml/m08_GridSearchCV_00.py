# m07_for_parameter_tuning.py 베이스

# [목적]
# m07 에서 이중 for 문으로 직접 하던 하이퍼파라미터 조합 탐색을 GridSearchCV 로 바꾼다 (iris + XGBClassifier)
# - GridSearchCV(estimator, param_grid, cv) : param_grid 의 모든 조합마다 cv(StratifiedKFold 5 조각) 로 교차 검증하고
#   평균 점수가 가장 높은 조합을 찾는다 → m07 처럼 best_score · best_parameters 를 직접 비교 · 저장할 필요가 없다
# - parameters 는 dict 의 리스트 : dict 하나 안의 값들은 모든 조합을 만들고, dict 끼리는 따로 탐색한다
#   → 2 × 3 × 3 = 18, 4 × 3 = 12, 4 × 3 = 12 → 총 42 가지 조합 (× 5 fold = 210 번 훈련)
# - n_estimators · max_depth · learning_rate · min_child_weight 는 XGBClassifier 의 파라미터
#   (dict 에 없는 파라미터는 모델 기본값을 쓴다)
# - refit=True(기본값) 라서 탐색이 끝나면 최적 조합으로 x_train 전체를 다시 훈련한 모델이 model 안에 남는다
#   → model.score · model.predict 는 그 최적 모델로 동작한다

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
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

model = GridSearchCV(
    estimator = xgb,            # 하이퍼파라미터를 튜닝할 모델
    param_grid = parameters,    # 탐색할 하이퍼파라미터 조합
    cv = kfold,                 # StratifiedKFold 5-Fold 교차검증
    verbose = 1,                # 학습 진행 상황 출력
    n_jobs = -1,                # 사용 가능한 모든 CPU 코어를 활용하여 병렬 처리
    # scoring='accuracy',       # 평가 지표: 정확도 (분류 모델의 기본값이므로 생략 가능)
    # refit=True,               # 최적 파라미터로 전체 훈련 데이터를 다시 학습 (기본값)
)

#3. 훈련
start = time.perf_counter()     # time.time() 보다 정밀한 시간 측정 (걸린 시간 재는 용도)
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 : ", model.best_estimator_)       # 최적 조합으로 다시 훈련된 모델 객체 (전체 파라미터 출력)
print("최적의 파라미터 : ", model.best_params_)          # param_grid 에서 고른 최적 조합 (dict)

#4. 평가, 예측
print("model.score : ", model.score(x_test, y_test))   # 최적 모델로 x_test 평가 (분류 : accuracy)
print("best_score : ", model.best_score_)              # 최적 조합의 교차 검증 평균 점수 (x_train 5 fold)
print("걸린시간 : ", round(end - start, 2), "초")


# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=6,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=500,
#               n_jobs=None, num_parallel_tree=None, ...)
# 최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}

# model.score : 0.9
# best_score : 0.95
# 걸린시간 : 4.39 초
# → 42 조합 × 5 fold = 210 번 훈련 ("Fitting 5 folds for each of 42 candidates")
# → 최적 조합 : learning_rate=0.1, max_depth=6, n_estimators=500 (첫 번째 dict 에서 선택)
# → best_score 0.95 는 x_train 안의 교차 검증 평균, model.score 0.9 는 처음 떼어 둔 x_test 30 개 평가 (27 개 맞힘)
# → n_jobs=-1 로 CPU 코어를 모두 써서 병렬로 돌린다 (걸린 시간은 실행마다 조금씩 다르다)