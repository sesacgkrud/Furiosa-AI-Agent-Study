# m08_GridSearchCV_00.py 베이스

# [목적]
# m08_00 의 GridSearchCV 를 회귀 데이터 diabetes 에 적용한다 (XGBClassifier → XGBRegressor)
# - y 가 연속값이라 train_test_split 의 stratify 를 빼고, StratifiedKFold 대신 KFold 를 쓴다
# - parameters 는 m08_00 과 같다 (n_estimators · max_depth · learning_rate · min_child_weight 는 XGBRegressor 에도 있다)
# - 회귀 모델이라 model.score · best_score_ 는 accuracy 가 아니라 R2

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import time
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [                                                                                          # 총 42번
    {"n_estimators" : [100, 500], "max_depth" : [6, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},     # 18번
    {"max_depth" : [6, 8, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},                               # 12번
    {"min_child_weight" : [2, 3, 5, 10], "learning_rate" : [0.1, 0.01, 0.001]},                         # 12번
]

#2. 모델 구성
xgb = XGBRegressor()

model = GridSearchCV(
    estimator = xgb,
    param_grid = parameters,
    cv = kfold,
    verbose = 1,
    n_jobs = -1,
)

#3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

#4. 평가, 예측
print("model.score : ", model.score(x_test, y_test))     # Regressor : R2
print("best_score : ", model.best_score_)
print("걸린시간 : ", round(end - start, 2), "초")


# (442, 10) (442,)
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# 최적의 파라미터 :  {'learning_rate': 0.01, 'min_child_weight': 10}
# model.score :  0.35203541143453265
# best_score :  0.37628718516426146
# 걸린시간 :  8.16 초
# → 세 번째 dict (min_child_weight · learning_rate) 에서 최적 조합이 나왔다 (나머지 n_estimators · max_depth 는 기본값)
# → min_child_weight=10 : 잎 하나에 필요한 최소 가중치를 크게 해서 트리를 덜 쪼갠다 → 데이터가 적은 diabetes 에서 과적합을 줄인다
# → R2 0.35 로 Day43 의 선형 모델(ElasticNet 0.4875 · ARDRegression 평균 0.5055) 보다 낮다
