# m09_RandomSearchCV_00.py 베이스

# [목적]
# m09_00 의 RandomizedSearchCV 를 회귀 데이터 diabetes 에 적용한다 (XGBClassifier → XGBRegressor)
# - y 가 연속값이라 train_test_split 의 stratify 를 빼고, StratifiedKFold 대신 KFold 를 쓴다
#   (stratify=y 를 그대로 두면 y 값을 클래스로 보는데, 1 개뿐인 값이 많아 나눌 수 없어 ValueError :
#    "The least populated classes in y have only 1 member, which is too few.")
# - 회귀 모델이라 model.score · best_score_ 는 accuracy 가 아니라 R2
# - 42 개 조합 중 n_iter (기본값 10) 개만 무작위로 뽑아 교차 검증 → GridSearchCV (m08_02) 와 비교한다

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import RandomizedSearchCV
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

model = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=parameters,
    cv=kfold,
    verbose=1,
    n_jobs=-1,
    refit=True,
    random_state=333,
)

#3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 :", model.best_estimator_)
print("최적의 파라미터 :", model.best_params_)

#4. 평가, 예측
print("model.score : ", model.score(x_test, y_test))     # Regressor : R2
print("best_score : ", model.best_score_)
print("걸린시간 : ", round(end - start, 2), "초")


# (442, 10) (442,)
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 파라미터 : {'min_child_weight': 10, 'learning_rate': 0.1}
# model.score :  0.3153844980290783
# best_score :  0.3595224474068816
# 걸린시간 :  5.41 초
# → 10 조합 × 5 fold = 50 번 훈련 (GridSearchCV m08_02 : 210 번, 8.16 초)
# → best_score 0.3595 로 GridSearchCV 의 0.3763 ({'learning_rate': 0.01, 'min_child_weight': 10}) 보다 조금 낮다
#   → 그 최적 조합이 무작위 10 개에 뽑히지 않았다 (min_child_weight=10 은 같지만 learning_rate 가 0.1)
