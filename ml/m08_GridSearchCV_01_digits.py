# m08_GridSearchCV_00.py 베이스

# [목적]
# m08_00 의 GridSearchCV + XGBClassifier 를 iris 대신 digits (손글씨 숫자 0 ~ 9, 10 클래스) 에 적용한다
# - 데이터만 load_iris → load_digits 로 바꾸고, parameters · StratifiedKFold · GridSearchCV 설정은 m08_00 과 같다
# - digits 는 (1797, 64) 로 iris (150, 4) 보다 데이터 · 컬럼 · 클래스가 많아서 같은 210 번 훈련이라도 더 오래 걸린다 (iris 4.39 초 → digits 18.52 초)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
x, y = load_digits(return_X_y=True)
print(x.shape, y.shape)     # (1797, 64) (1797,)

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
print("model.score : ", model.score(x_test, y_test))
print("best_score : ", model.best_score_)
print("걸린시간 : ", round(end - start, 2), "초")


# (1797, 64) (1797,)
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# 최적의 파라미터 :  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
# model.score :  0.9472222222222222
# best_score :  0.9610288424312815
# 걸린시간 :  18.52 초
# → 최적 조합은 iris (m08_00) 와 같다 (첫 번째 dict, learning_rate=0.1 · max_depth=6 · n_estimators=500)
# → m07 의 for 문 튜닝 (HistGradientBoosting, test 한 번 평가) 최고 0.9694 와 비슷한 수준
