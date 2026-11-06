# m09_RandomSearchCV_00.py 베이스

# [목적]
# m09_00 의 RandomizedSearchCV + XGBClassifier 를 iris 대신 digits (손글씨 숫자 0 ~ 9, 10 클래스) 에 적용한다
# - 데이터만 load_iris → load_digits 로 바꾸고, parameters · StratifiedKFold · RandomizedSearchCV 설정은 m09_00 과 같다
# - 42 개 조합 중 n_iter (기본값 10) 개만 무작위로 뽑아 교차 검증 → GridSearchCV (m08_01) 와 점수 · 시간을 비교한다

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
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
print("model.score : ", model.score(x_test, y_test))
print("best_score : ", model.best_score_)
print("걸린시간 : ", round(end - start, 2), "초")


# (1797, 64) (1797,)
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 파라미터 : {'n_estimators': 100, 'max_depth': 12, 'learning_rate': 0.1}
# model.score :  0.9416666666666667
# best_score :  0.958947928765002
# 걸린시간 :  7.5 초
# → 10 조합 × 5 fold = 50 번 훈련으로, GridSearchCV (m08_01 : 210 번, 18.52 초) 보다 빠르다
# → best_score 0.9589 로 GridSearchCV 의 0.961 (n_estimators=500 · max_depth=6) 과 거의 비슷하다
#   → 전체 조합을 다 돌지 않아도 비슷한 수준의 조합을 더 짧은 시간에 찾을 수 있다