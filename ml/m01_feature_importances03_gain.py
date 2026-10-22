# m01_feature_importances01.py 베이스

# [목적]
# XGBClassifier 하나만 남겨, xgboost 의 중요도 계산 방식 2가지 (F score / Gain score) 를 구분한다
#   - F score (weight) : 그 컬럼이 트리 분기에 쓰인 횟수 (빈도수)
#   - Gain score (gain) : 그 컬럼으로 분기했을 때 손실함수가 개선된 정도
# - model.get_booster() : sklearn 형식 모델 안의 xgboost 원래 모델(Booster) 을 꺼낸다
# - get_score(importance_type='gain') : 컬럼별 점수를 {'f0': ..., 'f1': ...} 딕셔너리로 반환 (f0 = 0번째 컬럼)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=random_state,
    stratify=y,
)

#2. 모델 구성
model = XGBClassifier(random_state=random_state)

#3. 컴파일, 훈련
model.fit(x_train, y_train)
print('====================', model.__class__.__name__, '====================')

#4. 평가, 예측
score = model.score(x_test, y_test)
print('acc: ', score)
print('feature_importances_: ', model.feature_importances_)
# XGBoost 의 feature_importances_ = Gain score 를 합이 1 이 되게 나눈 값 (F score 가 아니다)
#   예) f0 : 0.0928 / (0.0928 + 0.0944 + 2.8167 + 0.8144) = 0.0243

gain_score = model.get_booster().get_score(importance_type='gain')
print('Gain score: ', gain_score)
# Gain 스코어 : 손실함수의 개선 점수. 높으면 개선 많이 된 것!!

# F 스코어 : 빈도수 (분기에 쓰인 횟수) → importance_type='weight'
#   m01_04 의 plot_importance(model, importance_type='weight') 그래프 : f0 63 / f1 66 / f2 145 / f3 83
#   빈도수로는 f0 · f1 도 꽤 쓰이지만, Gain 으로 보면 f2 (petal length) 의 기여가 압도적이다

# ==================== XGBClassifier ====================
# acc:  0.9333333333333333
# feature_importances_:  [0.02430454 0.02472077 0.7376847  0.21328993]
# Gain score:  {'f0': 0.09280245006084442, 'f1': 0.09439174085855484, 'f2': 2.8167145252227783, 'f3': 0.8144086599349976}
