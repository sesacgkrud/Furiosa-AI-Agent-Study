# [목적]
# 머신러닝 트리 계열 모델 4종으로 iris 를 분류하고, 모델마다 feature_importances_ (컬럼 중요도) 를 확인한다
#
#   DecisionTreeClassifier      : 결정 트리 1개
#   RandomForestClassifier      : 결정 트리 여러 개를 따로 만들어 투표 (배깅)
#   GradientBoostingClassifier  : 앞 트리의 오차를 다음 트리가 보완 (부스팅)
#   XGBClassifier               : 부스팅을 빠르게 개선한 xgboost 라이브러리 모델
#
# 흐름
# - 딥러닝(keras)과 달리 compile / epochs 없이 model.fit(x, y) 한 번으로 훈련한다
# - model.score(x_test, y_test) : 분류 모델은 accuracy, 회귀 모델은 R2 를 반환 (evaluate 대신 사용)
# - models 리스트에 4개 모델을 넣고 for 문으로 같은 데이터에 차례로 fit · score
# - model.__class__.__name__ : 모델 클래스 이름을 문자열로 출력

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)             # (150, 4) (150,)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=random_state,
    stratify=y,
)

#2. 모델 구성
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]

for model in models:
    #3. 컴파일, 훈련
    model.fit(x_train, y_train)
    print('====================', model.__class__.__name__, '====================')

    #4. 평가, 예측
    score = model.score(x_test, y_test) # Classifier 기준 : accuracy(정확도) 반환 / (참고 : Regressor(회귀)에서 score -> R2(결정계수) 반환)
    print('acc: ', score)
    print(model.feature_importances_)   # 모델별로 [0.02430454 0.02472077 0.7376847  0.21328993] 이런 형식으로 출력 -> 모두 합하면 1 -> iris의 컬럼 4개 -> 수치가 높으면 중요도 ↑

############### 모델별 개별 실행 ###############
# DecisionTreeClassifier : 0.8333333333333334
# RandomForestClassifier : 0.9333333333333333
# GradientBoostingClassifier : 0.9666666666666667
# XGBClassifier : 0.9333333333333333
# XGBClassifier : 1.0 (random_state 123 -> 12313 변경 후)

############### 모델별 전체 실행 ###############
# ==================== DecisionTreeClassifier ====================
# acc:  0.8333333333333334
# [0.0125     0.03       0.92133357 0.03616643]
# ==================== RandomForestClassifier ====================
# acc:  0.9333333333333333
# [0.08860798 0.0214775  0.48716455 0.40274997]
# ==================== GradientBoostingClassifier ====================
# acc:  0.9666666666666667
# [0.00157033 0.02147603 0.82483537 0.15211826]
# ==================== XGBClassifier ====================
# acc:  0.9333333333333333
# [0.02430454 0.02472077 0.7376847  0.21328993]
# → 4개 모델 모두 3번째 컬럼(petal length) 중요도가 가장 높다
