# m05_all_estimators1_kfold.py 베이스

# [목적]
# m05_1 의 all_estimators + KFold 교차 검증을 회귀 모델(regressor) 로 바꿔 diabetes 회귀 데이터에 적용한다
# - all_estimators(type_filter="regressor") · load_diabetes (m04_2 와 같은 데이터)
# - 회귀라서 train_test_split 의 stratify 를 뺐다 (y 가 연속값)
# - cross_val_score 는 fold 별 R2 를 반환하고, test 예측은 accuracy_score → r2_score 로 평가한다

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings("ignore")

#1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델 구성
all = all_estimators(type_filter="regressor")
print("sk version : ", sk.__version__)
print("모델의 갯수 : ", len(all))

for name, model in all:
    try:
        model = model()

        #3. 훈련, 평가 (교차 검증)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)     # fold 별 R2 5 개

        #4. 예측
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        r2 = r2_score(y_test, y_predict)

        print("==============", name, "==============")
        print("r2 : ", scores, "\n평균 r2 : ", round(np.mean(scores), 4))
        print("cross_val_predict r2 : ", round(r2, 4))

    except:
        print(name, "모델 오류")

# 결과 (55 개 중 43 개 성공 · 12 개 모델 오류, 평균 r2 순)
# ARDRegression                   평균 r2 : 0.5055    cross_val_predict r2 : 0.3934   ← 교차 검증 1 등
# Lasso                           평균 r2 : 0.5044    cross_val_predict r2 : 0.4058
# ElasticNetCV                    평균 r2 : 0.5033    cross_val_predict r2 : 0.4099
# PLSRegression                   평균 r2 : 0.5031    cross_val_predict r2 : 0.4277
# LinearRegression                평균 r2 : 0.4969    cross_val_predict r2 : 0.3797
# RandomForestRegressor           평균 r2 : 0.445     cross_val_predict r2 : 0.3823
# DecisionTreeRegressor           평균 r2 : -0.0827   cross_val_predict r2 : -0.1547
# KernelRidge                     평균 r2 : -3.5887   cross_val_predict r2 : -5.043   ← 가장 낮음
# → m04_2 (test 한 번 평가) 에서는 ElasticNet 이 1 등이었지만, 5 번 평균으로 보면 ARDRegression · Lasso 등이 앞선다
#   → 한 번 나눈 test 점수만 보고 모델을 고르면 운에 좌우될 수 있다
# → 상위권은 선형 모델(Lasso · Ridge · ElasticNet 계열) 이 차지하고 점수 차이도 0.01 이내로 작다
# → random_state 를 안 준 모델(트리 · MLP 등) 은 실행마다 조금씩 달라진다
#
# 모델 오류 12 개 : m04_2 와 같다 (MultiOutput · RegressorChain · Stacking · Voting · MultiTask 4 개 · CCA · PLSCanonical
#                  · IsotonicRegression · RadiusNeighborsRegressor)
