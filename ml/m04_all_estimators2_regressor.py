# m04_all_estimators1_classifier.py 베이스

# [목적]
# m04_1 의 all_estimators 를 회귀 모델(regressor) 로 바꿔, sklearn 회귀 모델 전체를 diabetes 회귀 데이터에 돌려 본다
# - all_estimators(type_filter="regressor") : 회귀 모델 (이름, 클래스) 목록
# - 회귀 데이터라서 iris → load_diabetes, train_test_split 의 stratify 를 뺐다 (y 가 연속값)
# - 회귀 모델의 model.score 는 정답률이 아니라 R2(결정계수) 를 반환한다 → 음수도 나올 수 있다

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

#2. 모델 구성
# all = all_estimators(type_filter="classifier")
# print('all models : ', all)
# print("sk version : ", sk.__version__)          # sk version :  1.9.1
# print("모델의 갯수 : ", len(all))                 # 모델의 갯수(classifier) :  44

all = all_estimators(type_filter="regressor")
print('all models : ', all)
print("sk version : ", sk.__version__)          # sk version :  1.9.1
print("모델의 갯수 : ", len(all))                 # 모델의 갯수(regressor) :  55

for name, model in all:
    # print(name)
    # print(model)
    try:
        model = model()

        #3. 훈련
        model.fit(x_train, y_train)

        #4. 평가
        r2 = model.score(x_test, y_test)        # Regressor : R2
        print(name, "의 r2 : ", r2)

    except:
        print(name, "모델 오류")
# 결과 (55 개 중 43 개 성공 · 12 개 모델 오류)
# ElasticNet 의 r2 :  0.4874651287870655                ← 가장 높음
# TweedieRegressor 의 r2 :  0.47543852528693953
# LarsCV 의 r2 :  0.46696200554944733
# PLSRegression 의 r2 :  0.46403033441291164
# LinearRegression 의 r2 :  0.458616839855542
# RandomForestRegressor 의 r2 :  0.368474879049122
# DummyRegressor 의 r2 :  -0.0017077484775420704        ← y 평균으로만 찍는 기준 모델
# DecisionTreeRegressor 의 r2 :  -0.17807140937930188
# KernelRidge 의 r2 :  -3.712927514678162               ← 가장 낮음
# → diabetes 는 선형 모델(ElasticNet · Lars · Ridge 등) 이 트리 모델보다 높다
# → R2 는 y 평균으로만 찍는 것(DummyRegressor ≈ 0) 보다 못하면 음수가 나온다
# → random_state 를 안 준 모델(트리 · MLP 등) 은 실행마다 조금씩 달라진다
#
# 모델 오류 12 개 (try / except 로 건너뜀)
# - MultiOutputRegressor · RegressorChain · StackingRegressor · VotingRegressor : 안에 넣을 모델(estimator) 을 인자로 줘야 한다
# - MultiTaskElasticNet(CV) · MultiTaskLasso(CV) : y 가 2 차원(출력 여러 개) 이어야 한다
# - CCA · PLSCanonical : 기본 n_components=2 인데 y 가 1 개라 맞지 않는다
# - IsotonicRegression : x 가 컬럼 1 개(1 차원) 일 때만 동작한다
# - RadiusNeighborsRegressor : 반경(radius=1.0) 안에 이웃이 없는 test 데이터가 있어 오류
