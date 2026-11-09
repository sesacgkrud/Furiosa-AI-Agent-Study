# m06_all_estimators1_to_data.py 베이스

# [목적]
# m06_1 의 "여러 데이터셋 × all_estimators 전체 모델" 비교를 회귀 모델(regressor) 로 바꿔
# california · diabetes 회귀 데이터에 적용하고, 데이터별 최고 모델을 찾는다
# - all_estimators(type_filter="regressor") · fetch_california_housing · load_diabetes
# - 회귀라서 y 가 연속값 → train_test_split 의 stratify 를 빼고, StratifiedKFold 대신 KFold 를 쓴다
#   (StratifiedKFold 는 클래스 비율을 맞추는 방식이라 연속값 y 에는 쓸 수 없다)
# - 평가는 accuracy_score → r2_score

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn.utils import all_estimators
import sklearn as sk
import time

import warnings
warnings.filterwarnings("ignore")

#1. 데이터
california = fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)

datasets = [california, diabetes]
data_name = ['캘리포니아', '디아벳']

#2. 모델 구성
all = all_estimators(type_filter="regressor")

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

start = time.time()
for index, value in enumerate(datasets):
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, train_size=0.8,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("==========", data_name[index], "==========")
    model_name = []
    model_r2 = []

    for name, model in all:
        try:
            model = model()

            #3. 훈련
            scores = cross_val_score(model, x_train, y_train, cv=kfold)    # fold 별 R2 5 개

            #4. 평가
            y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

            r2 = np.round(r2_score(y_test, y_pred), 4)
            model_name.append(name)
            model_r2.append(r2)

        except:
            continue

    model_name = np.array(model_name)
    model_r2 = np.array(model_r2)
    max_index = np.argmax(model_r2)
    print("최고 모델 : ", model_name[max_index], "R2 : ", model_r2[max_index])


end = time.time()
print("걸린 시간 : ", round(end - start, 2), "초")

# 결과 (cross_val_predict R2 기준 최고 모델)
# ========== 캘리포니아 ==========
# 최고 모델 :  HistGradientBoostingRegressor R2 :  0.7947
# ========== 디아벳 ==========
# 최고 모델 :  GammaRegressor R2 :  0.4407
# 걸린 시간 :  3747.97 초 (약 62 분)
# → 데이터가 큰 california(20640 행) 에서 55 개 모델을 모두 교차 검증하느라 전체 시간이 길어졌다 (diabetes 는 442 행)
# → california 는 트리 부스팅 모델, diabetes 는 선형 계열 모델이 가장 높다
