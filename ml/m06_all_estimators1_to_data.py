# m05_all_estimators1_kfold.py 베이스

# [목적]
# m05_1 의 "all_estimators 분류 모델 전체 + 교차 검증" 을 데이터셋 4 개(iris · cancer · wine · digits) 에 한 번에 돌려
# 데이터마다 cross_val_predict acc 가 가장 높은 모델을 찾는다
# - datasets 리스트에 (x, y) 튜플을 담고, for index, value in enumerate(datasets) 로 데이터를 하나씩 꺼낸다
#   → data_name[index] 로 같은 순서의 데이터 이름을 출력한다
# - 모델 이름 · acc 를 리스트에 append 한 뒤 np.array 로 바꿔 np.argmax 로 최고 모델의 위치(index) 를 찾는다
# - KFold → StratifiedKFold : fold 마다 클래스 비율을 y 전체 비율과 같게 나눈다 (분류 데이터)
# - time.time() 으로 전체 걸린 시간을 잰다

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk
import time

import warnings
warnings.filterwarnings("ignore")

#1. 데이터
iris = load_iris(return_X_y=True)
# print(iris)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digits]
data_name = ['아이리스', '캔서', '와인', '디지트']

#2. 모델 구성
all = all_estimators(type_filter="classifier")

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

start = time.time()
for index, value in enumerate(datasets):
    x, y = value # x와 y 분리

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, train_size=0.8,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("==========", data_name[index], "==========")
    model_name = []
    model_acc = []

    for name, model in all:
        # print(model)
        try:
            model = model()

            #3. 훈련
            # model.fit(x_train, y_train)
            scores = cross_val_score(model, x_train, y_train, cv=kfold)

            #4. 평가
            y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

            acc = np.round(accuracy_score(y_test, y_pred), 4)
            model_name.append(name)
            model_acc.append(acc)

        except:
            # print(name, "모델 오류")
            continue

    model_name = np.array(model_name)
    model_acc = np.array(model_acc)
    max_index = np.argmax(model_acc)
    print("최고 모델 : ", model_name[max_index], "ACC : ", model_acc[max_index])


end = time.time()
print("걸린 시간 : ", round(end - start, 2), "초")

# 결과 (StratifiedKFold 로 수정 후, cross_val_predict acc 기준 최고 모델)
# ========== 아이리스 ==========
# 최고 모델 :  BaggingClassifier ACC :  0.9667
# ========== 캔서 ==========
# 최고 모델 :  AdaBoostClassifier ACC :  0.9825
# ========== 와인 ==========
# 최고 모델 :  NuSVC ACC :  1.0
# ========== 디지트 ==========
# 최고 모델 :  SVC ACC :  0.95
# 걸린 시간 :  90.14 초
# → random_state 를 안 준 모델(트리 · 앙상블 등) 은 실행마다 최고 모델이 바뀔 수 있다