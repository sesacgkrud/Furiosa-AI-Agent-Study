# m03_kfold_train_test_split00_iris.py 베이스
# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

# [목적]
# m03_00 의 train_test_split + KFold 교차 검증을 Kaggle Santander (고객 거래 여부 이진 분류) 에 적용한다
# - 데이터 불러오기 · 모델은 m02_07 과 같다 (target 을 x 에서 빼고, XGBClassifier)
# - 분류라서 stratify=y 를 그대로 둔다 (0 : 약 90% / 1 : 약 10% 비율을 train / test 에 똑같이)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#1. 데이터
path = "./_data/kaggle_santander/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델 구성
model = XGBClassifier(random_state=333)

#3. 훈련, 평가
scores = cross_val_score(model, x_train, y_train, cv=kfold)

print("acc : ", scores, "\n평균 acc : ", round(np.mean(scores), 4))
# acc :  [0.9120625  0.91334375 0.9120625  0.91346875 0.9133125 ]
# 평균 acc :  0.9128

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_test)
print(y_predict)
# train_58890 0, train_77217 0, ... (Length: 40000) -> 원값
# [0 0 0 ... 0 0 0] -> 예측값 (대부분 0 으로 예측)

acc = accuracy_score(y_test, y_predict)
print("cross_val_predict acc : ", round(acc, 4))
# cross_val_predict acc :  0.908 → 전부 0 으로 찍었을 때 (0.8995) 보다 조금 높다
