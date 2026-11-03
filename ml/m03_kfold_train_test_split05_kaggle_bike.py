# m03_kfold_train_test_split01_california.py 베이스
# https://www.kaggle.com/competitions/bike-sharing-demand/data

# [목적]
# m03_01 의 train_test_split + KFold 교차 검증을 Kaggle Bike Sharing (자전거 대여량 회귀) 에 적용한다
# - 데이터 불러오기 · 모델은 m02_05 와 같다 (casual · registered · count 를 x 에서 빼고, RandomForestRegressor)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#1. 데이터
path = "./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델 구성
model = RandomForestRegressor(random_state=333)

#3. 훈련, 평가
scores = cross_val_score(model, x_train, y_train, cv=kfold)

print("r2 : ", scores, "\n평균 r2 : ", round(np.mean(scores), 2))
# r2 :  [0.28966171 0.28584383 0.33420566 0.28991125 0.29994887]
# 평균 r2 :  0.3

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_test)
print(y_predict)
# 2011-07-14 21:00:00 261, 2012-08-02 06:00:00 183, ... (Length: 2178) -> 원값 (datetime index 가 붙은 Series)
# [117.38  192.06  355.11  ...  261.89  161.391  406.22233333] -> 예측값

r2 = r2_score(y_test, y_predict)
print("cross_val_predict r2 : ", round(r2, 4))
# cross_val_predict r2 :  0.1943 → x_test 2178 개만으로 다시 훈련 · 예측해서 train 교차 검증(0.3) 보다 낮다
