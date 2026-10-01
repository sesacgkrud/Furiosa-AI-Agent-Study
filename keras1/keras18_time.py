# keras18_time.py
# 훈련에 걸린 시간 재기 : fit 앞뒤에서 time.time() 을 찍고 그 차이를 본다
# 같은 데이터라도 batch_size / epochs / verbose 에 따라 시간이 달라진다

# keras14_kaggle_bike1.py 베이스

# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

#1. 데이터
path = "./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv) # [10886 rows x 11 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [6493 rows x 8 columns]

submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
# print(submission) # [6493 rows x 1 columns]

# print(train_csv.shape) # (10886, 11)
# print(test_csv.shape) # (6493, 8)
# print(submission.shape) # (6493, 1)

# print(train_csv.info())
# print(test_csv.info())
# print(train_csv.describe())

#################### 결측치 확인 ####################

# print(train_csv.isna().sum()) # 컬럼별 결측치 수 합계 출력
# print(train_csv.isnull().sum()) # 위와 동일

# print(test_csv.isna().sum())
# print(test_csv.isnull().sum())

#################### x, y 분리 ####################

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# print(x) # [10886 rows x 8 columns]

y = train_csv['count']
print(y, y.shape) # Name: count, Length: 10886, dtype: int64 (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=100,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu')) # 통상적으로 마지막에는 relu 넣지 않음

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time() # 현재 시간 반환(=시작 시간)
model.fit(x_train, y_train, epochs=2, batch_size=16)
end_time = time.time() # 현재 시간 반환(=종료 시간)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
loss = model.evaluate(x_test, y_test, )

print("loss(mse) :", loss)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
(print("mse :", mse))

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

print("소요 시간 :", round(end_time - start_time, 2), "초")

'''
##### submission.csv 만들기 // count 컬럼에 값 넣기 #####
y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + "submit/" + "submit_0904_1620.csv")
'''