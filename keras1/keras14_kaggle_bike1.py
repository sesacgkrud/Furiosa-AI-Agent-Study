# keras14_kaggle_bike1.py
# 캐글 자전거 대여량 예측 : 따릉이와 같은 흐름이지만 x 를 만들 때 지워야 할 컬럼이 더 있다
# casual + registered = count 라서 둘을 x 에 남기면 정답을 그대로 알려 주는 셈이 된다 -> 3개를 같이 drop
# 제출용 test.csv 에는 이 두 컬럼이 아예 없어서 컬럼 수(8개)를 맞추려면 반드시 지워야 한다

# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

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

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   # axis=1 : 컬럼 방향으로 지운다
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
model.fit(x_train, y_train, epochs=1000, batch_size=16)

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

##### submission.csv 만들기 // count 컬럼에 값 넣기 #####
y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + "submit/" + "submit_0904_1620.csv")

########################################################################

# 1차 시도 (random_state=77, Dense=64,32,16,1, epochs=2000, batch_size=16)
# loss(mse) : 23562.330078125
# r2 : 0.2606160044670105
# mse : 23562.328125
# RMSE : 153.50025447861643

# 2차 시도 (random_state=77, Dense=64,32,1, epochs=1000, batch_size=32)
# loss(mse) : 20959.08203125
# r2 : 0.3423055410385132
# mse : 20959.0859375
# RMSE : 144.77253170922998

# 3차 시도 (random_state=100, Dense=32,16,1, epochs=1000, batch_size=16)
# loss(mse) : 22091.6875
# r2 : 0.3145309090614319
# mse : 22091.689453125
# RMSE : 148.63273345103022

# 4차 시도 (random_state=100, Dense=32,16,1, epochs=1000, batch_size=16) -> 마지막 relu 추가
# loss(mse) : 21878.287109375
# r2 : 0.3211525082588196
# mse : 21878.287109375
# RMSE : 147.9131066179566