# keras52_optimizer01_california.py
# 캘리포니아 집값 (회귀) - optimizer 의 learning_rate 를 직접 지정해 본다
# 스케일링은 keras27 그대로다 (split 전에 전체 x 로 fit 하는 방식이라 데이터 누수가 있다)
#
# [ 오늘 바뀌는 곳은 compile 한 줄이다 ]
#  before : model.compile(..., optimizer='adam')                  -> learning_rate 가 기본값 0.001 로 고정
#  after  : model.compile(..., optimizer=Adam(learning_rate=...))  -> 보폭을 내가 정한다
#
# [ learning_rate = 가중치를 한 번에 얼마나 크게 고칠지 정하는 보폭 ]
#  크면 (0.05 / 0.01)  : 최저점을 건너뛰고 그 주변에서 튕겨 다닌다
#  작으면 (0.00001)    : 한 걸음이 작아서 epoch 가 끝날 때까지 최저점에 닿지 못한다
#  데이터마다 맞는 값이 달라서 후보를 주석으로 적어 두고 하나씩 열어 가며 직접 비교했다
#  이 파일에서 고른 값 : learning_rate = 0.01
#
# keras27_Scaler01_california.py 베이스

from sklearn.datasets import fetch_california_housing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

'''
MinMaxScaler Algorithm
-> (원값 - Min) / (Max - Min)
-> 0 ~ 1 사이로 수렴
'''

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) # sklearn 에서 fit -> 실행하다 로 생각
x = scaler.transform(x) # x에 있는 모든 데이터는 0~1 사이로 수렴
print(x)
print('Min :', np.min(x), 'Max :', np.max(x)) # Min : 0.0 Max : 1.0000000000000002 -> 부동 소숫점 연산에 의한 오차 발생 (문제X)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.75,
    random_state=100,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
          validation_split=0.2)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

from sklearn.metrics import r2_score, mean_squared_error

print("loss(mse) :", loss)
y_predict = model.predict(x_test).ravel()

r2 = r2_score(y_test, y_predict)

print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# loss(mse) : 0.25228360295295715
# r2 : 0.8086916350028439
# mse : 0.2522835515735914
# RMSE : 0.5022783606463566

# ===== learning rate 변경 =====
# loss(mse) : 0.2797977328300476
# r2 : 0.7878274481224297
# mse : 0.2797977230891145
# RMSE : 0.5289590939657948

# [결론] 기본값 0.001 일 때 RMSE 0.5023 -> 0.01 로 키우니 0.5290 으로 나빠졌다.
#        보폭이 커서 최저점 근처에서 넘어 다닌 것이다. 키운다고 항상 좋아지지 않는다.
