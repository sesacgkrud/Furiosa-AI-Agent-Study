# keras11_1_california.py
# 사이킷런이 주는 실제 데이터(캘리포니아 집값)로 처음 훈련해 본다 (회귀 : 집값이라는 숫자를 맞힌다)
# datasets.data = x(특성 8개), datasets.target = y(정답) -> x 의 열 개수가 곧 input_dim

from sklearn.datasets import fetch_california_housing

# import 가 안 되는 경우
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=100,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(75))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=32)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss :", loss)

# loss : 0.5886844992637634
# loss : 0.6215315461158752