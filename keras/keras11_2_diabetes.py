# keras11_2_diabetes.py
# 당뇨 수치 데이터(회귀) : 특성 10개로 1년 뒤 병의 진행도를 예측한다
# 데이터가 442개로 적어서 random_state 에 따라 loss 가 크게 흔들린다

from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=42,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=10))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2000, batch_size=16)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss :", loss)

# loss : 2662.560546875
# loss : 2879.285888671875

# loss : 3111.257080078125
# loss : 1865.6835937