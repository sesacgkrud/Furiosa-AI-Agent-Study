# keras16_validation2.py
# [실습] 1 ~ 16 을 슬라이싱으로 train 8개 / val 4개 / test 4개로 자른다

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

# 실습 8개 4개 4개 자르기
x_train = x[:8]
y_train = x[:8]

x_val = x[8:12]
y_val = x[8:12]

x_test = x[12:]
y_test = x[12:]

# print(x_train.shape, x_val.shape, x_test.shape)
# exit()

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=4,
          validation_data=(x_val, y_val),
          )

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)