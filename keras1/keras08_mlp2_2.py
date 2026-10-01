# keras08_mlp2_2.py
# range() 로 데이터를 만드는 연습 : np.array(range(1, 11)) -> 1 ~ 10
# [실습] 컬럼 3개를 넣어 [10, 31, 211] 의 예측값이 11 에 가까워지게 만든다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10))
# print(x) # [0 1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 10))
# print(x) # [1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 11))
# print(x) # [ 1  2  3  4  5  6  7  8  9 10]

# x = np.array([range(10), range(21, 31,), range(201, 211)])
# print(x.shape) # (3, 10)

x = np.array([range(10), range(21, 31,), range(201, 211)]).T
# print(x.shape) # (10, 3)

y = np.array(range(1, 11))
# print(y)
# print(y.shape) # (10,)

#2. 모델 구성
## [실습]
## [10, 31, 211]
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=5)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss :", loss)
result = model.predict(np.array([[10, 31, 211]]))
print("[10, 31, 211]의 예측값 :", result)

# 목표 : 11.00까지 합격 -> [10, 31, 211]의 예측값 : [[10.999958]]
# loss : 1.1829058843559892e-09