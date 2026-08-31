from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=1))
model.add(Dense(250)) # input_dim 생략 가능
model.add(Dense(1)) # input_dim 생략 가능

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=800, batch_size=3) # 3개씩 잘라서 훈련, 결과에서 2/2 -> 1 epochs를 2번으로 나눠 훈련했다는 의미
model.fit(x, y, epochs=800, batch_size=3) # 4개씩 잘라서 훈련, 결과에서 2/2 -> 1 epochs를 2번으로 나눠 훈련했다는 의미 (4개, 2개)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :", loss)
# result = model.predict(np.array([1,2,3,4,5]))
# print("예측값 :", result)

# 목표 : 0.3xxx, 결과 : 0.33325