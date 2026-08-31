from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델 구성
model = Sequential()

# model.add(Dense(3, input_dim=1)) # 입력:1, 출력:3 의 의미
# model.add(Dense(5, input_dim=3)) # 입력:3, 출력:5 의 의미
# model.add(Dense(4, input_dim=5)) # 입력:5, 출력:4 의 의미
# model.add(Dense(1, input_dim=4)) # 입력:4, 출력:1 의 의미

model.add(Dense(500, input_dim=1))
model.add(Dense(300, input_dim=500))
model.add(Dense(1, input_dim=300))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=900)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :", loss)
result = model.predict(np.array([1,2,3,4,5]))
print("예측값 :", result)

# 목표 : 0.37xx, 결과 : 0.379999