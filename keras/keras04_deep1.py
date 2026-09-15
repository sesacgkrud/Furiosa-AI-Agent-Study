# keras04_deep1.py
# 딥러닝(deep) : Dense 층을 여러 개 쌓아서 입력 -> 은닉층 -> 출력으로 연결한다
# 앞 층의 출력 개수가 다음 층의 입력 개수가 된다 (500 -> 300 -> 1)

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