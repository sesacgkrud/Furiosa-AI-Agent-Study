# keras03.py
# y 값을 [1,2,4,3,5] 로 바꿔서 일직선으로 못 맞추는 데이터를 준다 -> loss 가 0 까지 내려가지 않는다
# 이럴 때는 epochs 를 늘려가며 loss 를 목표치까지 낮추는 연습을 한다

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5]) # y 값 변경

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=900)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :", loss)
result = model.predict(np.array([1,2,3,4,5]))
print("예측값 :", result)

# 목표 : 0.35 이하, 결과 : 0.3800