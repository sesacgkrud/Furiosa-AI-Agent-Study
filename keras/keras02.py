# from tensorflow.python.keras.models import Sequential # python을 입력하면 노란 줄이 없어지는데, python 버전에 따라 실행하면 에러가 발생하기도 함
# from tensorflow.python.keras.layers import Dense # python을 입력하면 노란 줄이 없어지는데, python 버전에 따라 실행하면 에러가 발생하기도 함

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x, y) # loss='mse'의 값
print("loss :", loss)
result = model.predict(np.array([1,2,3,4,5,6,7]))
print("7의 예측값 :", result)