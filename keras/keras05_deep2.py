# keras05_deep2.py
# 두 번째 층부터는 input_dim 을 생략한다 (앞 층의 출력 개수를 케라스가 알아서 입력으로 잡는다)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델 구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(300)) # input_dim 생략 가능
model.add(Dense(1)) # input_dim 생략 가능

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=800)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss :", loss)
# result = model.predict(np.array([1,2,3,4,5]))
# print("예측값 :", result)

# 목표 : 0.31xx, 결과 : 0.3238