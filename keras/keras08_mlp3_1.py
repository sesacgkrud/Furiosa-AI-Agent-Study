# keras08_mlp3_1.py
# 출력(y)이 2개인 다중 출력 모델 : y.shape 가 (10, 2) 이면 마지막 Dense 도 2 여야 한다
# 입력 개수는 input_dim, 출력 개수는 마지막 Dense 의 숫자로 맞춘다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
               [10,9,8,7,6,5,4,3,2,1,]]).transpose()
print(x.shape, y.shape) # (10, 3) (10, 2)

#2. 모델 구성
## [실습]

model = Sequential()
model.add(Dense(20, input_dim=3))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2)) # output 2개

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=10)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss :", loss)
result = model.predict(np.array([[10, 31, 211]]))
print("[10, 31, 211]의 예측값 :", result)

# 목표 : [11.00, 0.00] 까지 합격 -> [10, 31, 211]의 예측값 : [[10.906576   0.0132049]]
# loss : 0.001240719691850245