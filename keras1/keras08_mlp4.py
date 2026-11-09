# keras08_mlp4.py
# [목적] 입력 1개 -> 출력 3개 모델 : input_dim=1, 마지막 Dense(3)
# x.shape (10,) / y.shape (10, 3) 처럼 입력과 출력의 열 개수가 달라도 된다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
               [10,9,8,7,6,5,4,3,2,1,],
               [9,8,7,6,5,4,3,2,1,0]]).transpose()
print(x.shape, y.shape) # (10,) (10, 3)

#2. 모델 구성
## [실습]

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(3)) # output 3개

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=10)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss :", loss)
# x 10 하나를 넣으면 y 3 개 [11, 0, -1] 쪽 값이 나오는지 확인 (출력 문구의 [10, 0, -1] 은 입력이 아니라 기대값 메모)
result = model.predict(np.array([10]))
print("[10, 0, -1]의 예측값 :", result)

# 목표 : [11.00, 0.00, -1.00] 까지 합격 -> [10, 0, -1]의 예측값 : [[10.99978     0.01122689 -1.0102037 ]]
# loss : 4.762872777064331e-05