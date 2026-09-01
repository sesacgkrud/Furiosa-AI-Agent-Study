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
result = model.predict(np.array([10]))
print("[10, 0, -1]의 예측값 :", result)

# 목표 : [11.00, 0.00, -1.00] 까지 합격 -> [10, 0, -1]의 예측값 : [[10.99978     0.01122689 -1.0102037 ]]
# loss : 4.762872777064331e-05