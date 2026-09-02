import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

## [찾아보기] Numpy List의 슬라이싱 => 7:3 으로 나누자

x_train = x[:7]
y_train = y[:7]

x_test = x[7:]
y_test = y[7:]

print("x_train :", x_train)
print("x_test :", x_test)

print("y_train :", y_train)
print("y_test :", y_test)

##### 이하 keras09_train_test1.py 동일 #####

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=4)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test) # 검증을 테스트 데이터로 평가
print("loss :", loss)

# loss : 0.04448358342051506