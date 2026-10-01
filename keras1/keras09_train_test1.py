# keras09_train_test1.py
# train / test 분리 : 훈련은 x_train 으로만 하고, 평가는 한 번도 안 본 x_test 로 한다
# 훈련에 쓴 데이터로 평가하면 외운 답을 다시 맞히는 셈이라 성능을 제대로 알 수 없다
# 여기서는 7 : 3 을 손으로 잘라서 나눈다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test) # 검증을 테스트 데이터로 평가
print("loss :", loss)