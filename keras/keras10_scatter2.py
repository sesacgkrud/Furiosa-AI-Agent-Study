# keras10_scatter2.py
# keras10_scatter1 보다 데이터를 20개로 늘리고 더 흩어지게 만들어서 예측선을 다시 그려 본다
# 데이터가 흩어질수록 loss 는 커지고, 모델은 점들의 가운데를 지나는 선을 찾는다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=321,     # train_size 를 안 쓰면 기본값 0.75 로 나뉜다
)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=4)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss :", loss)
result = model.predict(np.array([21]))
print("21의 예측값 :", result)

result = model.predict(x)

# 그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, result, color='red')
plt.show()

# loss : 8.1221923828125
# loss : 10.929346084594727
# 21의 예측값 : [[19.900915]]