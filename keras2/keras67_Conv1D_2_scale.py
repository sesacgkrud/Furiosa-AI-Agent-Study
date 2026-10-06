# keras55_LSTM2_scale.py 베이스

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60],
             ])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape) # (13, 3) (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)
y = y.reshape(y.shape[0], 1)
print(x.shape) # (13, 3, 1)
print(y.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, padding='same', input_shape=(3, 1)))   # (None, 3, 10)  param (2×1+1)×10 = 30
model.add(Conv1D(10, 2, padding='same'))                                            # (None, 3, 10)  param (2×10+1)×10 = 210
# Flatten 은 Conv1D 바로 뒤에 한 번만 쓴다
#   (Flatten 전에 Dense 를 쌓으면 3차원 그대로 시점마다 따로 계산되고, 2차원에서 한 번 더 Flatten 하는 건 의미가 없다)
model.add(Flatten())                                                                # (None, 30)

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss :', results)

x_predict = np.array([50, 60, 70]).reshape(1, 3, 1) # 80 맞춰보기
y_predict = model.predict(x_predict)

print('[80] 예측 결과', y_predict)

# 목표 : 79.5
# loss : 0.115653857588768
# [80] 예측 결과 [[78.97944]]

########## Conv1D 적용 ##########
# 목표 : 80
# loss : 0.0001750103838276118
# [80] 예측 결과 [[80.00446]]