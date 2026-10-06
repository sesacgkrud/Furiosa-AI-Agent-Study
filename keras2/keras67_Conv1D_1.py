# keras54_RNN1.py 베이스

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.layers import Conv1D, Flatten, GlobalAveragePooling1D

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])    # 원본 시계열 (x, y 로 자르기 전)

# 한 칸씩 밀면서 3개 묶음을 만든다 -> 마지막 묶음이 [7,8,9] 라 7줄이 나온다
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])

y = np.array([4,5,6,7,8,9,10])           # 각 묶음의 '다음 값' 이 정답이다

print(x.shape, y.shape)                  # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1) # 2차원에서 3차원으로 변환
print(x.shape)                           # (7, 3, 1) -> (N, timesteps, features)

#2. 모델 구성
model = Sequential()
# padding='same' : 앞뒤를 0 으로 채워서 timesteps 3 을 그대로 유지한다 (valid 면 3 -> 2 -> 1 로 줄어든다)
model.add(Conv1D(filters=10, kernel_size=2, padding='same', input_shape=(3, 1)))   # (None, 3, 10)  param (2×1+1)×10 = 30
model.add(Conv1D(10, 2, padding='same'))                                            # (None, 3, 10)  param (2×10+1)×10 = 210
model.add(Flatten())                                                                # (None, 30)

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))     # 숫자 하나를 맞히는 회귀라 출력층은 1칸, activation 없음

model.summary()

#3. 컴파일, 훈련
# 데이터가 7줄뿐이라 train_test_split / validation 없이 전부 훈련에 쓴다
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000,)   # 520 epoch 은 11.0 ~ 11.3 으로 흔들려서 늘렸다

#4. 평가, 예측
# 훈련에 쓴 데이터로 그대로 평가한다 (7줄짜리 연습용이라 나눌 것이 없다)
results = model.evaluate(x, y)
print('loss :', results)

# 예측도 훈련 때와 같은 3차원 모양으로 넣어야 한다
#  [8,9,10] 1묶음 / 3칸 / 값 1개 -> (1, 3, 1)
x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과 :', y_predict)

# loss : 5.951385173830204e-05
# 1/1 [==============================] - 0s 105ms/step
# [8,9,10]의 결과 : [[10.846822]]

# [결론] 정답 11 에 못 미치는 10.85 가 나왔다.
#        훈련 데이터의 y 가 4 ~ 10 까지뿐이라 모델은 10 보다 큰 값을 본 적이 없다.
#        RNN 이 규칙(+1)을 이해한 게 아니라 본 범위 안에서 흉내 낸 것에 가깝다.

########## 결과 ########## <- Conv1D 적용
# [8,9,10]의 결과 : [[11.009894]]