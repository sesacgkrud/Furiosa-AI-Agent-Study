# keras54_RNN3_input_length.py
# 입력 모양을 적는 두 가지 방법 - keras54_RNN1.py 베이스
#
# [ 같은 뜻을 두 가지로 쓸 수 있다 ]
#  input_shape=(3, 1)                 -> (timesteps, features) 를 묶어서 한 번에
#  input_length=3, input_dim=1        -> 나눠서 각각
#   input_length = timesteps (몇 칸을 보는가)
#   input_dim    = features  (한 칸에 값이 몇 개인가)
#  Dense 에서 쓰던 input_dim 과 이름이 같다. RNN 은 거기에 input_length 가 하나 더 붙는 것이다
#
# [ 둘은 순서를 바꿔 써도 된다 ]
#  input_length=3, input_dim=1  /  input_dim=1, input_length=3  둘 다 동작한다
#  (키워드로 이름을 붙여 넘기기 때문에 순서가 상관없다)
#
# 모델 구조는 keras54_RNN1 과 완전히 같다. 표기법만 바꿔 본 파일이다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)                  # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1) # 2차원에서 3차원으로 변환
print(x.shape)                           # (7, 3, 1)

#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))
# model.add(SimpleRNN(10, input_shape=(3, 1)))
model.add(SimpleRNN(units=10, input_length=3, input_dim=1))   # 위와 동일 (나눠서 입력 가능)
# model.add(SimpleRNN(units=10, input_dim=1, input_length=3)) # 순서 바꿀 수 있음


# 3차원으로 들어가서 1(벡터) 또는 2(Metrics)차원으로 나옴 -> 바로 Dense와 연결 가능
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=520,)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss :', results)

x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과 :', y_predict)

# loss : 5.951385173830204e-05
# 1/1 [==============================] - 0s 105ms/step
# [8,9,10]의 결과 : [[10.846822]]
