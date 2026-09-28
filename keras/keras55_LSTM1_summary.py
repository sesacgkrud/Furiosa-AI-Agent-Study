# keras55_LSTM1_summary.py
# LSTM 의 파라미터 개수를 SimpleRNN 과 비교한다 - keras54_RNN2_summary.py 베이스
#
# [ 왜 LSTM 이 나왔나 ]
#  SimpleRNN 은 앞 칸의 결과를 계속 곱해 가며 뒤로 넘긴다
#  timesteps 가 길어지면 앞쪽에서 온 값이 점점 희미해져 사라진다 (장기 기억 문제)
#  LSTM 은 "계속 들고 갈 값(cell state)" 을 따로 두고,
#  게이트로 얼마나 버리고 / 받아들이고 / 내보낼지를 스스로 정한다
#
# [ 그래서 파라미터가 4배다 ]
#  SimpleRNN 한 덩어리 = (units * features) + (units * units) + units
#   units=10, features=1 -> (10 * 1) + (10 * 10) + 10 = 120
#  LSTM 은 그 덩어리가 4개다 (cell state 후보 + forget / input / output 게이트)
#   120 * 4 = 480       <- summary 의 480 과 일치한다
#
# [ 같은 units 면 항상 LSTM 이 RNN 의 4배 ]
#  파라미터가 4배라는 것은 계산도 그만큼 더 한다는 뜻이다
#  기억을 오래 들고 가는 대신 느려지는 것이 LSTM 의 값이다
#
# 훈련은 하지 않고 model.summary() 까지만 본다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

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
# SimpleRNN 을 LSTM 으로 바꿔 끼우기만 하면 된다 (입출력 모양이 같아서 뒤 코드는 그대로)
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))
# model.add(SimpleRNN(10, input_shape=(3, 1)))
# model.add(LSTM(10, input_shape=(3, 1)))
model.add(GRU(10, input_shape=(3, 1)))

# 3차원으로 들어가서 1(벡터) 또는 2(Metrics)차원으로 나옴 -> 바로 Dense와 연결 가능
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 10)                480

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 565
# Trainable params: 565
# Non-trainable params: 0
# _________________________________________________________________

# [비교] units=10 기준
#  SimpleRNN : (10 * 1) + (10 * 10) + 10       = 120
#  LSTM      : 120 * 4                         = 480
#  Dense(7)  : 앞 층 10 * 7 + 7                = 77   (RNN2 는 앞 층이 5라서 42였다)
#  Dense(1)  : 7 * 1 + 1                       = 8


# GRU Summary Result
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  gru (GRU)                   (None, 10)                390       
                                                                 
#  dense (Dense)               (None, 7)                 77        
                                                                 
#  dense_1 (Dense)             (None, 1)                 8         
                                                                 
# =================================================================
# Total params: 475
# Trainable params: 475
# Non-trainable params: 0