# keras54_RNN2_summary.py
# SimpleRNN 의 파라미터 개수를 summary 로 확인한다 - keras54_RNN1.py 베이스
#
# [ 왜 층을 줄였나 ]
#  RNN1 은 Dense 를 10층 쌓아서 summary 가 길다
#  여기서는 계산을 눈으로 따라가려고 SimpleRNN(5) + Dense(7) + Dense(1) 로 줄였다
#  훈련은 하지 않고 model.summary() 까지만 본다
#
# [ SimpleRNN 파라미터 계산식 ]
#  Param = (units * features) + (units * units) + (units * bias)
#   units * features : 입력값을 받는 가중치            (5 * 1 = 5)
#   units * units    : 앞 칸의 결과를 다시 받는 가중치  (5 * 5 = 25)  <- RNN 에만 있는 항
#   units * bias     : 편향                            (5 * 1 = 5)
#   -> 5 + 25 + 5 = 35
#  가운데 항(units * units)이 "앞에서 계산한 것을 다음 칸으로 넘긴다" 는 RNN 의 정체다
#  timesteps 는 식에 없다 : 같은 가중치를 timesteps 번 돌려 쓰기 때문에 3칸이든 100칸이든 35개다
#
# [ Dense 는 기존과 같다 ]
#  Dense(7)  : 앞 층 5 * 7 + 7 = 42
#  Dense(1)  : 앞 층 7 * 1 + 1 = 8

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
model.add(SimpleRNN(5, input_shape=(3, 1)))     # units 를 5 로 줄여서 35개가 나오는지 확인한다

# 3차원으로 들어가서 1(벡터) 또는 2(Metrics)차원으로 나옴 -> 바로 Dense와 연결 가능
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()
# Param의 갯수 = (unit * feature) + (units * bias) + (units * units)
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 5)                 35

#  dense (Dense)               (None, 7)                 42

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 85
# Trainable params: 85
# Non-trainable params: 0
# _________________________________________________________________

# [확인] Output Shape 이 (None, 5) 2차원이다.
#        (3, 1) 3차원으로 들어갔는데 timesteps 3 이 사라졌다
#        -> 3칸을 다 돌고 '마지막 결과' 만 내보내기 때문이다. 그래서 Dense 와 바로 연결된다.
