# keras71_trainable_weights2.py 베이스
# [변경] 데이터를 x=1, y=1 하나로 줄이고 동결 X (trainable=True) 로 1000 epoch 훈련한 뒤,
#        model.weights 로 꺼낸 가중치만으로 예측값을 손으로 계산해 본다 (맨 아래 수식 주석)
#  - 층도 keras71_trainable_weights1 과 같은 Dense 3 → 2 → 1 로 줄여서 계산을 따라가기 쉽게 했다

import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

print(tf.__version__)

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
x = np.array([1,])
y = np.array([1,])

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

######################## 동결 ########################
# model.trainable = False
model.trainable = True
######################## 동결 ########################


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=1)

#4. 평가, 예측
y_predict = model.predict(x)
# print(y_predict)    # [[1.0002013]] ← 아래 수식으로 계산한 값과 같다

print('=============================================')
print(model.weights)
print('=============================================')

############## model.weights 로 직접 계산해보기 ##############
# Dense 연산 : y = x · W + b   (activation 없음 = linear)
#
# [model.weights 출력값]
# dense/kernel   (1, 3) : [[ 0.08450541, -0.08842915,  0.8283203 ]]
# dense/bias     (3,)   :  [-0.05152767, -0.05362988,  0.05395508]
# dense_1/kernel (3, 2) : [[-0.9780324 ,  0.7826826 ],
#                          [ 0.6976057 ,  1.0858648 ],
#                          [-0.5848015 , -0.68448645]]
# dense_1/bias   (2,)   :  [-0.06430405, -0.05273598]
# dense_2/kernel (2, 1) : [[-0.08098985],
#                          [-1.1340631 ]]
# dense_2/bias   (1,)   :  [ 0.05222915]
#
# 입력 x = 1
#
# 1) 첫 번째 Dense(3) : h1 = x · W1 + b1
#    h1_1 = 1 * 0.08450541    + (-0.05152767) =  0.03297774
#    h1_2 = 1 * (-0.08842915) + (-0.05362988) = -0.14205903
#    h1_3 = 1 * 0.8283203     + 0.05395508    =  0.88227538
#    h1 = [0.03297774, -0.14205903, 0.88227538]
#
# 2) 두 번째 Dense(2) : h2 = h1 · W2 + b2
#    h2_1 = 0.03297774 * (-0.9780324)          = -0.03225330
#         + (-0.14205903) * 0.6976057          = -0.09910119
#         + 0.88227538 * (-0.5848015)          = -0.51595598
#         + (-0.06430405)                       (bias)
#         = -0.71161452
#    h2_2 = 0.03297774 * 0.7826826             =  0.02581110
#         + (-0.14205903) * 1.0858648          = -0.15425690
#         + 0.88227538 * (-0.68448645)         = -0.60390556
#         + (-0.05273598)                       (bias)
#         = -0.78508734
#    h2 = [-0.71161452, -0.78508734]
#
# 3) 세 번째 Dense(1) : y = h2 · W3 + b3
#    y = (-0.71161452) * (-0.08098985)          =  0.05763356
#      + (-0.78508734) * (-1.1340631)           =  0.89033859
#      + 0.05222915                              (bias)
#      = 1.0002013   ≒ 1   (y = 1 에 맞게 훈련됨)
#
# 한 줄 수식 : y = ((x·W1 + b1)·W2 + b2)·W3 + b3
# -> activation 이 없으면 층을 여러 개 쌓아도 결국 y = a·x + c 형태의 1차 함수
#    a = W1·W2·W3                = 0.72779308
#    c = b1·W2·W3 + b2·W3 + b3   = 0.27240818
#    y = 0.72779308 * 1 + 0.27240818 = 1.0002013
