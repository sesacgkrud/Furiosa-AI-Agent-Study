# keras69_ensemble3.py 베이스 (입력 1개, 출력 2개)

# [목적] x2, x3 데이터셋을 빼고 x1 하나만 넣는 단일 모델의 끝을 두 갈래로 나눠 y1 · y2 를 함께 예측한다
#   - 입력이 하나라 모델 병합(Concatenate) 이 필요 없다 -> 단일 모델의 출력(output1) 을 분기 두 곳의 입력으로 가져간다
#   - Model(inputs=input1, ...) 처럼 입력은 리스트 없이, 출력은 outputs=[...] 리스트로
#   - fit · evaluate 의 x 는 그대로, y 는 [y1, y2] 리스트로 넣는다
#   - Total params 1,222 (han1 ~ han4 1,035 + ld1 60 + ld2 110 + last1 11 + last2 6)

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T

y1 = np.array(range(3001, 3101))

y2 = np.array(range(13001, 13101))

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, y1, y2,
    train_size=0.7,
    random_state=77,
)
print(x1_train.shape, y1_train.shape, y2_train.shape)
print(x1_test.shape, y1_test.shape, y2_test.shape)

#2-1. 모델1 (단일 모델)
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)

#2-2. 분기1 (모델1 의 출력 output1 을 입력으로 가져온다)
last_dense1 = Dense(10, name='ld1')(output1)
last_dense2 = Dense(10, name='ld2')(last_dense1)
last_output1 = Dense(1, name='last1')(last_dense2)

#2-3. 분기2 (같은 output1 을 입력으로 가져온다, 레이어 구성 안 하는 경우)
last_output2 = Dense(1, name='last2')(output1)

#2-4. 모델 구성
model = Model(inputs=input1, outputs=[last_output1, last_output2])
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=8)

#4. 평가, 예측
result = model.evaluate(x1_test, [y1_test, y2_test])
print('loss: ', result)

x1_pred = np.array([range(100, 106), range(400, 406)]).T

# 입력이 1개라 리스트 없이 그대로 넣고, 출력이 2개라 [y1 예측, y2 예측] 리스트로 나와서 각각 받는다
y1_pred, y2_pred = model.predict(x1_pred)
print(y1_pred, y2_pred)
# [[3614.191 ]
#  [3624.818 ]
#  [3635.4453]
#  [3646.0723]
#  [3656.6995]
#  [3667.3264]] [[15145.823 ]
#  [15190.235 ]
#  [15234.6455]
#  [15279.057 ]
#  [15323.466 ]
#  [15367.877 ]]

print('y1 예측값: ', y1_pred.reshape(-1))
print('y2 예측값: ', y2_pred.reshape(-1))
# y1 예측값:  [3614.191  3624.818  3635.4453 3646.0723 3656.6995 3667.3264]
# y2 예측값:  [15145.823  15190.235  15234.6455 15279.057  15323.466  15367.877 ]