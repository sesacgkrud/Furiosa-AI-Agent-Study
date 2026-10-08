# keras69_ensemble2.py 베이스 (출력 2개)

# [목적] 입력 3개를 합친 모델(ensemble2) 끝을 두 갈래로 나눠 y 2개(화성 온도 y1, 비트코인 가격 y2)를 함께 예측한다 (다중 출력)
#   - 병합한 mg3 뒤에 분기1 (Dense 2층 + last1) / 분기2 (바로 last2) 를 붙이고 Model(outputs=[last_output1, last_output2])
#   - fit · evaluate 의 y 도 [y1, y2] 리스트로 넣는다 -> loss 는 출력마다 계산해서 더한다
#   - Total params 9,815 (ensemble2 9,634 + ld1 60 + ld2 110 + last1 11 + last2 6 - last 6)

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T
x2_datasets = np.array([range(101, 201), range(411, 511),
                        range(150, 250)]).transpose()
x3_datasets = np.array([range(100), range(301, 401),
                        range(77, 177), range(33, 133)]).T

y1 = np.array(range(3001, 3101))

y2 = np.array(range(13001, 13101)) # 비트코인 가격

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2,
    train_size=0.7,
    random_state=77,
)
print(x1_train.shape, x2_train.shape, x3_train.shape, y1_train.shape, y2_train.shape)
print(x1_test.shape, x2_test.shape, x3_test.shape, y1_test.shape, y2_test.shape)

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)

#2-2. 모델2
input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
dense22 = Dense(40, name='han22')(dense21)
dense23 = Dense(30, name='han23')(dense22)
dense24 = Dense(20, name='han24')(dense23)
output21 = Dense(3, name='han25')(dense24)

#2-3. 모델3
input100 = Input(shape=(4,))
dense101 = Dense(50, name='han101')(input100)
dense102 = Dense(40, name='han102')(dense101)
dense103 = Dense(30, name='han103')(dense102)
dense104 = Dense(20, name='han104')(dense103)
output100 = Dense(5, name='han105')(dense104)

#2-4. 모델 병합
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = Concatenate(name='mg1')([output1, output21, output100])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)

#2-5. 분기1
last_dense1 = Dense(10, name='ld1')(merge3)
last_dense2 = Dense(10, name='ld2')(last_dense1)
last_output1 = Dense(1, name='last1')(last_dense2)

#2-6. 분기2 (레이어 구성 안 하는 경우)
last_output2 = Dense(1, name='last2')(merge3)

#2-7. 모델 구성
model = Model(inputs=[input1, input21, input100], outputs=[last_output1, last_output2])
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=100, batch_size=8)

#4. 평가, 예측
result = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print('loss: ', result)                                                     # 출력이 2개라 [전체 loss, last1 loss, last2 loss] 리스트로 나온다

x1_pred = np.array([range(100, 106), range(400, 406)]).T
x2_pred = np.array([range(200, 206), range(510, 516),
                        range(249, 255)]).transpose()
x3_pred = np.array([range(100, 106), range(400, 406),
                    range(177, 183), range(133, 139)]).T

# 입력 3개를 리스트로 한 번에 넣고, 출력이 2개라 [y1 예측, y2 예측] 리스트로 나와서 각각 받는다
#  - 데이터 규칙상 정답은 y1 3101 ~ 3106, y2 13101 ~ 13106
y1_pred, y2_pred = model.predict([x1_pred, x2_pred, x3_pred])
print(y1_pred, y2_pred)
# [[3093.9756]
#  [3094.959 ]
#  [3095.939 ]
#  [3096.9202]
#  [3097.9026]
#  [3098.8843]] [[13070.581]
#  [13071.586]
#  [13072.592]
#  [13073.594]
#  [13074.6  ]
#  [13075.604]]

print('y1 예측값: ', y1_pred.reshape(-1))
print('y2 예측값: ', y2_pred.reshape(-1))

# loss:  [0.23944051563739777, 0.22564421594142914, 0.013796297833323479]
# y1 예측값:  [3093.9756 3094.959  3095.939  3096.9202 3097.9026 3098.8843]
# y2 예측값:  [13070.581 13071.586 13072.592 13073.594 13074.6   13075.604]