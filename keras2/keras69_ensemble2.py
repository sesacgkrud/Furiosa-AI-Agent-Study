# keras69_ensemble1.py 베이스

# [목적] ensemble1 의 입력 2개에 x3 (100, 4) 를 하나 더 넣어 입력 3개 모델을 만든다
#   - 모델3 : Input(shape=(4,)) -> Dense 5층 -> output100 (None, 5)
#   - Concatenate([output1, output21, output100]) 로 세 모델 출력을 합친다
#   - train_test_split · fit · evaluate · predict 에 x3 를 함께 넣는다

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
                        range(77, 177), range(33, 133)]).T # (100, 4)

y = np.array(range(3001, 3101))

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y,
    train_size=0.7,
    random_state=77,
)
print(x1_train.shape, x2_train.shape, x3_train.shape, y_train.shape)
print(x1_test.shape, x2_test.shape, x3_test.shape, y_test.shape)

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
last_output = Dense(1, name='last')(merge3)

#2-5. 모델 구성
model = Model(inputs=[input1, input21, input100], outputs=last_output)
model.summary()

# 모델3 이 추가되어 병합층 mg1 이 (None, 5 + 3 + 5) = (None, 13), mg2 param 13 * 10 + 10 = 140
#  han101 (None, 50) 250 · han102 2,040 · han103 1,230 · han104 620 · han105 (None, 5) 105
#  Total params: 9,634 (37.63 KB)   (ensemble1 5,339)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], y_train, epochs=100, batch_size=8)

#4. 평가, 예측
result = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('loss: ', result)

x1_pred = np.array([range(100, 106), range(400, 406)]).T
x2_pred = np.array([range(200, 206), range(510, 516),
                        range(249, 255)]).transpose()
x3_pred = np.array([range(100, 106), range(400, 406),
                    range(177, 183), range(133, 139)]).T   # (6, 4)

y_pred = model.predict([x1_pred, x2_pred, x3_pred])
print(y_pred)
# [[3092.5786]
#  [3094.1484]
#  [3095.7173]
#  [3097.2864]
#  [3098.8555]
#  [3100.4246]]

print('예측값: ', y_pred.reshape(-1))
# loss:  0.009273995645344257
# 예측값:  [3092.5786 3094.1484 3095.7173 3097.2864 3098.8555 3100.4246]