# [목적] 서로 다른 데이터 2개(x1, x2)를 각자의 모델에 넣고, 두 모델의 출력을 합쳐(Concatenate) y 하나를 예측하는 앙상블 모델
#   - Sequential 은 입력이 하나뿐이라 함수형(Input / Model) 으로 만든다
#   - 모델1 : x1 (100, 2) -> Dense 4층 -> output1 (None, 5)
#   - 모델2 : x2 (100, 3) -> Dense 5층 -> output21 (None, 3)
#   - 병합  : Concatenate([output1, output21]) -> (None, 8) -> Dense -> last (None, 1)
#   - Model(inputs=[input1, input21], outputs=last_output) -> fit · evaluate · predict 에 x 를 [x1, x2] 리스트로 넣는다

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T   # (100, 2)
                        # 삼성 종가       하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), # (100, 3)
                        # 원유가          환율
                        range(150, 250)]).transpose()
                        # 금시세

y = np.array(range(3001, 3101))
            # 화성 화씨 온도

# train / test 분리
#  x1, x2, y 를 한 번에 넣어야 같은 행끼리 같은 순서로 섞여서 나뉜다 (따로 나누면 x 와 y 의 짝이 어긋난다)
#  반환 순서 : 넣은 배열마다 (train, test) 한 쌍씩 -> x1_train, x1_test, x2_train, x2_test, y_train, y_test
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y,
    train_size=0.7,
    random_state=77,
)
print(x1_train.shape, x2_train.shape, y_train.shape)   # (70, 2) (70, 3) (70,)
print(x1_test.shape, x2_test.shape, y_test.shape)      # (30, 2) (30, 3) (30,)

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)   # 모델1 · 2 를 따로 Model 로 만들 필요 없음 -> 출력 텐서(output1, output21)를 바로 합친다

#2-2. 모델2
input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
dense22 = Dense(40, name='han22')(dense21)
dense23 = Dense(30, name='han23')(dense22)
dense24 = Dense(20, name='han24')(dense23)
output21 = Dense(3, name='han25')(dense24)
# model2 = Model(inputs=input21, outputs=output21)

#2-3. 모델 병합
from tensorflow.keras.layers import concatenate, Concatenate

# 둘 중 아무거나 사용 : concatenate 는 함수, Concatenate 는 층(클래스) -> 마지막 축(feature)으로 이어 붙인다 (5 + 3 = 8)
# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output21])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

#2-4. 모델 구성
model = Model(inputs=[input1, input21], outputs=last_output)
model.summary()

# from tensorflow.keras.layers.merge import concatenate -> ModuleNotFoundError (지금 버전은 tensorflow.keras.layers 에서 import)
# Model: "functional"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ input_layer_1 (InputLayer)    │ (None, 3)                 │               0 │ -                          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ input_layer (InputLayer)      │ (None, 2)                 │               0 │ -                          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han21 (Dense)                 │ (None, 50)                │             200 │ input_layer_1[0][0]        │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han1 (Dense)                  │ (None, 10)                │              30 │ input_layer[0][0]          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han22 (Dense)                 │ (None, 40)                │           2,040 │ han21[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han2 (Dense)                  │ (None, 20)                │             220 │ han1[0][0]                 │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han23 (Dense)                 │ (None, 30)                │           1,230 │ han22[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han3 (Dense)                  │ (None, 30)                │             630 │ han2[0][0]                 │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han24 (Dense)                 │ (None, 20)                │             620 │ han23[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han4 (Dense)                  │ (None, 5)                 │             155 │ han3[0][0]                 │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ han25 (Dense)                 │ (None, 3)                 │              63 │ han24[0][0]                │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ mg1 (Concatenate)             │ (None, 8)                 │               0 │ han4[0][0], han25[0][0]    │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ mg2 (Dense)                   │ (None, 10)                │              90 │ mg1[0][0]                  │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ mg3 (Dense)                   │ (None, 5)                 │              55 │ mg2[0][0]                  │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ last (Dense)                  │ (None, 1)                 │               6 │ mg3[0][0]                  │
# └───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
#  Total params: 5,339 (20.86 KB)
#  Trainable params: 5,339 (20.86 KB)
#  Non-trainable params: 0 (0.00 B)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8)  # 입력이 2개라 x 도 [x1, x2] 리스트로

#4. 평가, 예측
result = model.evaluate([x1_test, x2_test], y_test)
print('loss: ', result)

x1_pred = np.array([range(100, 106), range(400, 406)]).T   # (6, 2)
                        # 삼성 종가       하이닉스 종가
x2_pred = np.array([range(200, 206), range(510, 516), # (6, 3)
                        # 원유가          환율
                        range(249, 255)]).transpose()
                        # 금시세

#  예측값 구하기
#  - 입력이 2개인 모델이라 predict 에도 [x1, x2] 리스트로 넣는다 (fit · evaluate 와 같은 형태)
#  - 데이터 규칙상 x1 첫 열이 100 ~ 105 이면 정답 y 는 3101 ~ 3106
y_pred = model.predict([x1_pred, x2_pred])
print(y_pred)
# [[3100.5999]
#  [3101.8586]
#  [3103.1177]
#  [3104.3767]
#  [3105.6355]
#  [3106.8943]]

print('예측값: ', y_pred.reshape(-1))                               # (6, 1) -> (6,) 로 펴서 한 줄로 출력

# loss:  0.05035315081477165
# 예측값:  [3100.5999 3101.8586 3103.1177 3104.3767 3105.6355 3106.8943]