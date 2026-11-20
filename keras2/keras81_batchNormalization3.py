# keras81_batchNormalization2.py 베이스

# [목적]
# BatchNormalization 비교 3 / 3 : 2015 BatchNormalization 논문 방식
#   (Ioffe & Szegedy, 2015, "Batch Normalization: Accelerating Deep Network Training
#    by Reducing Internal Covariate Shift")
#   층(bias 없음, 활성화 없음) -> BatchNormalization -> Activation('relu')
#
# [keras 방식(keras81_batchNormalization2) 과 다른 점]
#   1. 순서 : 논문은 활성화 함수 "전" 의 값 (x·W + b) 을 정규화한 뒤 relu 를 통과시킨다
#            -> 층에서 activation 을 빼고, BatchNormalization 다음에 Activation('relu') 층을 따로 둔다
#            -> keras 방식은 relu 를 먼저 거쳐 음수가 0 으로 잘린 값을 정규화한다
#   2. bias : BatchNormalization 이 평균을 빼면 bias 를 더한 효과가 사라지고, beta 가 그 역할을 대신한다
#            -> 논문대로 use_bias=False 로 bias 를 뺀다
#            -> 첫 번째 Conv2D(32) 파라미터 3*3*1*32 + 32 = 320 -> 288 (bias 32 개만큼 줄어든다)
#
# [Activation]
#   층 안의 activation='relu' 와 똑같은 계산을 하는 별도의 층 (파라미터 0 개)
#   -> BatchNormalization 과 relu 사이에 끼워 넣으려고 층으로 분리했다

import numpy as np
import pandas as pd
import time
import random
import tensorflow as tf

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

########## 스케일링 2 ##########
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

########## y 원핫 인코딩 ##########
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

#2. 모델 구성
# 층(use_bias=False, activation=None) -> BatchNormalization -> Activation('relu') 순서
model = Sequential()
model.add(Conv2D(32, (3,3), activation=None, use_bias=False, input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(3,3), activation=None, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation=None, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation=None, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))      # 출력층은 BatchNormalization 없이 softmax 그대로
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=25,
    restore_best_weights=True,
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=80, batch_size=128,
          verbose=1,
          validation_split=0.1,
          callbacks=[es]
          )

end_time = time.time()

#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)
y_predict=  np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# ===== GPU 기록 (tf29xgpu, RTX 4060) =====
# Total params: 3,995,178 (훈련 가능 3,994,666 / 훈련 불가 512)
#   -> 2 번보다 256 개 적다 : use_bias=False 로 Conv2D · Dense(128) 의 bias 32 + 32 + 64 + 128 = 256 개가 빠졌다
# Epoch 80/80 까지 모두 훈련 (val_acc 최고 0.9950 이 62 epoch → 그 뒤 18 epoch 라 patience 25 에 못 미쳐 early stopping 이 걸리지 않음)
#   -> 끝까지 가면 restore_best_weights 가 적용되지 않아 마지막 epoch 가중치로 평가
# loss : 0.02419729344546795
# acc : 0.992900013923645
# accuracy_score : 0.9929
# 소요 시간 : 238.62 초

# ===== 3 개 비교 (같은 시드 333, GPU) =====
# | 파일                       | 방식                          | 멈춘 epoch (best) | loss   | acc    | 시간     |
# | 1. BatchNormalization 없음 | Conv2D(relu)                  | 54 (29)           | 0.0345 | 0.9927 | 178.45초 |
# | 2. keras 방식              | Conv2D(relu) -> BN            | 75 (50)           | 0.0296 | 0.9925 | 462.44초 |
# | 3. 2015 논문 방식          | Conv2D(bias X) -> BN -> relu  | 80 (끝까지)       | 0.0242 | 0.9929 | 238.62초 |
#
# -> 가장 좋은 것 : 3 번 논문 방식 (loss 가장 낮고 acc 가장 높다)
# -> 다만 acc 차이는 0.9925 ~ 0.9929, 10000 장 중 2 ~ 4 장 차이라 mnist 에서는 거의 같다
#    차이가 뚜렷한 것은 loss : 0.0345 -> 0.0296 -> 0.0242 로 BN 을 쓰면 낮아지고, relu 앞에 둘 때 더 낮다
# -> BN 을 쓴 2 · 3 번은 늦게까지 val_acc 가 올라 더 오래 훈련됐다 (최고 val_acc epoch : 1 번 29 / 2 번 50 / 3 번 62)
# -> GPU 계산은 시드를 고정해도 완전히 같게 나오지 않아 다시 돌리면 순위가 바뀔 수 있다
