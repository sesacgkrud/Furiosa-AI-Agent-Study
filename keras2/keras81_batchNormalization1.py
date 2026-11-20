# keras36_cnn3_mnist2.py 베이스

# [목적]
# BatchNormalization 비교 1 / 3 : BatchNormalization 을 쓰지 않은 기준 모델
#   1. keras81_batchNormalization1.py : BatchNormalization 없음                      <- 이 파일
#   2. keras81_batchNormalization2.py : keras 방식  Conv2D(activation='relu') -> BatchNormalization
#   3. keras81_batchNormalization3.py : 2015 논문 방식  Conv2D(bias 없음) -> BatchNormalization -> relu
#
# [공정한 비교를 위해 세 파일에서 똑같이 맞춘 것]
#   - 데이터 / 스케일링 / Conv2D · Dense 개수와 크기 / Dropout 위치 / compile / EarlyStopping / batch_size
#   - 랜덤 시드 333 고정 (random · numpy · tensorflow)
#   -> 다른 것은 BatchNormalization 유무와 위치뿐

import numpy as np
import pandas as pd
import time
import random
import tensorflow as tf

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

random.seed(333)        # Python 초기값(랜덤값) 고정
np.random.seed(333)     # Numpy 초기값(랜덤값) 고정
tf.random.set_seed(333) # Tensorflow 초기값(랜덤값) 고정

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))

########## 스케일링 2 ##########
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5
print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)

########## y 원핫 인코딩 ##########
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))                 # 출력 : (22, 22, 64)
model.add(Dropout(0.25))
model.add(Flatten())                                                        # (22, 22, 64) -> 30976. Dense(10) 앞에 꼭 펴줘야 출력이 (None, 10)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
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
# Total params: 3,994,410 (훈련 가능 3,994,410 / 훈련 불가 0)
# Epoch 54: early stopping (best epoch 29 가중치 복원)
# loss : 0.03446640819311142
# acc : 0.9926999807357788
# accuracy_score : 0.9927
# 소요 시간 : 178.45 초
