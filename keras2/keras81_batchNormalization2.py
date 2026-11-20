# keras81_batchNormalization1.py 베이스

# [목적]
# BatchNormalization 비교 2 / 3 : keras 방식
#   층(activation='relu') -> BatchNormalization  (활성화 함수를 거친 출력을 정규화)
#   keras 코드에서 흔히 쓰는 순서로, keras41_dnn3_cifar10.py 에서 쓴 방식과 같다
#
# [BatchNormalization]
#   배치(128 개) 단위로 층의 출력을 채널(필터)마다 평균 0 · 분산 1 로 정규화한 뒤
#   gamma(크기) · beta(이동) 를 곱하고 더해 다시 조절한다 (gamma · beta 는 훈련으로 학습)
#   - 층마다 입력 분포가 일정해져서 큰 learning rate 에서도 안정적으로, 빠르게 학습된다
#   - 출력 shape 는 그대로 (Conv2D 뒤면 4 차원 그대로, Dense 뒤면 2 차원 그대로)
#   - 파라미터 : 채널마다 4 개 (gamma, beta, moving_mean, moving_variance)
#               -> 훈련 가능 2 개 (gamma, beta) / 훈련 불가 2 개 (moving_mean, moving_variance)
#               -> Conv2D(32) 뒤 BatchNormalization 파라미터 32 * 4 = 128
#   - 훈련 때는 그 배치의 평균 · 분산, 평가 · 예측 때는 훈련 중 모아 둔 moving_mean · moving_variance 사용

import numpy as np
import pandas as pd
import time
import random
import tensorflow as tf

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
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
# 층(activation='relu') -> BatchNormalization 순서
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
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
# Total params: 3,995,434 (훈련 가능 3,994,922 / 훈련 불가 512)
#   -> BatchNormalization 4 개 (32 + 32 + 64 + 128 = 256 채널) * 4 = 1,024 개 추가, 그중 moving_mean · moving_variance 512 개는 훈련 불가
# Epoch 75: early stopping (best epoch 50 가중치 복원)
# loss : 0.02957165800035
# acc : 0.9925000071525574
# accuracy_score : 0.9925
# 소요 시간 : 462.44 초
# -> BatchNormalization 없는 1 번보다 loss 는 낮아졌지만 (0.0345 -> 0.0296) acc 는 0.9927 -> 0.9925 로 거의 같다
