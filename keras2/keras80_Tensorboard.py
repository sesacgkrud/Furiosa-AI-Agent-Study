# keras36_cnn3_mnist1.py 베이스

# [TensorBoard]
#   훈련 중 loss · acc 같은 기록과 모델 구조를 log_dir 폴더에 로그 파일로 저장하는 콜백
#   - EarlyStopping 처럼 callbacks=[...] 에 넣으면 epoch 마다 로그가 쌓인다
#   - log_dir 아래에 train / validation 폴더가 따로 생긴다 (validation_split 을 줬기 때문)
#   - 저장된 로그는 터미널에서 tensorboard --logdir=로그폴더 로 실행해 웹 브라우저(기본 http://localhost:6006) 에서 그래프로 본다
#   [인자]
#   - log_dir         : 로그를 저장할 폴더
#   - histogram_freq  : 몇 epoch 마다 가중치 히스토그램을 기록할지 (0 = 기록 안 함)
#   - write_graph     : 모델 구조(그래프) 를 기록할지
#   - write_images    : 가중치를 이미지로 기록할지
#
# EarlyStopping patience=50 이 epochs=50 과 같아서 이 설정에서는 중간에 멈추지 않고 50 epoch 를 모두 돈다
# (restore_best_weights=True 라 끝난 뒤 val_loss 가 가장 낮았던 가중치로 되돌린다)

import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten

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

########## 4차원으로 reshape ##########
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
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=(2,2), activation='relu'))
model.add(Conv2D(16, kernel_size=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=(2,2), activation='relu'))

model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

###################### 추가 ######################
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 50,
    verbose = 1,
    restore_best_weights = True,
)

tb = TensorBoard(
    log_dir = "C:/furiosa_study/_save/tensorboard/_keras80/",   # 이 폴더 아래 train / validation 로그가 생긴다
    histogram_freq=0,                                           # 가중치 히스토그램 기록 안 함
    write_graph=True,                                           # 모델 구조 기록
    write_images=True,                                          # 가중치를 이미지로 기록
)
##################################################

start_time = time.time()

model.fit(x_train, y_train, epochs=50, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, tb],                                   # EarlyStopping 과 TensorBoard 를 함께 적용
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

# ===== GPU 기록 =====   (TensorBoard 적용)
# loss : 0.039372559636831284
# acc : 0.9898999929428101
# accuracy_score : 0.9899
# 소요 시간 : 137.94 초
# -> keras36_cnn3_mnist1 GPU 기록 (acc 0.9897 / 141.31 초) 과 비슷하다 → TensorBoard 로그를 남겨도 성능 · 속도에 큰 차이가 없다
