# keras36_cnn5_cifar10.py
# cifar10 컬러 이미지 10종 분류 (비행기 / 자동차 / 새 / 고양이 ... ) - 컬러(3채널) 이미지를 처음 다뤘다 (목표 acc 0.67)
# keras36_cnn3_mnist2.py 베이스
#
# mnist / fashion 과 달라지는 점
#  1) 32 x 32 x 3 컬러 -> load_data() 가 처음부터 (50000, 32, 32, 3) 4차원으로 준다
#     => x 를 reshape 할 필요가 없다 (mnist 는 흑백이라 채널 1 을 직접 붙여야 했다)
#  2) y 도 처음부터 (50000, 1) 2차원이라 OneHotEncoder 에 바로 넣을 수 있다
#  3) input_shape 의 마지막이 3 (RGB) -> 첫 Conv2D 의 파라미터가 채널 수만큼 늘어난다
#     (3 x 3 x 3 + 1) x 32 = 896   <- 흑백이었으면 (3 x 3 x 1 + 1) x 32 = 320
#  4) 사진이라 손글씨보다 훨씬 어렵다. 같은 모델인데 acc 가 0.99 -> 0.72 로 떨어진다

import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 처음부터 4차원(장수, 가로, 세로, 채널) 으로 들어온다 -> x 는 reshape 없이 바로 Conv2D 에 넣는다
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# print(np.max(x_train), np.min(x_train)) # 255 0
# print(np.max(x_test), np.min(x_test))   # 255 0

########## 스케일링 : -1 ~ 1 ##########
# 컬러여도 R / G / B 모두 0 ~ 255 라 단위가 같다 -> 나눗셈 하나로 전 채널이 같이 스케일링된다
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5
# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

########## y 원핫 인코딩 ##########
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1,1)    # cifar 의 y 는 이미 (50000, 1) 2차원이라 이 줄이 없어도 된다
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)

#2. 모델 구성
# mnist2 와 층 구성은 똑같지만 입력이 28x28 이 아니라 32x32 라 출력 크기가 전부 4씩 크다
# padding 을 주지 않았으므로 (커널 크기 - 1) 만큼 줄어든다 : (3,3) -> 2씩, (5,5) -> 4씩
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))    # 출력 : (30, 30, 32)  param 896 = (3x3x3+1)x32
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))                 # 출력 : (28, 28, 32)
model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))                 # 출력 : (24, 24, 32)
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))                 # 출력 : (22, 22, 64)
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))                 # 출력 : (20, 20, 64)
model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))                 # 출력 : (16, 16, 64)
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))                # 출력 : (14, 14, 128)
model.add(Dropout(0.25))
# model.add(Flatten())                                                # (14,14,128) -> 25088. 4차원을 2차원으로 펴준다
model.add(GlobalAveragePooling2D())

model.add(Dense(units=256, activation='relu'))                      # units = Dense 의 첫 번째 인자 이름
model.add(Dropout(0.4))
model.add(Dense(units=128, activation='relu'))

model.add(Dense(10, activation='softmax'))                          # 10종 분류 -> 출력 10개 + softmax
model.summary()
# Total params: 6,724,490
#  -> 입력이 4픽셀 커진 것뿐인데 Flatten 이 12800 -> 25088 로 두 배가 되면서 파라미터가 3,578,186 -> 6,724,490 이 됐다

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

# 목표가 정확도라서 val_loss 가 아니라 val_acc 를 기준으로 잡는다
# acc 는 높을수록 좋으므로 mode='max' 를 같이 바꿔야 한다
es = EarlyStopping(
    monitor='val_acc',      # val_loss 기준이면 loss 가 가장 낮은 epoch 를 고르는데, 그 epoch 가 acc 최고점은 아니다
    mode='max',             # acc 는 높을수록 좋으므로 max
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

# 원핫으로 훈련했으므로 예측도 (10000, 10) 확률로 나온다 -> argmax 로 가장 큰 자리를 뽑아 되돌린다
y_predict = model.predict(x_test)
y_predict=  np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# ===== GPU 기록 =====
# loss : 1.0898630619049072
# acc : 0.7267000079154968
# accuracy_score : 0.7267
# 소요 시간 : 415.61 초

# ===== GPU 기록 ===== <- MaxPooling 적용
# loss : 0.9036821126937866
# acc : 0.7540000081062317
# accuracy_score : 0.754
# 소요 시간 : 202.33 초

# ===== GPU 기록 ===== <- GlobalAveragePooling2D 적용
# loss : 0.8530533313751221
# acc : 0.7727000117301941
# accuracy_score : 0.7727
# 소요 시간 : 199.62 초

# 목표 : acc 0.67  -> 0.7267 로 달성
# -> 같은 모델로 mnist 는 0.9938 이 나왔는데 cifar10 은 0.7267 이다.
#    28x28 흑백 손글씨보다 32x32 컬러 사진이 훨씬 어려운 문제라는 뜻