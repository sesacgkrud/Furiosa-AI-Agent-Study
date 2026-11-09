# keras41_dnn3_cifar10.py
# [목적] cifar10 컬러 이미지를 (50000, 3072) 로 펴서 Dense + BatchNormalization 만으로 분류하고 CNN 결과와 비교 (목표 : CNN 을 이겨라)
# keras41_dnn1_mnist.py 베이스
#
# 아래 설명 블록은 베이스 파일에서 복사된 내용 (이 파일에서 새로 바뀐 점은 바로 위 [목적] 참고)
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
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
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

x_train = x_train.reshape(-1, 3072)
x_test = x_test.reshape(-1, 3072)

########## y 원핫 인코딩 ##########
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

# y_train = y_train.reshape(-1,1)    # cifar 의 y 는 이미 (50000, 1) 2차원이라 이 줄이 없어도 된다
# y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)

#2. 모델 구성
# Conv2D 없이 Dense 를 두 개씩 묶고 묶음마다 BatchNormalization + Dropout
# 입력은 32 x 32 x 3 = 3072 개 값을 한 줄로 편 2차원 (50000, 3072)
model = Sequential()

model.add(Dense(1024, activation='relu',input_shape=(3072,)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(512, activation='relu',))
model.add(Dense(512, activation='relu',))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(256, activation='relu',))
model.add(Dense(256, activation='relu',))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))                       # 10종 분류 -> 출력 10개 + softmax
model.summary()
# Total params: 4,204,298  (이 파일의 층 구성으로 계산한 값)
#  -> 첫 층 3072 x 1024 + 1024 = 3,146,752 가 대부분 (BatchNormalization 의 이동평균 · 이동분산 3,840 개는 Non-trainable)

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

model.fit(x_train, y_train, epochs=500, batch_size=128,
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

# ===== GPU 기록 ===== <- DNN
# Epoch 70: early stopping
# loss : 1.42743718624115
# acc : 0.5784000158309937
# accuracy_score : 0.5784
# 소요 시간 : 130.08 초

# 목표 : CNN을 이겨라