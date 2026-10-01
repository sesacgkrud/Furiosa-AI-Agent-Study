# keras36_cnn6_cifar100.py
# cifar100 컬러 이미지 100종 분류 (목표 acc 0.4)
# keras36_cnn5_cifar10.py 베이스 - 이미지는 cifar10 과 똑같이 32x32 컬러 5만 장이고 정답 종류만 10개 -> 100개
#  - 클래스는 10배인데 장수는 그대로라 한 종류당 훈련 이미지가 5000장 -> 500장으로 줄어 훨씬 어렵다
#  - 마지막 Dense 가 100 이 되고 원핫도 (50000, 100) 이 된다
#
# 1차 실행 후 keras39 에서 배운 MaxPooling2D 를 모델에 넣었다
#  MaxPooling2D() : 2x2 구역마다 가장 큰 값 하나만 남긴다 -> 가로 세로가 절반, 파라미터 0
#  padding 없는 Conv2D 는 (3,3) 마다 2씩만 줄어서 Flatten 이 커지는데,
#  MaxPooling 으로 크기를 줄이면 Flatten 이 작아져 파라미터가 크게 줄어든다
#   MaxPooling 없던 1차 모델 : Flatten 25088 -> Total params 6,736,100
#   현재 모델 (MaxPooling 2개) : Flatten 1152  -> Total params 627,972

import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# 컬러라 처음부터 4차원 (장수, 가로, 세로, 채널=3) -> x 는 reshape 없이 바로 Conv2D 에 넣는다

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

########## 스케일링 : -1 ~ 1 ##########
# R / G / B 모두 0 ~ 255 로 단위가 같아서 나눗셈 하나로 전 채널이 같이 스케일링된다
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5
print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

########## y 원핫 인코딩 ##########
# 클래스가 0 ~ 99 라 원핫 결과가 100 컬럼이 된다
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)   # 희소행렬이 아니라 numpy 배열로 받는다

y_train = y_train.reshape(-1,1)    # cifar 의 y 는 이미 (50000, 1) 2차원이라 이 줄이 없어도 된다
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)

#2. 모델 구성
# padding 없는 (3,3) Conv2D -> 가로 세로 2씩 감소 / MaxPooling2D() -> 절반
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32 * 32 * 3)))    # 출력 : (30, 30, 32)  param 896 = (3x3x3+1)x32
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))                 # 출력 : (28, 28, 32)
model.add(MaxPooling2D())  
                                                                            # 출력 : (14, 14, 32)  2x2 중 최대값만 -> 절반, param 0
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))                 # 출력 : (12, 12, 64)
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))                 # 출력 : (10, 10, 64)
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))                # 출력 : (8, 8, 128)
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))                # 출력 : (6, 6, 128)
model.add(MaxPooling2D())
                                                                            # 출력 : (3, 3, 128)
model.add(Dropout(0.25))
# model.add(Flatten())                                                        # (3,3,128) -> 1152. 4차원을 2차원으로 펴준다 (값과 순서가 그대로라 reshape 와 같다)
model.add(GlobalAveragePooling2D())

model.add(Dense(units=256, activation='relu'))                              # units = Dense 의 첫 번째 인자 이름 (Dense(256) 과 같다)
model.add(Dropout(0.4))
model.add(Dense(units=128, activation='relu'))

model.add(Dense(100, activation='softmax'))                                 # 100종 분류 -> 출력 100개 + softmax
model.summary()
# Total params: 627,972
#  -> flatten(1152) -> dense(256) = 1152 x 256 + 256 = 295,168
#     MaxPooling 없던 1차 모델은 이 자리가 25088 x 256 + 256 = 6,422,784 였다

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

# 목표가 정확도라서 val_loss 가 아니라 val_acc 를 기준으로 잡고, acc 는 높을수록 좋으므로 mode='max'
es = EarlyStopping(
    monitor='val_acc',      # val_loss 기준이면 loss 가 가장 낮은 epoch 를 고르는데, 그 epoch 가 acc 최고점은 아니다
    mode='max',             # acc 는 높을수록 좋으므로 max
    patience=25,
    restore_best_weights=True,
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=128,
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

# 예측 결과는 (10000, 100) 확률 -> argmax 로 가장 큰 자리를 뽑아 0 ~ 99 로 되돌린다
y_predict = model.predict(x_test)
y_predict=  np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# ===== GPU 기록 (1차) ===== <- MaxPooling 없음 (Conv2D 7층, Total params 6,736,100)
# loss : 3.258212089538574
# acc : 0.26930001378059387
# accuracy_score : 0.2693
# 소요 시간 : 394.21 초

# ===== GPU 기록 (2차) ===== <- MaxPooling 적용
# loss : 2.2390894889831543
# acc : 0.43709999322891235
# accuracy_score : 0.4371
# 소요 시간 : 227.8 초

# ===== GPU 기록 ===== <- GlobalAveragePooling2D 적용
# loss : 2.1279296875
# acc : 0.46209999918937683
# accuracy_score : 0.4621
# 소요 시간 : 227.22 초

# 목표 : acc 0.4