# keras43_hamsu04_cifar100.py
# keras41_dnn4_cifar100.py (Sequential DNN) 을 함수형(Model) 으로 바꾼 파일
# 층 구성을 그대로 옮겼으므로 keras41_dnn4 와 Total params 가 같아야 제대로 변환한 것
#
# keras41_dnn4 의 내용 (그대로 유지)
#  1) 정답이 0 ~ 99 -> 원핫이 (50000, 100), 마지막 Dense 가 100
#  2) 클래스는 10배인데 장수는 그대로라 한 종류당 훈련 이미지가 5000장 -> 500장으로 줄어 훨씬 어렵다
#  3) 출력이 100칸이라 마지막 은닉층을 128 로 끝낸다
#     dnn1 / dnn2 처럼 16 까지 좁히면 100종을 16개 숫자로 구분해야 해서 acc 가 크게 떨어진다
#
# 이미지를 DNN 에 넣으려면 2차원으로 편다 : (50000, 32, 32, 3) -> (50000, 3072)
#  Conv2D 는 가로 세로 위치 관계를 보지만, Dense 는 3072개 컬럼을 그냥 나열된 값으로 본다

import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Model                   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization   # Input -> 함수형 모델의 입력층
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# 컬러라 처음부터 4차원 (장수, 가로, 세로, 채널=3)

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

########## 2차원으로 reshape ##########
# Dense 는 (샘플, 컬럼) 2차원 입력 -> 32 x 32 x 3 = 3072 개 컬럼으로 편다 (채널 3 을 빠뜨리면 shape 에러)
x_train = x_train.reshape(-1, 32 * 32 * 3)
x_test = x_test.reshape(-1, 32 * 32 * 3)
print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)

########## y 원핫 인코딩 ##########
# 클래스가 0 ~ 99 라 원핫 결과가 100 컬럼이 된다
ohe = OneHotEncoder(sparse_output=False)   # 희소행렬이 아니라 numpy 배열로 받는다

# cifar 의 y 는 이미 (50000, 1) 2차원이라 reshape 없이 바로 넣을 수 있다
y_train = ohe.fit_transform(y_train)   # train 으로 기준(클래스 목록)을 만든다
y_test = ohe.transform(y_test)         # test 는 그 기준으로 변환만

print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)

#2. 모델 구성 (함수형) - keras41_dnn4 와 층 / BatchNormalization / Dropout 위치가 똑같다
# Dense 를 두 개씩 묶고 묶음마다 BatchNormalization + Dropout
# BatchNormalization : 층을 지날 때마다 흩어지는 출력 분포를 배치 단위로 다시 맞춰준다 -> 깊은 Dense 모델이 잘 학습된다
#
# 변수 이름은 층마다 새로 붙인다 (dense1 / bn1 / drop1 ...)
# 앞 층과 같은 이름을 다시 쓰면 그 변수가 덮어써져서 해당 층이 모델 연결에서 빠진다
input1 = Input(shape=(3072,))                     # 입력층 : 컬럼 3072개 (Sequential 의 input_shape=(3072,))
dense1 = Dense(1024, activation='relu')(input1)   # (input1) -> input1 뒤에 연결
bn1 = BatchNormalization()(dense1)
drop1 = Dropout(0.4)(bn1)                         # BatchNormalization 다음이므로 bn1 에 연결

dense2 = Dense(512, activation='relu')(drop1)     # Dropout 다음 층은 drop1 에 연결
dense3 = Dense(512, activation='relu')(dense2)
bn2 = BatchNormalization()(dense3)
drop2 = Dropout(0.4)(bn2)

dense4 = Dense(256, activation='relu')(drop2)
dense5 = Dense(256, activation='relu')(dense4)
bn3 = BatchNormalization()(dense5)
drop3 = Dropout(0.3)(bn3)

dense6 = Dense(128, activation='relu')(drop3)
dense7 = Dense(128, activation='relu')(dense6)
bn4 = BatchNormalization()(dense7)
drop4 = Dropout(0.3)(bn4)

# 다중 분류(원핫) 출력층은 softmax 가 반드시 있어야 칸별 확률(합 = 1)이 나온다
output1 = Dense(100, activation='softmax')(drop4)   # 100종 분류 -> 출력 100개 + softmax

model = Model(inputs=input1, outputs=output1)   # 시작(input1) ~ 끝(output1) 범위를 정해서 모델 완성

model.summary()
# Total params: 4,201,316  <- keras41_dnn4(Sequential) 과 같아야 제대로 변환한 것
#  -> 첫 층 3072 x 1024 + 1024 = 3,146,752 로 전체의 75% 를 차지한다
#     BatchNormalization 파라미터는 채널마다 4개(감마 / 베타 / 이동평균 / 이동분산)이고,
#     이동평균 / 이동분산은 훈련으로 배우지 않아 Non-trainable params 3,840 으로 따로 표시된다

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

# 목표가 정확도라서 val_loss 가 아니라 val_acc 를 기준으로 잡는다
#  - loss 는 낮을수록 좋고 acc 는 높을수록 좋으므로 mode 를 'max' 로 같이 바꿔야 한다
es = EarlyStopping(
    monitor='val_acc',      # val_loss 기준이면 loss 가 가장 낮은 epoch 를 고르는데, 그 epoch 가 acc 최고점은 아니다
    mode='max',             # acc 는 높을수록 좋으므로 max
    patience=25,            # acc 는 들쭉날쭉해서 loss 기준보다 patience 를 넉넉히 준다
    restore_best_weights=True,  # 멈춘 뒤 val_acc 가 가장 높았던 가중치로 되돌린다
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

# ===== GPU 기록 ===== <- DNN
# loss : 3.4793953895568848
# acc : 0.29899999499320984
# accuracy_score : 0.299
# 소요 시간 : 201.95 초

# ===== GPU 기록 ===== <- 함수형 적용
# Epoch 107: early stopping
# loss : 3.402829647064209
# acc : 0.2971999943256378
# accuracy_score : 0.2972
# 소요 시간 : 185.42 초
