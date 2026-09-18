# keras43_hamsu03_cifar10.py
# keras41_dnn3_cifar10.py (Sequential DNN) 를 함수형(Model) 으로 바꾼 파일
#
# 함수형은 모델 종류(DNN / CNN)와 상관없이 '#2. 모델 구성' 을 쓰는 문법만 다르다
#  Sequential : model.add() 로 위에서 아래로 자동 연결
#  함수형     : Input 층을 따로 만들고, 층마다 뒤에 (앞층) 을 붙여 직접 연결한 뒤
#               Model(inputs=시작, outputs=끝) 으로 범위를 정해 완성
# -> 층 구성을 그대로 옮겼으므로 keras41_dnn3 과 Total params 가 같아야 제대로 변환한 것
#
# mnist / fashion 과 달라지는 점
#  1) 32 x 32 x 3 컬러 -> 편면 컬럼이 784 가 아니라 3072 개
#  2) y 도 처음부터 (50000, 1) 2차원이라 reshape 없이 OneHotEncoder 에 바로 넣을 수 있다

import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model                   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input   # Input -> 함수형 모델의 입력층
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 컬러라 처음부터 4차원 (장수, 가로, 세로, 채널=3)
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

########## 2차원으로 reshape ##########
# Dense 는 (샘플, 컬럼) 2차원 입력 -> 32 x 32 x 3 = 3072 개 컬럼으로 편다 (채널 3 을 빠뜨리면 shape 에러)
x_train = x_train.reshape(-1, 32*32*3)
x_test = x_test.reshape(-1, 32*32*3)
print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)

########## y 원핫 인코딩 ##########
ohe = OneHotEncoder(sparse_output=False)   # 희소행렬이 아니라 numpy 배열로 받는다

# y_train = y_train.reshape(-1,1)    # cifar 의 y 는 이미 (50000, 1) 2차원이라 이 줄이 없어도 된다
# y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)   # train 으로 기준(클래스 목록)을 만든다
y_test = ohe.transform(y_test)         # test 는 그 기준으로 변환만

print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)

#2. 모델 구성 (함수형) - keras41_dnn3 과 층 / Dropout 위치가 똑같다
# 변수 이름은 층마다 새로 붙인다 (dense1 / dense2 ...)
# 앞 층과 같은 이름을 다시 쓰면 그 변수가 덮어써져서 해당 층이 모델 연결에서 빠진다
input1 = Input(shape=(32*32*3,))                  # 입력층 : 컬럼 3072개 (Sequential 의 input_shape=(3072,))
dense1 = Dense(512, activation='relu')(input1)    # (input1) -> input1 뒤에 연결
dense2 = Dense(256, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
drop1 = Dropout(0.4)(dense4)                      # dense4 출력의 40% 를 훈련 때마다 끈다

dense5 = Dense(64, activation='relu')(drop1)      # Dropout 다음 층은 drop1 에 연결
dense6 = Dense(64, activation='relu')(dense5)
dense7 = Dense(32, activation='relu')(dense6)
dense8 = Dense(32, activation='relu')(dense7)
drop2 = Dropout(0.3)(dense8)

dense9 = Dense(16, activation='relu')(drop2)
dense10 = Dense(16, activation='relu')(dense9)
drop3 = Dropout(0.2)(dense10)

# 다중 분류(원핫) 출력층은 softmax 가 반드시 있어야 칸별 확률(합 = 1)이 나온다
output1 = Dense(10, activation='softmax')(drop3)

model = Model(inputs=input1, outputs=output1)   # 시작(input1) ~ 끝(output1) 범위를 정해서 모델 완성

model.summary()
# Total params: 1,770,634  <- keras41_dnn3(Sequential) 과 같아야 제대로 변환한 것
#  -> 첫 층 3072 x 512 + 512 = 1,573,376 으로 전체의 89% 를 차지한다

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

# ===== GPU 기록 ===== <- 함수형 적용
# Epoch 83: early stopping
# loss : 2.6085219383239746
# acc : 0.5189999938011169
# accuracy_score : 0.519
# 소요 시간 : 99.77 초