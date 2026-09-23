# keras52_ReduceLR13_cifar10.py
# cifar10 (다중 분류, 함수형 DNN) - ReduceLROnPlateau 로 훈련 도중 learning_rate 를 줄인다
# 증폭(keras51_augment3) 까지 끝낸 함수형 DNN 에 learning_rate + ReduceLROnPlateau 를 붙인다
#
# [ 오늘 추가되는 곳은 콜백 하나다 ]
#  keras52_optimizer 는 learning_rate 를 처음부터 끝까지 고정했다
#  하지만 한 값으로 두 가지를 다 할 수는 없다
#   초반에는 크게 움직여 빨리 내려가야 하고, 최저점 근처에서는 작게 움직여야 지나치지 않는다
#
# [ ReduceLROnPlateau : 훈련 도중 learning_rate 를 줄여 준다 ]
#  plateau = 고원. val_loss 가 더 이상 줄지 않고 평평해지는 구간을 말한다
#  monitor 가 patience 동안 좋아지지 않으면 learning_rate 에 factor 를 곱한다
#   factor=0.5 -> 0.0005 -> 절반 -> 또 절반 ... 으로 떨어진다
#  EarlyStopping 과 역할이 다르다 : es 는 '멈춘다', rlr 은 '보폭을 줄여 더 해 본다'
#   그래서 rlr 의 patience 를 es 보다 짧게 줘야 "줄여 보고 -> 그래도 안 되면 멈춘다" 순서가 된다
#  이 파일 : 시작 learning_rate = 0.0005 / es patience = 25 / rlr patience = 20
#
# keras51_augment3_cifar10.py 베이스

import numpy as np
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model                   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input   # Input -> 함수형 모델의 입력층
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)   # (50000, 32, 32, 3)
print(y_train.shape)   # (50000, 1)
print(x_test.shape)    # (10000, 32, 32, 3)
print(y_test.shape)    # (10000, 1)

datagen = ImageDataGenerator(
    horizontal_flip=True,       # 좌우반전 - 뒤집어도 같은 대상이다
    width_shift_range=0.1,      # 좌우로 최대 10% 이동
    rotation_range=15,          # -15 ~ +15 도 범위에서 회전
    fill_mode='nearest',        # 이동 / 회전으로 생긴 빈 자리를 가장 가까운 픽셀값으로 채운다
)

augment_size = 40000

randidx = np.random.randint(
    x_train.shape[0],
    size=augment_size
)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)   # (40000, 32, 32, 3)
print(y_augmented.shape)   # (40000, 1)

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,              # 원본 순서를 유지해야 y_augmented 와 짝이 맞는다
).next()[0]                     # [0] = x, [1] = y

print(x_augmented.shape)   # (40000, 32, 32, 3)

x_train = np.concatenate(
    [x_train, x_augmented],
    axis=0
)

y_train = np.concatenate(
    [y_train, y_augmented],
    axis=0
)

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

x_train = x_train.reshape(-1, 32 * 32 * 3)
x_test = x_test.reshape(-1, 32 * 32 * 3)

print(x_train.shape)   # (90000, 3072)
print(x_test.shape)    # (10000, 3072)

ohe = OneHotEncoder(sparse_output=False)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(y_train.shape)   # (90000, 10)
print(y_test.shape)    # (10000, 10)

#2. 모델 구성 (함수형) - keras41_dnn3 과 층 / Dropout 위치가 똑같다
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

output1 = Dense(10, activation='softmax')(drop3)

model = Model(inputs=input1, outputs=output1)   # 시작(input1) ~ 끝(output1) 범위를 정해서 모델 완성

model.summary()

#3. 컴파일, 훈련
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009
learning_rate = 0.0005

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',      # val_loss 기준이면 loss 가 가장 낮은 epoch 를 고르는데, 그 epoch 가 acc 최고점은 아니다
    mode='max',             # acc 는 높을수록 좋으므로 max
    patience=25,            # acc 는 들쭉날쭉해서 loss 기준보다 patience 를 넉넉히 준다
    restore_best_weights=True,  # 멈춘 뒤 val_acc 가 가장 높았던 가중치로 되돌린다
    verbose=1,
)

# val_loss 가 patience 동안 좋아지지 않으면 learning_rate 에 factor 를 곱해 줄인다
# verbose=1 -> 줄어드는 순간 'ReduceLROnPlateau reducing learning rate to ...' 가 찍힌다
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=500, batch_size=128,
          verbose=1,
          validation_split=0.1,
          callbacks=[es, rlr]
          )

end_time = time.time()

#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

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

# ===== GPU 기록 ===== <- 증폭 적용
# Epoch 143: early stopping
# loss : 2.7626540660858154
# acc : 0.5432999730110168
# accuracy_score : 0.5433
# 소요 시간 : 306.04 초

# ===== GPU 기록 ===== <- learning rate 적용
# loss : 2.9907498359680176
# acc : 0.5307999849319458
# accuracy_score : 0.5308
# 소요 시간 : 273.07 초

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후

# Epoch 100: early stopping
# loss : 7.196678161621094
# acc : 0.5346999764442444
# accuracy_score : 0.5347
# 소요 시간 : 187.66 초

# [결론] 0.005 고정 0.5308 -> 시작 0.0005 + ReduceLROnPlateau 0.5347 (Epoch 100).
#        증폭 기록 0.5433 과 거의 같다. cifar10 은 DNN 구조 자체의 한계에 걸려 있어
#        learning_rate 를 어떻게 만져도 0.53 근처에서 멈춘다 (CNN 기록은 0.7727).
