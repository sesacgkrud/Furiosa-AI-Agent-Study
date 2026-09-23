# keras52_ReduceLR14_cifar100.py
# cifar100 (다중 분류, 함수형 DNN + BatchNormalization) - ReduceLROnPlateau 적용
# 증폭(keras51_augment4) 까지 끝낸 함수형 DNN 에 learning_rate + ReduceLROnPlateau 를 붙인다
#
# [ 오늘 추가되는 곳은 콜백 하나다 ]
#  keras52_optimizer 는 learning_rate 를 처음부터 끝까지 고정했다
#  하지만 한 값으로 두 가지를 다 할 수는 없다
#   초반에는 크게 움직여 빨리 내려가야 하고, 최저점 근처에서는 작게 움직여야 지나치지 않는다
#
# [ ReduceLROnPlateau : 훈련 도중 learning_rate 를 줄여 준다 ]
#  plateau = 고원. val_loss 가 더 이상 줄지 않고 평평해지는 구간을 말한다
#  monitor 가 patience 동안 좋아지지 않으면 learning_rate 에 factor 를 곱한다
#   factor=0.5 -> 0.01 -> 절반 -> 또 절반 ... 으로 떨어진다
#  EarlyStopping 과 역할이 다르다 : es 는 '멈춘다', rlr 은 '보폭을 줄여 더 해 본다'
#   그래서 rlr 의 patience 를 es 보다 짧게 줘야 "줄여 보고 -> 그래도 안 되면 멈춘다" 순서가 된다
#  이 파일 : 시작 learning_rate = 0.01 / es patience = 25 / rlr patience = 20
#
# keras51_augment4_cifar100.py 베이스

import numpy as np
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Model                   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization   # Input -> 함수형 모델의 입력층
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

datagen = ImageDataGenerator(
    # rescale=1./255,             # 픽셀값 0~255 를 0~1 로 (imshow 도 0~1 범위를 그대로 그린다)
    horizontal_flip=True,       # 좌우반전 - 사람 / 사물은 뒤집어도 같은 대상이다
    # vertical_flip=True,       # 상하반전 - 얼굴 사진에는 어울리지 않아 끔
    width_shift_range=0.1,      # 좌우로 최대 10% 이동
    # height_shift_range=0.1,   # 위아래로 이동
    rotation_range=15,          # -15 ~ +15 도 범위에서 회전
    # zoom_range=1.2,           # 확대 / 축소
    # shear_range=0.7,          # 한 점을 고정하고 기울이기
    fill_mode='nearest',        # 이동 / 회전으로 생긴 빈 자리를 가장 가까운 픽셀값으로 채운다
)

augment_size = 40000

print(x_train.shape[0])         # 50000  <- cifar100 은 5만 장이다
randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False) # 5만 개 데이터 중 4만 개 랜덤 뽑기, 중복 불가
                                # replace=False 는 원본 장수(5만)보다 많이 뽑을 수 없다 -> 4만은 가능
print(randidx)                  # [17907 45366 39807 ... 46953 30618  1412] -> 벡터 형태 확인 가능
print(len(randidx))             # 40000 -> 뽑은 개수 확인

print(np.min(randidx), np.max(randidx)) # 0 ~ 49,999 사이에서 뽑은 값의 최소 / 최대 (매번 다름)

x_augmented = x_train[randidx].copy()   # 새로운 메모리 공간 할당해서 저장
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape) # (40000, 32, 32, 3) (40000, 1)
                                            # 컬러라 처음부터 4차원이어서 reshape 없이 flow 에 넣을 수 있다

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,              # 원본 순서를 유지해야 y_augmented 와 짝이 맞는다
).next()[0]                     # [0] = x, [1] = y

x_train = np.concatenate(
    [x_train, x_augmented],
    axis=0
)

y_train = np.concatenate(
    [y_train, y_augmented],
    axis=0
)

x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5
print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

x_train = x_train.reshape(-1, 32 * 32 * 3)
x_test = x_test.reshape(-1, 32 * 32 * 3)
print(x_train.shape, x_test.shape) # (90000, 3072) (10000, 3072)

ohe = OneHotEncoder(sparse_output=False)   # 희소행렬이 아니라 numpy 배열로 받는다

y_train = ohe.fit_transform(y_train)   # train 으로 기준(클래스 목록)을 만든다
y_test = ohe.transform(y_test)         # test 는 그 기준으로 변환만

print(y_train.shape, y_test.shape) # (90000, 100) (10000, 100) -> 증폭본을 합친 뒤라 90000

#2. 모델 구성 (함수형) - keras41_dnn4 와 층 / BatchNormalization / Dropout 위치가 똑같다
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

output1 = Dense(100, activation='softmax')(drop4)   # 100종 분류 -> 출력 100개 + softmax

model = Model(inputs=input1, outputs=output1)   # 시작(input1) ~ 끝(output1) 범위를 정해서 모델 완성

model.summary()

#3. 컴파일, 훈련
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate),
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

# ===== GPU 기록 ===== <- 증폭 적용
# Epoch 260: early stopping
# loss : 3.504183530807495
# acc : 0.3325999975204468
# accuracy_score : 0.3326
# 소요 시간 : 889.23 초

# ===== GPU 기록 ===== <- learning rate 적용
# loss : 3.5019049644470215
# acc : 0.31439998745918274
# accuracy_score : 0.3144
# 소요 시간 : 759.51 초

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후

# Epoch 234: early stopping
# loss : 3.6142590045928955
# acc : 0.3269999921321869
# accuracy_score : 0.327
# 소요 시간 : 738.67 초

# [결론] 0.01 고정 0.3144 -> 같은 0.01 에서 시작 + ReduceLROnPlateau 0.3270 (Epoch 234).
#        시작값이 같은데 결과가 올랐다 -> 줄여 나간 것 자체가 효과를 낸 경우다.
