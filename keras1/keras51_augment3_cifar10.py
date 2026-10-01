# keras51_augment3_cifar10.py
# cifar10 데이터에 증폭(augmentation)을 적용해 훈련 데이터를 늘린다 - keras50_flow2_next.py 베이스
#
# [ 증폭의 흐름 ]
#  1) 원본에서 augment_size 만큼 인덱스를 랜덤으로 뽑는다
#  2) 뽑은 이미지를 datagen.flow 에 통과시켜 변형본을 만든다
#  3) 원본 + 변형본을 concatenate 로 합쳐 훈련 데이터를 늘린다
#  -> 사진을 새로 찍지 않고도 "같은 대상의 조금 다른 모습" 을 학습시킬 수 있다 (과적합 완화)
#
# [ mnist / fashion 과 달라지는 점 ]
#  - 컬러라 처음부터 4차원 (50000, 32, 32, 3) -> 증폭 전 reshape 이 필요 없다
#  - y 도 (50000, 1) 2차원이라 원핫에 바로 넣을 수 있다
#  - 스케일링은 (x - 127.5)/127.5 로 -1 ~ 1 (Day13 부터 컬러 이미지에 쓰던 방식)
#
# [ 주의할 점 ]
#  - datagen 의 rescale 을 끈다 : 원본과 증폭본을 합친 뒤 한 번에 스케일링해야 범위가 어긋나지 않는다
#  - 증폭은 train 에만 한다. test 는 실제로 들어올 데이터 그대로여야 평가가 의미 있다

import numpy as np
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model                   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input   # Input -> 함수형 모델의 입력층
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 1. 데이터
# =========================================================

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)   # (50000, 32, 32, 3)
print(y_train.shape)   # (50000, 1)
print(x_test.shape)    # (10000, 32, 32, 3)
print(y_test.shape)    # (10000, 1)


# =========================================================
# 2. 데이터 증강
# =========================================================

# rescale 은 켜지 않는다 -> 원본과 합친 뒤 아래 4번에서 한 번에 스케일링한다
datagen = ImageDataGenerator(
    horizontal_flip=True,       # 좌우반전 - 뒤집어도 같은 대상이다
    width_shift_range=0.1,      # 좌우로 최대 10% 이동
    rotation_range=15,          # -15 ~ +15 도 범위에서 회전
    fill_mode='nearest',        # 이동 / 회전으로 생긴 빈 자리를 가장 가까운 픽셀값으로 채운다
)

augment_size = 40000

# 50,000개 중에서 40,000개 랜덤 선택 (randint 라 같은 번호가 여러 번 뽑힐 수 있다)
randidx = np.random.randint(
    x_train.shape[0],
    size=augment_size
)

# .copy() 로 새 메모리에 담는다 (원본 x_train 이 함께 바뀌는 것을 막는다)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)   # (40000, 32, 32, 3)
print(y_augmented.shape)   # (40000, 1)


# 40,000장 증강
# batch_size 를 전체 장수로 주면 한 번에 다 변형된다 -> next() 로 그 한 묶음을 꺼낸다
# flow 결과는 (x, y) 튜플이라 [0] 으로 x 만 받는다 (y 는 변형되지 않아 y_augmented 를 그대로 쓴다)
x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,              # 원본 순서를 유지해야 y_augmented 와 짝이 맞는다
).next()[0]                     # [0] = x, [1] = y

print(x_augmented.shape)   # (40000, 32, 32, 3)


# =========================================================
# 3. 원본 + 증강 데이터 합치기
# =========================================================

x_train = np.concatenate(
    [x_train, x_augmented],
    axis=0
)

y_train = np.concatenate(
    [y_train, y_augmented],
    axis=0
)

print(x_train.shape)   # (90000, 32, 32, 3)
print(y_train.shape)   # (90000, 1)


# =========================================================
# 4. 스케일링
# =========================================================

# 합친 뒤 스케일링한다 -> 원본과 증폭본이 같은 범위가 된다
# x_test 도 반드시 같은 식으로 바꾼다 (훈련과 평가의 범위가 다르면 acc 가 크게 떨어진다)
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5


# =========================================================
# 5. Dense 입력을 위해 2차원으로 변환
# =========================================================

x_train = x_train.reshape(-1, 32 * 32 * 3)
x_test = x_test.reshape(-1, 32 * 32 * 3)

print(x_train.shape)   # (90000, 3072)
print(x_test.shape)    # (10000, 3072)


# =========================================================
# 6. y 원핫 인코딩
# =========================================================

ohe = OneHotEncoder(sparse_output=False)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(y_train.shape)   # (90000, 10)
print(y_test.shape)    # (10000, 10)

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

# ===== GPU 기록 ===== <- 증폭 적용
# Epoch 143: early stopping
# loss : 2.7626540660858154
# acc : 0.5432999730110168
# accuracy_score : 0.5433
# 소요 시간 : 306.04 초