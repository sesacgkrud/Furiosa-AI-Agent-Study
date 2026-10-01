# keras52_ReduceLR15_man_woman.py
# man_woman (이진 분류, CNN + npy) - ReduceLROnPlateau 로 훈련 도중 learning_rate 를 줄인다
# 여자만 증폭(keras51_augment5) 해 남 / 여를 1 : 1 로 맞춘 데이터에 learning_rate + ReduceLROnPlateau 를 붙인다
# val 은 증폭 전에 train_test_split 으로 떼어 둔 원본 그대로 쓴다
#
# [ 오늘 추가되는 곳은 콜백 하나다 ]
#  keras52_optimizer 는 learning_rate 를 처음부터 끝까지 고정했다
#  하지만 한 값으로 두 가지를 다 할 수는 없다
#   초반에는 크게 움직여 빨리 내려가야 하고, 최저점 근처에서는 작게 움직여야 지나치지 않는다
#
# [ ReduceLROnPlateau : 훈련 도중 learning_rate 를 줄여 준다 ]
#  plateau = 고원. val_loss 가 더 이상 줄지 않고 평평해지는 구간을 말한다
#  monitor 가 patience 동안 좋아지지 않으면 learning_rate 에 factor 를 곱한다
#   factor=0.5 -> 0.0018 -> 절반 -> 또 절반 ... 으로 떨어진다
#  EarlyStopping 과 역할이 다르다 : es 는 '멈춘다', rlr 은 '보폭을 줄여 더 해 본다'
#   그래서 rlr 의 patience 를 es 보다 짧게 줘야 "줄여 보고 -> 그래도 안 되면 멈춘다" 순서가 된다
#  이 파일 : 시작 learning_rate = 0.0018 / es patience = 10 / rlr patience = 20
#   -> es 가 더 짧아서 lr 을 줄여 보기도 전에 훈련이 끝난다 (아래 [결론] 참고)
#
# keras51_augment5_man_woman_여자만.py 베이스

import numpy as np
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1. 데이터
np_path = './_data/man_woman_npy/'
x_train = np.load(np_path + 'keras46_03_x_train.npy')
y_train = np.load(np_path + 'keras46_03_y_train.npy')
x_test = np.load(np_path + 'keras46_03_x_test.npy')
y_test = np.load(np_path + 'keras46_03_y_test.npy')

print(x_train.shape, y_train.shape)         # (21733, 100, 100, 3) (21733,)
print(np.unique(y_train, return_counts=True))

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y_train,
)

print(x_train.shape, x_val.shape)           # (17386, 100, 100, 3) (4347, 100, 100, 3)
print(np.unique(y_train, return_counts=True))
# (array([0., 1.]), array([11313, 6073]))   # val 쪽은 2829 / 1518

datagen = ImageDataGenerator(
    horizontal_flip=True,       # 좌우반전 - 얼굴은 뒤집어도 같은 사람이다
    # vertical_flip=True,       # 상하반전 - 얼굴 사진에는 어울리지 않아 끔
    width_shift_range=0.1,      # 좌우로 최대 10% 이동
    # height_shift_range=0.1,   # 위아래로 이동
    rotation_range=15,          # -15 ~ +15 도 범위에서 회전
    # zoom_range=1.2,           # 확대 / 축소
    # shear_range=0.7,          # 한 점을 고정하고 기울이기
    fill_mode='nearest',        # 이동 / 회전으로 생긴 빈 자리를 가장 가까운 픽셀값으로 채운다
)

man_idx = np.where(y_train == 0)[0]
woman_idx = np.where(y_train == 1)[0]
print(len(man_idx), len(woman_idx))     # val 을 뗀 뒤의 남 / 여 원본 장수

augment_size = len(man_idx) - len(woman_idx)
print(augment_size)             # 11313 - 6073 = 5240

randidx = np.random.randint(len(woman_idx), size=augment_size)
pick = woman_idx[randidx]       # 여자 인덱스 중에서 augment_size 개 (전체 인덱스가 아니라 여자 목록 안에서 고른다)

x_augmented = x_train[pick].copy()   # .copy() 로 새 메모리에 담는다 (원본이 함께 바뀌는 것을 막는다)
y_augmented = y_train[pick].copy()

print(x_augmented.shape, y_augmented.shape)     # (augment_size, 100, 100, 3) (augment_size,)
print(np.unique(y_augmented, return_counts=True))   # (array([1.]), array([augment_size])) -> 전부 여자인지 확인

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,              # 원본 순서를 유지해야 y_augmented 와 짝이 맞는다
).next()[0]                     # [0] = x, [1] = y

print(x_augmented.shape)        # (augment_size, 100, 100, 3)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)         # (22626, 100, 100, 3) (22626,)
print(np.unique(y_train, return_counts=True))

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (5,5), padding='same', activation='relu', input_shape=(100, 100, 3)))
                                # padding='same' -> 크기 유지 : (100, 100, 64)
model.add(Conv2D(64, (5,5), activation='relu'))
                                # padding 없음 -> (커널 - 1) 만큼 감소 : (96, 96, 64)
model.add(MaxPooling2D())       # 절반으로 : (48, 48, 64), 파라미터 0
model.add(Dropout(0.2))

model.add(Conv2D(32, (5,5), padding='same', activation='relu'))
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())            # 4차원 -> 2차원

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

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
learning_rate = 0.0018

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'])

# 목표가 정확도라서 val_loss 가 아니라 val_acc 를 기준으로 잡는다
#  - loss 는 낮을수록 좋고 acc 는 높을수록 좋으므로 mode 도 'max' 로 같이 바꿔야 한다
es = EarlyStopping(
    monitor='val_acc',          # val_loss 최저점 epoch 와 val_acc 최고점 epoch 는 서로 다르다
    mode='max',                 # acc 는 높을수록 좋으므로 max
    patience=10,                # 데이터가 많아 한 epoch 가 길다 -> 너무 오래 기다리지 않도록 짧게
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

#################### MCP SAVE 파일명 만들기 ####################
# 실행할 때마다 이전 파일이 덮어써지지 않도록 날짜 / 시간을 파일명에 넣는다
import datetime
date = datetime.datetime.now()  # 현재 시간 반환

date = date.strftime('%m%d_%H%M')

path = './_save/keras51/'
filename = '{epoch:04d}_{val_acc:.4f}.keras'    # mcp 가 보는 기준(val_acc)을 파일명에도 그대로 쓴다
filepath = ''.join([path, 'k51_05_', date, '_', filename])

## 만들어지는 파일명 예)
# './_save/keras51/' + 'k51_05_' + '0922_1147' + '0053_0.9321.keras'

#################### #################### ####################

mcp = ModelCheckpoint(
    monitor='val_acc',          # es 와 기준이 다르면 저장되는 epoch 와 되돌리는 epoch 가 어긋난다
    mode='max',                 # acc 는 높을수록 좋으므로 max
    save_best_only=True,        # 기록이 갱신될 때만 저장 -> 마지막에 남는 파일이 가장 좋은 모델
    filepath=filepath,
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1,
          validation_data=(x_val, y_val),   # 증폭 전에 떼어 둔 원본 val 로 검증한다
          callbacks=[es, mcp, rlr],      # callbacks 에 넣어야 실제로 동작한다
          )

end_time = time.time()

#4. 평가 예측
# test 는 증폭하지 않은 원본 그대로다 (실제로 들어올 데이터와 같아야 평가가 의미 있다)
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

# predict 결과는 (5434, 1) 짜리 sigmoid 확률 -> 0.5 기준 반올림해야 0 / 1 정답과 비교할 수 있다
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# 여자만 (남 - 여) 장 증폭, val 은 증폭 전에 train_test_split 으로 분리
# 비교 대상 : keras47_03 (증폭 없음) acc 0.908
#  -> test acc 는 거의 같다. 대신 남 / 여를 1 : 1 로 맞춰 한쪽으로 기우는 것을 막았다

# ===== 기록 ===== <- 여자만 5240장 증폭
# loss : 0.27418696880340576
# acc : 0.908170759677887
# accuracy_score : 0.9081707765918292
# 소요 시간 : 427.09 초

# ===== GPU 기록 ===== <- learning rate 적용
# Epoch 29: early stopping
# loss : 0.34509173035621643
# acc : 0.89952152967453
# accuracy_score : 0.8995215311004785
# 소요 시간 : 653.36 초

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후

# Epoch 24: early stopping
# loss : 0.3432677388191223
# acc : 0.9066985845565796
# accuracy_score : 0.9066985645933014
# 소요 시간 : 538.36 초

# [결론] 0.0018 고정 0.8995 / 653초 -> ReduceLROnPlateau 0.9067 / 538초 (Epoch 24).
#        [주의] es patience=10 / rlr patience=20 이라 24 epoch 에서 es 가 먼저 걸렸다.
#        rlr 이 한 번도 작동하지 못한 상태의 기록이다 -> 이 차이는 rlr 효과가 아니다.
#        rlr 을 쓰려면 es patience 를 rlr patience 보다 길게 잡아야 한다.
