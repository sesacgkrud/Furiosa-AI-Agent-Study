# keras51_augment5_man_woman_여자만.py
# man_woman 데이터에서 '여자' 사진만 증폭해 남 / 여 장수를 맞춘다
#
# [ 왜 한쪽만 증폭하는가 ]
#  train 기준 man 14142 / woman 7591 로 남자가 약 2배 많다
#  이대로 훈련하면 모델이 "애매하면 남자" 라고 답하는 쪽으로 기울기 쉽다
#  적은 쪽(woman)만 부족한 장수만큼 늘려 남 / 여를 1 : 1 로 맞춘다
#
# [ 검증 데이터는 증폭 '전에' train_test_split 으로 먼저 뗀다 ]
#  validation_split 은 섞지 않고 맨 뒤 20% 를 뗀다
#  증폭본을 뒤에 붙인 뒤 validation_split 을 쓰면 검증 데이터가 전부 '증폭한 여자 사진' 이 된다
#   -> 훈련에서 여자가 다시 줄어 균형이 깨지고, val_acc 도 여자만 맞히는 비율이 되어 es / mcp 기준이 틀어진다
#  합친 뒤에 나눠도 같은 사진의 원본 / 증폭본이 train 과 val 에 나눠 들어가 val_acc 가 부풀려진다
#  그래서 원본 train 에서 val 을 먼저 떼고, 남은 train 의 여자 사진만 증폭한다 (val 은 원본 그대로)
#
# [ keras51_augment1 ~ 4 와 달라지는 점 ]
#  1) 전체에서 랜덤으로 뽑지 않고 y == 1(woman) 인 것만 골라서 뽑는다
#  2) 여자 사진 안에서 뽑을 때 중복을 허용할지 정해야 한다
#     np.random.randint         -> 중복 허용 (이 파일)
#     np.random.choice(replace=False) -> 중복 불가. 원본(6073)보다 많이 뽑을 수는 없다
#     val 을 뗀 뒤에는 5240장만 뽑으면 되어 둘 다 가능하지만, 뽑을 장수가 원본보다 많아져도 돌아가도록 randint 를 쓴다
#  3) 데이터가 이미 0~1 이다
#     mnist / cifar 는 0~255 원본이라 합친 뒤 스케일링했지만,
#     여기서 쓰는 npy 는 keras46_03 이 rescale=1./255 로 읽어 저장한 것이라 스케일링이 끝나 있다
#     -> datagen 에도 rescale 을 쓰지 않고, 합친 뒤에도 나누지 않는다 (두 번 나누면 값이 0 근처로 뭉개진다)

import numpy as np
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
########## npy 불러오기 (keras46_03 이 저장한 것) ##########
np_path = './_data/man_woman_npy/'
x_train = np.load(np_path + 'keras46_03_x_train.npy')
y_train = np.load(np_path + 'keras46_03_y_train.npy')
x_test = np.load(np_path + 'keras46_03_x_test.npy')
y_test = np.load(np_path + 'keras46_03_y_test.npy')

print(x_train.shape, y_train.shape)         # (21733, 100, 100, 3) (21733,)
print(np.unique(y_train, return_counts=True))
# (array([0., 1.]), array([14142, 7591]))   # man 0 / woman 1 -> 남자가 약 2배

########## train / val 분리 (증폭 전에) ##########
# stratify=y_train -> val 에도 남 / 여가 원래 비율(약 2 : 1) 그대로 들어간다
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

#################### 여기부터 증폭이다 ####################

# 이미 0~1 로 스케일링된 데이터라 rescale 은 쓰지 않는다
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

########## 여자 사진만 골라내기 ##########
# np.where 는 조건을 만족하는 위치(인덱스)를 돌려준다 -> 여자 사진의 번호만 모인다
man_idx = np.where(y_train == 0)[0]
woman_idx = np.where(y_train == 1)[0]
print(len(man_idx), len(woman_idx))     # val 을 뗀 뒤의 남 / 여 원본 장수

# 부족한 장수(남 - 여)만큼만 증폭해 1 : 1 로 맞춘다
# 8000 처럼 숫자로 고정하면 val 을 뗀 뒤에는 오히려 여자가 더 많아진다
augment_size = len(man_idx) - len(woman_idx)
print(augment_size)             # 11313 - 6073 = 5240

# randint 는 중복을 허용하므로 같은 사진이 여러 번 뽑힐 수 있다
# 같은 사진이라도 flow 를 거치면 뒤집기 / 이동 / 회전이 매번 다르게 적용돼 서로 다른 사진이 된다
randidx = np.random.randint(len(woman_idx), size=augment_size)
pick = woman_idx[randidx]       # 여자 인덱스 중에서 augment_size 개 (전체 인덱스가 아니라 여자 목록 안에서 고른다)

x_augmented = x_train[pick].copy()   # .copy() 로 새 메모리에 담는다 (원본이 함께 바뀌는 것을 막는다)
y_augmented = y_train[pick].copy()

print(x_augmented.shape, y_augmented.shape)     # (augment_size, 100, 100, 3) (augment_size,)
print(np.unique(y_augmented, return_counts=True))   # (array([1.]), array([augment_size])) -> 전부 여자인지 확인

########## 뽑은 이미지를 변형 ##########
# batch_size 를 전체 장수로 주면 한 번에 다 변형된다 -> next() 로 그 한 묶음을 꺼낸다
# flow 결과는 (x, y) 튜플이라 [0] 으로 x 만 받는다 (y 는 변형되지 않아 y_augmented 를 그대로 쓴다)
x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,              # 원본 순서를 유지해야 y_augmented 와 짝이 맞는다
).next()[0]                     # [0] = x, [1] = y

print(x_augmented.shape)        # (augment_size, 100, 100, 3)

########## 원본 + 증폭 데이터 합치기 ##########
# 원본과 증폭본 모두 0~1 이므로 합친 뒤 따로 나누지 않는다
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)         # (22626, 100, 100, 3) (22626,)
print(np.unique(y_train, return_counts=True))
# (array([0., 1.]), array([11313, 11313]))  # 남 / 여 1 : 1 (val 은 원본 그대로 약 2 : 1)

#2. 모델 구성
# keras47_03 과 같은 구조 (이미지라서 Conv2D)
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

# 이진 분류 출력층은 sigmoid 가 있어야 0 ~ 1 확률이 나온다
# man=0 / woman=1 로 훈련하므로 예측값은 '여자일 확률' 이 된다
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
# sigmoid 1칸 조합이라 loss 는 binary_crossentropy
model.compile(loss='binary_crossentropy', optimizer='adam',
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
          callbacks=[es, mcp],      # callbacks 에 넣어야 실제로 동작한다
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