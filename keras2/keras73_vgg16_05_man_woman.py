# keras52_ReduceLR15_man_woman.py 베이스 + keras73_vgg16_01_cifar10.py 의 VGG16 전이학습
# [목적] Day18 에서 Conv2D 를 직접 쌓아 훈련했던 man_woman (남자 / 여자) 데이터에 VGG16 전이학습을 적용한다
#  - 데이터 준비 (val 분리 · 여자 사진 증폭) · learning_rate · ReduceLROnPlateau · 평가 코드는 베이스 그대로
#  - 모델만 VGG16 (imagenet, include_top=False, 동결) + GlobalAveragePooling2D + Dense 로 바꿨다

import numpy as np
import time
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import tensorflow as tf

from keras.applications import VGG16

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
np_path = './_data/man_woman_npy/'
x_train = np.load(np_path + 'keras46_03_x_train.npy')
y_train = np.load(np_path + 'keras46_03_y_train.npy')
x_test = np.load(np_path + 'keras46_03_x_test.npy')
y_test = np.load(np_path + 'keras46_03_y_test.npy')

print(x_train.shape, y_train.shape)
print(np.unique(y_train, return_counts=True))

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y_train,
)

print(x_train.shape, x_val.shape)
print(np.unique(y_train, return_counts=True))

datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    rotation_range=15,
    fill_mode='nearest',
)

man_idx = np.where(y_train == 0)[0]
woman_idx = np.where(y_train == 1)[0]
print(len(man_idx), len(woman_idx))

augment_size = len(man_idx) - len(woman_idx)
print(augment_size)

randidx = np.random.randint(len(woman_idx), size=augment_size)
pick = woman_idx[randidx]

x_augmented = x_train[pick].copy()
y_augmented = y_train[pick].copy()

print(x_augmented.shape, y_augmented.shape)
print(np.unique(y_augmented, return_counts=True))

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)
print(np.unique(y_train, return_counts=True))

#2. 모델 구성
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(100, 100, 3)
)

# imagenet 으로 미리 훈련된 VGG16 가중치 고정 (Dense 만 훈련)
vgg16.trainable = False

model = Sequential()
model.add(vgg16)

# VGG16 출력 (3, 3, 512) -> 채널마다 평균 (512,) : Flatten (4608,) 보다 Dense 파라미터가 9배 적다
model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0018

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=5,                 # es patience(10) 보다 짧게 -> es 가 멈추기 전에 rlr 이 학습률을 줄일 수 있다
    verbose=1,
    factor=0.5,
)

#################### MCP SAVE 파일명 만들기 ####################
import datetime
date = datetime.datetime.now()

date = date.strftime('%m%d_%H%M')

path = './_save/keras73/'
filename = '{epoch:04d}_{val_acc:.4f}.keras'
filepath = ''.join([path, 'k73_05_', date, '_', filename])

#################### #################### ####################

mcp = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    filepath=filepath,
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[es, mcp, rlr],
          )

end_time = time.time()

#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

########## vgg16 적용 ##########
# Epoch 20: early stopping
# loss : 0.3055677115917206
# acc : 0.8728376626968384
# accuracy_score : 0.8728376886271623
# 소요 시간 : 411.34 초
#
# 베이스 (keras52_ReduceLR15, Conv2D 직접 쌓기) 0.9067 / 538.36초 -> VGG16 동결 0.8728 / 411.34초
#  - 시간은 줄었지만 acc 는 베이스보다 낮다
#  - imagenet 특징을 고정하고 Dense 만 훈련해서 얼굴 남 / 여 구분에 딱 맞는 특징을 새로 배우지 못한 것으로 볼 수 있다
