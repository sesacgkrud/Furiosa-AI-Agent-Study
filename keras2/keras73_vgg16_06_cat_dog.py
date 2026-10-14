# keras45_04_catdog_load_npy.py 베이스 + keras73_vgg16_03_man_horse.py 의 VGG16 모델
# [목적] Day15 에서 Conv2D 를 직접 쌓아 훈련했던 cat_dog (고양이 / 개) 데이터에 VGG16 전이학습을 적용한다
#  - 데이터 · compile · EarlyStopping · ModelCheckpoint · 평가 코드는 베이스 그대로, 모델만 VGG16 + Flatten + Dense

import numpy as np
import time
import random

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout    # Conv2D, MaxPooling2D 는 VGG16 안에 있으므로 import 안 함
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

import tensorflow as tf

from keras.applications import VGG16

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
start_time1 = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/cat_dog/training_set/'
path_test = './_data/image/cat_dog/test_set'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),
    batch_size=8005,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=2023,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,
)

x_train_idg = xy_train[0][0]
y_train_idg = xy_train[0][1]
x_test_idg = xy_test[0][0]
y_test_idg = xy_test[0][1]

end_time1 = time.time()

print(x_train_idg.shape, y_train_idg.shape)
print(x_test_idg.shape, y_test_idg.shape)

del xy_train, xy_test, x_train_idg, y_train_idg, x_test_idg, y_test_idg

########## 방법 2 : npy 불러오기 ##########
start_time2 = time.time()

np_path = './_data/kaggle_cat_dog_npy/'
x_train = np.load(np_path + 'keras45_03_x_train.npy')
y_train = np.load(np_path + 'keras45_03_y_train.npy')
x_test = np.load(np_path + 'keras45_03_x_test.npy')
y_test = np.load(np_path + 'keras45_03_y_test.npy')

end_time2 = time.time()

########## 두 방법 시간 비교 ##########
print('ImageDataGenerator 시간 :', round(end_time1 - start_time1, 2), '초')
print('npy 로드 시간 :', round(end_time2 - start_time2, 2), '초')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델 구성
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(100, 100, 3)
)

# trainable 을 적지 않으면 기본값 True (동결 X) -> 동결 O / 동결 X 를 직접 골라 적는다
vgg16.trainable=False      # 가중치 동결
# vgg16.trainable=True     # 가중치 동결 X

model = Sequential()
model.add(vgg16)                # 출력 (None, 3, 3, 512) : 100 -> 50 -> 25 -> 12 -> 6 -> 3

# 특징 추출은 VGG16 이 하므로 베이스의 Conv2D 대신 분류 부분(Flatten · Dense) 만 붙인다
model.add(Flatten())            # (3, 3, 512) -> (4608,)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=20,            # epochs(100) 보다 작게 줘야 실제로 멈추고 restore_best_weights 도 동작한다
    restore_best_weights=True,
    verbose=1,
)

#################### MCP SAVE 파일명 만들기 ####################
import datetime
date = datetime.datetime.now()

date = date.strftime('%m%d_%H%M')

path = './_save/keras73/'
filename = '{epoch:04d}_{val_acc:.4f}.keras'
filepath = ''.join([path, 'k73_06_', date, '_', filename])

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
          validation_split=0.1,
          callbacks=[es, mcp],
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
# Epoch 24: early stopping
# loss : 0.3016343116760254
# acc : 0.8645575642585754
# accuracy_score : 0.8645575877409788
# 소요 시간 : 148.4 초
