# keras47_02_load_npy_rps.py 베이스 + keras73_vgg16_03_man_horse.py 의 VGG16 모델
# [목적] Day16 에서 Conv2D 를 직접 쌓아 훈련했던 rps (가위 / 바위 / 보 3종) 데이터에 VGG16 전이학습을 적용한다
#  - 데이터 · compile · EarlyStopping · ModelCheckpoint · 평가 코드는 베이스 그대로, 모델만 VGG16 + Flatten + Dense

import numpy as np
import time
import random

from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout    # Conv2D, MaxPooling2D 는 VGG16 안에 있으므로 import 안 함
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
from keras.applications import VGG16

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
########## 방법 1 : ImageDataGenerator 로 폴더에서 읽기 (시간 측정용) ##########
start_time1 = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/rps/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),
    batch_size=2049,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
)

x_idg = xy_train[0][0]
y_idg = xy_train[0][1]

end_time1 = time.time()

print(x_idg.shape, y_idg.shape)

del xy_train, x_idg, y_idg

########## 방법 2 : npy 불러오기 ##########
start_time2 = time.time()

np_path = './_data/rps_npy/'
x_train = np.load(np_path + 'keras46_02_x_train.npy')
y_train = np.load(np_path + 'keras46_02_y_train.npy')
x_test = np.load(np_path + 'keras46_02_x_test.npy')
y_test = np.load(np_path + 'keras46_02_y_test.npy')

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

vgg16.trainable=False      # 가중치 동결
# vgg16.trainable=True     # 가중치 동결 X

# 베이스에서 직접 쌓던 Conv2D · MaxPooling2D 대신 vgg16 을 층 하나로 넣는다
model = Sequential()
model.add(vgg16)                # 출력 (None, 3, 3, 512) : 100 -> 50 -> 25 -> 12 -> 6 -> 3
model.add(Flatten())            # (3, 3, 512) -> (4608,)

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=20,
    restore_best_weights=True,
    verbose=1,
)

#################### MCP SAVE 파일명 만들기 ####################
import datetime
date = datetime.datetime.now()

date = date.strftime('%m%d_%H%M')

path = './_save/keras73/'
filename = '{epoch:04d}_{val_acc:.4f}.keras'
filepath = ''.join([path, 'k73_04_', date, '_', filename])

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
          validation_split=0.2,
          callbacks=[es, mcp],
          )

end_time = time.time()

#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)
y_predict_arg = np.argmax(y_predict, axis=1)
y_test_arg = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

########## vgg16 적용 ##########
# Epoch 21: early stopping
# loss : 0.025024868547916412
# acc : 1.0
# accuracy_score : 1.0
# 소요 시간 : 29.77 초
