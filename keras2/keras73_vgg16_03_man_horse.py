# keras47_01_load_npy_horse.py 베이스 + keras73_vgg16_01_cifar10.py 의 VGG16 전이학습
# [목적] Day16 에서 Conv2D 를 직접 쌓아 훈련했던 horse-human (사람 / 말) 데이터에 VGG16 전이학습을 적용한다
#  - 데이터 · compile · EarlyStopping · ModelCheckpoint · 평가 코드는 베이스 그대로
#  - 모델만 Conv2D 직접 쌓기 -> VGG16 (imagenet, include_top=False, 동결) + Flatten + Dense 로 바꿨다

import numpy as np
import time
import random

from keras.preprocessing.image import ImageDataGenerator
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

path_train = './_data/image/horse-human/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),
    batch_size=1027,
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

np_path = './_data/horse_npy/'
x_train = np.load(np_path + 'keras46_01_x_train.npy')
y_train = np.load(np_path + 'keras46_01_y_train.npy')
x_test = np.load(np_path + 'keras46_01_x_test.npy')
y_test = np.load(np_path + 'keras46_01_y_test.npy')

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

# 특징 추출은 VGG16 (Conv2D 13개) 이 하므로 Conv2D 를 더 쌓지 않고 분류 부분(Flatten · Dense) 만 붙인다
#  - VGG16 출력이 (3, 3) 이라 MaxPooling2D 를 두 번 넣으면 (1, 1) -> (0, 0) 이 되어 에러가 난다
model.add(Flatten())            # (3, 3, 512) -> (4608,)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

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
filepath = ''.join([path, 'k73_03_', date, '_', filename])

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
# Epoch 30: early stopping
# loss : 0.029129479080438614
# acc : 0.9854369163513184
# accuracy_score : 0.9854368932038835
# 소요 시간 : 23.43 초
