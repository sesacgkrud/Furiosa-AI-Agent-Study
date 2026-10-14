# keras76_all_T_01_cifar10.py 베이스 (model_list · 동결 · GAP 설명은 cifar10 파일 참고)
#  + keras73_vgg16_05_man_woman.py 의 데이터 준비 (val 분리 · 여자 사진 증폭 설명은 keras73_vgg16_05 참고)
# [목적] 같은 model_list 를 man_woman (남자 / 여자 이진 분류) 에 붙여 보고 모델별 test loss · acc · 소요 시간을 비교한다
#        데이터 준비를 keras73_vgg16_05 와 똑같이 해서 VGG16 한 개로 한 결과와도 비교할 수 있다
#
# cifar10 파일과 다른 점
#  - 데이터 : keras46_03 이 저장한 man_woman npy (100x100 컬러) -> val 분리 -> 여자 사진만 (남 - 여) 장 증폭
#  - input_shape : (32, 32, 3) -> (100, 100, 3)
#  - 스케일링 없음 : npy 를 저장할 때 rescale=1./255 로 이미 0 ~ 1 이다
#  - InceptionV3 · InceptionResNetV2 : 입력이 최소 75x75 라서 32x32 cifar 에는 못 넣었지만 100x100 은 넣을 수 있다
#  - validation_split 대신 validation_data=(x_val, y_val) : 증폭 전에 떼어 둔 원본 val 로 검증한다

import warnings
warnings.filterwarnings('ignore') # console 창에 나오는 warnings 안 보이게 함 (나오는 경우도 있지만, 대부분 해결)

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception

model_list = [VGG19(include_top=False, input_shape=(100, 100, 3)),
              ResNet50(include_top=False, input_shape=(100, 100, 3)),
              ResNet101(include_top=False, input_shape=(100, 100, 3)),
              DenseNet201(include_top=False, input_shape=(100, 100, 3)),
              InceptionV3(include_top=False, input_shape=(100, 100, 3)),
              InceptionResNetV2(include_top=False, input_shape=(100, 100, 3)),
              MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
            #   EfficientNetB0(include_top=False, input_shape=(100, 100, 3)),
              ]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

np_path = './_data/man_woman_npy/'
x_train = np.load(np_path + 'keras46_03_x_train.npy')
y_train = np.load(np_path + 'keras46_03_y_train.npy')
x_test = np.load(np_path + 'keras46_03_x_test.npy')
y_test = np.load(np_path + 'keras46_03_y_test.npy')

print(x_train.shape, y_train.shape)         # (21733, 100, 100, 3) (21733,)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y_train,
)

datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    rotation_range=15,
    fill_mode='nearest',
)

man_idx = np.where(y_train == 0)[0]
woman_idx = np.where(y_train == 1)[0]

augment_size = len(man_idx) - len(woman_idx)    # 11313 - 6073 = 5240

randidx = np.random.randint(len(woman_idx), size=augment_size)
pick = woman_idx[randidx]

x_augmented = x_train[pick].copy()
y_augmented = y_train[pick].copy()

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)         # (22626, 100, 100, 3) (22626,)
print(np.unique(y_train, return_counts=True))

for model_i in model_list:
    model_i.trainable = False       # 동결 O
    # model_i.trainable = True      # 동결 X

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(1, activation='sigmoid'))   # y 가 0 / 1 (남 / 여) 이진 분류

    #3. 컴파일, 훈련
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    )

    start_time = time.time()

    model.fit(x_train,
              y_train,
              epochs=9999999,
              batch_size=32,
              verbose=0,
              validation_data=(x_val, y_val),
              callbacks=[es]
    )

    end_time = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('==============================')
    print('모델명 :', model_i.name, 'loss :', loss[0], 'acc :', loss[1],
          '시간 :', round(end_time - start_time, 2), '초')

# ==============================
