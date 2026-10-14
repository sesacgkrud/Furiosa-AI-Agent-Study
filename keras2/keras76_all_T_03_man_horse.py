# keras76_all_T_01_cifar10.py 베이스 (model_list · 동결 · GAP 설명은 cifar10 파일 참고)
# [목적] 같은 model_list 를 man_horse (사람 / 말 2종, 원핫) 에 붙여 보고 모델별 test loss · acc · 소요 시간을 비교한다
#        데이터는 keras73_vgg16_03_man_horse 와 같은 npy 를 쓴다 -> VGG16 한 개로 한 결과와도 비교할 수 있다
#
# cifar10 파일과 다른 점
#  - 데이터 : keras46_01 이 저장한 horse npy (100x100 컬러, train / test 분리까지 끝난 상태)
#  - input_shape : (32, 32, 3) -> (100, 100, 3)
#  - 스케일링 없음 : npy 를 저장할 때 rescale=1./255 로 이미 0 ~ 1 이다 (한 번 더 나누면 0 ~ 0.004 가 된다)

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
            #   InceptionV3(), InceptionResNetV2(), # 남자/여자, 개/고양이 와 같은 큰 데이터에는 활용
              MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
            #   EfficientNetB0(include_top=False, input_shape=(100, 100, 3)),
              ]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import time
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

np_path = './_data/horse_npy/'
x_train = np.load(np_path + 'keras46_01_x_train.npy')
y_train = np.load(np_path + 'keras46_01_y_train.npy')
x_test = np.load(np_path + 'keras46_01_x_test.npy')
y_test = np.load(np_path + 'keras46_01_y_test.npy')

print(x_train.shape, y_train.shape)     # (821, 100, 100, 3) (821, 2)
print(x_test.shape, y_test.shape)       # (206, 100, 100, 3) (206, 2)

for model_i in model_list:
    model_i.trainable = False       # 동결 O
    # model_i.trainable = True      # 동결 X

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(2, activation='softmax'))   # y 가 원핫 (N, 2) (horses / humans)

    #3. 컴파일, 훈련
    # y 가 원핫이라 categorical_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
              validation_split = 0.2,
              callbacks=[es]
    )

    end_time = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('==============================')
    print('모델명 :', model_i.name, 'loss :', loss[0], 'acc :', loss[1],
          '시간 :', round(end_time - start_time, 2), '초')

# ==============================
