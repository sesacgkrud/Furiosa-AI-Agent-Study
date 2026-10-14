# keras76_all_T_01_cifar10.py 베이스 (model_list · 동결 · GAP 설명은 cifar10 파일 참고)
# [목적] 같은 model_list 를 cifar100 에 붙여 보고 모델별 test loss · acc · 소요 시간을 비교한다
#  - cifar10 과 다른 점 : 데이터 cifar100, 마지막 층 Dense(100, softmax) (100종 분류)

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

model_list = [VGG19(include_top=False, input_shape=(32, 32, 3)),
              ResNet50(include_top=False, input_shape=(32, 32, 3)),
              ResNet101(include_top=False, input_shape=(32, 32, 3)),
              DenseNet201(include_top=False, input_shape=(32, 32, 3)),
            #   InceptionV3(), InceptionResNetV2(), # 남자/여자, 개/고양이 와 같은 큰 데이터에는 활용
              MobileNetV2(include_top=False, input_shape=(32, 32, 3)),
            #   EfficientNetB0(include_top=False, input_shape=(32, 32, 3)),
              ]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar100
import tensorflow as tf
import time
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# y 는 0 ~ 99 정수 (50000, 1)

# 스케일링
x_train = x_train/255.
x_test = x_test/255.

for model_i in model_list:
    model_i.trainable = False       # 동결 O
    # model_i.trainable = True      # 동결 X

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(100, activation='softmax'))     # 100종 분류 -> 출력 100 (10 이면 정답 10 ~ 99 를 표현할 수 없다)

    #3. 컴파일, 훈련
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
