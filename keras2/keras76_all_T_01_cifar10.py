# keras75_All_TransferLearning.py 베이스 (keras.applications 모델 import · model_list)
# [목적] keras.applications 의 여러 전이학습 모델을 model_list 에 담아 for 문으로 하나씩 cifar10 에 붙여 보고,
#        어떤 모델이 cifar10 에 잘 맞는지 test loss · acc · 소요 시간으로 비교한다
#
# 흐름
#  1) model_list : include_top=False + input_shape=(32, 32, 3) 으로 모델마다 특징 추출 부분만 불러온다
#  2) for model_i in model_list :
#       model_i.trainable = False 로 동결 -> Sequential 에 층처럼 넣고 GlobalAveragePooling2D + Dense 를 붙인다
#       -> compile · fit · evaluate -> 모델명 · loss · acc · 소요 시간 출력
#  3) 모든 모델이 같은 데이터 · 같은 Dense · 같은 compile / fit 설정을 쓴다 -> 모델(특징 추출 부분) 만 바꿨을 때를 비교
#
# GlobalAveragePooling2D 를 쓰는 이유
#  - 모델마다 마지막 출력 크기가 다르다 (VGG19 (1, 1, 512), ResNet50 (1, 1, 2048), DenseNet201 (1, 1, 1920) 등)
#  - GAP 는 (가로, 세로, 채널) 을 채널마다 평균내서 (채널,) 로 만든다 -> 어떤 모델이 와도 같은 코드로 Dense 를 붙일 수 있다

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

# include_top=False : 분류 부분(Flatten · Dense) 을 빼야 input_shape 를 (32, 32, 3) 으로 바꿀 수 있다 (Day28 keras72_1)
model_list = [VGG19(include_top=False, input_shape=(32, 32, 3)),
              ResNet50(include_top=False, input_shape=(32, 32, 3)),
              ResNet101(include_top=False, input_shape=(32, 32, 3)),
              DenseNet201(include_top=False, input_shape=(32, 32, 3)),
            #   InceptionV3(), InceptionResNetV2(), # 남자/여자, 개/고양이 와 같은 큰 데이터에는 활용
              MobileNetV2(include_top=False, input_shape=(32, 32, 3)),
            #   EfficientNetB0(include_top=False, input_shape=(32, 32, 3)),
              ]
# InceptionV3 · InceptionResNetV2 는 입력이 최소 75x75 라서 32x32 cifar 에는 넣을 수 없다
# MobileNetV2 는 imagenet 가중치가 (224, 224) 기준이라 32x32 를 넣어도 이름이 mobilenetv2_1.00_224 로 나온다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import time
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# 모델마다 똑같은 데이터를 쓰므로 for 문 밖에서 한 번만 불러온다
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 스케일링
x_train = x_train/255.
x_test = x_test/255.

for model_i in model_list:
    model_i.trainable = False       # 동결 O : 불러온 모델의 imagenet 가중치는 고정, 아래 Dense 만 훈련
    # model_i.trainable = True      # 동결 X : 불러온 모델까지 전부 훈련

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(10, activation='softmax'))

    #3. 컴파일, 훈련
    # y 가 원핫이 아닌 정수 (50000, 1) 라서 sparse_categorical_crossentropy (Day23)
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
              epochs=9999999,       # 사실상 무한 -> EarlyStopping 이 멈출 때까지 훈련
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

# ===== 동결 X (불러온 모델까지 전부 훈련된 상태) 기록 =====
# ==============================
# 모델명 : vgg19 loss : 1.344775676727295 acc : 0.476500004529953
# ==============================
# 모델명 : resnet50 loss : 4.328162670135498 acc : 0.20090000331401825
# ==============================
# 모델명 : resnet101 loss : 2.5120909214019775 acc : 0.3361000120639801
# ==============================
# 모델명 : densenet201 loss : 1.0415445566177368 acc : 0.6588000059127808
# ==============================
# 모델명 : mobilenetv2_1.00_224 loss : 1.8287359476089478 acc : 0.3926999866962433
#
# densenet201 0.6588 > vgg19 0.4765 > mobilenetv2 0.3927 > resnet101 0.3361 > resnet50 0.2009
#  - 같은 Dense · 같은 설정인데 모델에 따라 acc 가 3배 넘게 차이 난다
#  - 모두 Day28 keras73_vgg16_01 (VGG16 동결 X 0.8158) · 이전 최고 (keras40_GAP03 0.7727) 보다 낮다
