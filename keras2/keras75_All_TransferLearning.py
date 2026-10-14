# [목적] keras.applications 에 들어 있는 imagenet 전이학습 모델들을 불러와서
#        모델마다 전체 가중치 개수(model.weights) 와 훈련 가능한 가중치 개수(model.trainable_weights) 를 비교한다
#
#  - keras.applications : VGG16 (Day28) 말고도 VGG19 · ResNet · DenseNet · Inception · MobileNet · NASNet · EfficientNet · Xception 등
#                         미리 훈련된 모델을 같은 방식(weights · include_top · input_shape) 으로 불러올 수 있다
#  - 인자 없이 VGG16() 처럼 부르면 include_top=True, imagenet 가중치, 모델마다 정해진 원래 입력 크기 (VGG · ResNet 224, Inception · Xception 299 등)
#  - model_list : 모델을 리스트에 담아 두고 for 문으로 같은 출력 코드를 반복한다
#  - model.name : 모델 이름 (vgg16, resnet50 ...)

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

model_list = [VGG16(), VGG19(), ResNet50(), ResNet50V2(), ResNet101(), ResNet101V2(), ResNet152(), ResNet152V2(),
              DenseNet201(), DenseNet169(), DenseNet121(), InceptionV3(), InceptionResNetV2(),
              MobileNet(), MobileNetV2(), MobileNetV3Small(), MobileNetV3Large(), NASNetMobile(), NASNetLarge(),
              EfficientNetB0(), EfficientNetB1(), EfficientNetB2(),
              ]

model = Xception()      # for 문 전에 Xception 하나만 먼저 summary 와 가중치 개수를 본다

model.trainable = True

model.summary()
# Total params: 22,910,480 (87.40 MB)
# Trainable params: 22,855,952 (87.19 MB)

print('==============================')
print('모델명 : ', model.name)
print('전체 가중치 갯수 : ', len(model.weights))
print('전체 훈련 가능 가중치 갯수 : ', len(model.trainable_weights))

for model in model_list:
    # model.trainable = False
    print('==============================')
    print('모델명 : ', model.name)
    print('전체 가중치 갯수 : ', len(model.weights))
    print('전체 훈련 가능 가중치 갯수 : ', len(model.trainable_weights))


# ===== 실행 결과 : 모델명 / 전체 가중치 갯수 / 전체 훈련 가능 가중치 갯수 =====
# xception                 236   156
# vgg16                     32    32
# vgg19                     38    38
# resnet50                 320   214
# resnet50v2               272   174
# resnet101                626   418
# resnet101v2              544   344
# resnet152                932   622
# resnet152v2              816   514
# densenet201             1006   604
# densenet169              846   508
# densenet121              606   364
# inception_v3             378   190
# inception_resnet_v2      898   490
# mobilenet_1.00_224       137    83
# mobilenetv2_1.00_224     262   158
# MobileNetV3Small         210   142
# MobileNetV3Large         266   174
# nasnet_mobile           1126   742
# nasnet_large            1546  1018
# efficientnetb0           314   213
# efficientnetb1           442   301
# efficientnetb2           442   301
#
# VGG16 · VGG19 는 전체 = 훈련 가능 (Conv2D · Dense 의 kernel · bias 만 있다 : VGG16 16층 x 2 = 32)
# 나머지는 전체 > 훈련 가능 : BatchNormalization 층의 이동 평균 · 이동 분산(moving_mean · moving_variance) 도 가중치로 세지만
#                           fit 에서 역전파로 고치는 값이 아니라서 trainable_weights 에는 들어가지 않는다
#  -> model.trainable = True 로 둔 Xception 도 236 / 156 으로 다르다
