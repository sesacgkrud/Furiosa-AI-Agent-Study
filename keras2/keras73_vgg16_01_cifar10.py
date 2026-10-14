# keras40_GAP03_cifar10.py 베이스 (이전 cifar10 최고 기록 0.7727) + keras72_1_vgg16_include_top.py 의 VGG16
# [목적] VGG16 전이학습으로 이전 cifar10 최고 기록을 갱신할 수 있는지, 동결 O / 동결 X 를 비교한다
#
# 전이학습 (Transfer Learning)
#  - 다른 큰 데이터로 이미 훈련이 끝난 모델의 가중치를 가져와서 내 데이터에 다시 쓰는 방법
#  - VGG16 은 imagenet (사진 약 120만 장, 1000종) 으로 미리 훈련되어 있다
#    → 선 · 모서리 · 질감 · 모양 같은 "사진에서 특징을 뽑는 법" 을 이미 알고 있다
#  - 그 특징 추출 부분(Conv2D 13개) 만 가져오고, 분류하는 부분(Dense) 은 cifar10 에 맞게 직접 붙인다
#
# 이전 cifar10 (keras40_GAP03) 과 달라진 점
#  1) Conv2D · MaxPooling2D 를 직접 쌓지 않고, VGG16 하나를 층처럼 model.add(vgg16) 으로 넣는다
#  2) 이전 모델은 가중치가 랜덤값에서 시작했지만, VGG16 부분은 imagenet 으로 훈련된 가중치에서 시작한다
#  3) vgg16.trainable 로 VGG16 부분의 가중치를 훈련시킬지(동결 X) 말지(동결 O) 고를 수 있다 (keras71 에서 배운 trainable)
#  4) 데이터 · 스케일링 · 원핫 · compile · EarlyStopping · fit 설정은 이전과 똑같이 두었다
#     → 모델만 바꿨을 때 결과가 어떻게 달라지는지 공정하게 비교하기 위해서

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten    # Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D 는 VGG16 안에 있으므로 import 안 함
import tensorflow as tf
import random
import time

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16    # keras.applications : 미리 훈련된 유명 모델들을 불러오는 곳 (VGG16, ResNet50 등)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)
# VGG16 도 (가로, 세로, 채널 3) 컬러 이미지를 받으므로 cifar10 은 reshape 없이 그대로 넣을 수 있다

########## 스케일링 : -1 ~ 1 ##########
# 이전 cifar10 과 같은 스케일링 (VGG16 이 imagenet 을 훈련할 때 쓴 전처리와는 다르다)
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

########## y 원핫 인코딩 ##########
# y_test 는 fit_transform 대신 transform : train 에서 정한 10개 클래스 순서를 그대로 쓴다
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)
# print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)

#2. 모델 구성
# VGG16(...) : 미리 훈련된 VGG16 모델을 불러온다
#  weights='imagenet' : imagenet 으로 훈련된 가중치를 내려받아 넣는다 (처음 실행할 때만 다운로드, 이후에는 ~/.keras/models 에 저장된 파일 사용)
#                       weights=None 이면 구조만 같고 가중치는 랜덤 → 이전처럼 처음부터 훈련하는 것과 같아진다
#  include_top=False  : 맨 위(top) 의 Flatten · Dense 4096 · Dense 4096 · Dense 1000 을 빼고 Conv2D 부분만 가져온다
#                       → imagenet 의 1000종 분류용 Dense 는 cifar10 (10종) 에 맞지 않으므로 직접 붙인다
#  input_shape=(32, 32, 3) : include_top=False 일 때만 원래 크기 (224, 224, 3) 대신 내 데이터 크기로 바꿀 수 있다
#                       → MaxPooling 5번으로 가로 세로가 32 → 16 → 8 → 4 → 2 → 1 로 줄어 출력이 (1, 1, 512)
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

######################## 동결 ########################
# vgg16.trainable : keras71 에서 model.trainable 로 한 것을 VGG16 부분에만 적용한다 (아래에 붙이는 Dense 는 항상 훈련된다)
#
#  동결 O (trainable=False)
#   - VGG16 의 가중치 14,714,688 개는 imagenet 값 그대로 고정, Dense 3개의 62,410 개만 훈련된다
#   - VGG16 은 "특징을 뽑는 도구" 로만 쓰이고, 그 특징으로 10종을 나누는 법만 새로 배운다
#   - 역전파로 고칠 가중치가 적어서 epoch 당 시간이 짧다
#   - 단점 : imagenet 은 224x224 큰 사진이라 32x32 작은 cifar10 에 딱 맞는 특징은 아닐 수 있다
#
#  동결 X (trainable=True)
#   - VGG16 의 가중치도 cifar10 에 맞게 계속 고친다 → 14,777,098 개 전체가 훈련된다
#   - 랜덤값이 아니라 imagenet 으로 이미 잘 훈련된 가중치에서 출발하므로, 처음부터 훈련하는 것보다 유리하다
#   - 단점 : 고칠 가중치가 약 237배라 epoch 당 시간이 훨씬 길다
#           학습률이 크면 미리 훈련된 가중치가 크게 바뀌어 오히려 망가질 수 있다 (keras73_vgg16_02_cifar100 동결 X 결과 참고)
vgg16.trainable=False      # 2. 가중치 동결
# vgg16.trainable=True     # 3. 가중치 동결 X
######################## 동결 ########################
# vgg16.summary()

# Sequential 에 VGG16 모델 전체를 층 하나처럼 넣는다 → summary 에 vgg16 (Functional) 한 줄로 나온다
model = Sequential()
model.add(vgg16)                                # 출력 (None, 1, 1, 512)
model.add(Flatten())                            # (1, 1, 512) → (512,)  include_top=False 로 뺀 Flatten 을 직접 붙인다
model.add(Dense(100))                           # 파라미터 (512 + 1) x 100 = 51,300
model.add(Dense(100))                           # 파라미터 (100 + 1) x 100 = 10,100
model.add(Dense(10, activation='softmax'))      # 파라미터 (100 + 1) x 10 = 1,010 / 10종 분류라 출력 10 + softmax
# Dense(100) 두 개에 activation 이 없다 → keras71_trainable_weights3 에서 본 것처럼 activation 없는 층은 겹쳐도 1차 함수 하나와 같다

model.summary()
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ vgg16 (Functional)                   │ (None, 1, 1, 512)           │      14,714,688 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 512)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 100)                 │          51,300 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 100)                 │          10,100 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 10)                  │           1,010 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 14,777,098 (56.37 MB)
#  Trainable params: 62,410 (243.79 KB)
#  Non-trainable params: 14,714,688 (56.13 MB)
# ↑ trainable=False 일 때의 summary
#   Trainable 62,410 = Dense 3개 (51,300 + 10,100 + 1,010) / Non-trainable 14,714,688 = VGG16 전체
#   trainable=True 로 바꾸면 Trainable params: 14,777,098 / Non-trainable params: 0 이 된다
#   (동결해도 Total params 는 그대로 → keras71 처럼 가중치는 있지만 훈련만 안 되는 것)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=25,
    restore_best_weights=True,
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=80, batch_size=128,
          verbose=1,
          validation_split=0.1,
          callbacks=[es]
          )

end_time = time.time()

#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# 실습 (이전과 비교해볼 것)
# 1. 이전에 본인이 한 최상의 결과
# 2. 가중치를 동결하고 훈련시켰을 때, trainable=False
# 3. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True
# 4. 소요 시간까지 비교

# ===== 1. 이전 최상의 결과 (keras40_GAP03_cifar10.py) ===== <- GlobalAveragePooling2D 적용
# loss : 0.8530533313751221
# acc : 0.7727000117301941
# accuracy_score : 0.7727
# 소요 시간 : 199.62 초

# ===== 2. VGG16 동결 O (trainable=False) ===== <- Trainable params 62,410
# Epoch 68: early stopping
# loss : 1.083117127418518
# acc : 0.6266000270843506
# accuracy_score : 0.6266
# 소요 시간 : 276.84 초

# ===== 3. VGG16 동결 X (trainable=True) ===== <- Trainable params 14,777,098
# Epoch 46: early stopping
# loss : 0.8923309445381165
# acc : 0.8158000111579895
# accuracy_score : 0.8158
# 소요 시간 : 1011.2 초

# [결론] 동결 X 0.8158 > 이전 최상(GAP) 0.7727 > 동결 O 0.6266
#  - 동결 X 가 이전 최고 기록을 약 0.04 넘었다
#    imagenet 가중치에서 출발해 cifar10 에 맞게 VGG16 까지 고쳤기 때문
#  - 동결 O 는 이전 기록보다 낮았다
#    imagenet 으로 배운 특징을 그대로 쓰고 Dense 62,410 개만 훈련해서, 32x32 작은 cifar10 에는 부족했다
#  - 시간 : epoch 당 평균 동결 O 약 4.1초 (276.84 / 68) vs 동결 X 약 22초 (1011.2 / 46)
#    → 훈련할 가중치가 많은 동결 X 가 epoch 당 약 5배 오래 걸렸다
