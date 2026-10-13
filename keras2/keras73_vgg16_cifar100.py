# keras73_vgg16_cifar10.py 베이스 (VGG16 · 전이학습 · 동결 설명은 cifar10 파일 참고)
# keras40_GAP04_cifar100.py (이전 cifar100 최고 기록 0.4621) 와 비교한다
#
# cifar10 과 달라진 점
#  1) cifar10 → cifar100 : 이미지는 똑같이 32x32 컬러 5만 장, 정답 종류만 10개 → 100개
#     → 한 종류당 훈련 이미지가 5000장 → 500장으로 줄어 훨씬 어렵다
#  2) 원핫 결과가 (50000, 100), 마지막 Dense 가 100 + softmax
#  3) 나머지 (스케일링 · VGG16 설정 · compile · EarlyStopping · fit) 는 cifar10 과 같다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
import random
import time

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16
from tensorflow.keras.datasets import cifar100     # cifar10 -> cifar100
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)
# print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100) <- 클래스 100개

#2. 모델 구성
# input_shape 가 cifar10 과 같은 (32, 32, 3) 이라 VGG16 부분은 그대로 (출력 (1, 1, 512), 파라미터 14,714,688)
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

vgg16.trainable=False      # 2. 가중치 동결
# vgg16.trainable=True     # 3. 가중치 동결 X

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))   # 출력 10 -> 100
# 마지막 Dense 파라미터 (100 + 1) x 100 = 10,100 (cifar10 은 1,010)
#  → Trainable params (동결 O) : 51,300 + 10,100 + 10,100 = 71,500 (cifar10 은 62,410)
#  → Trainable params (동결 X) : 14,714,688 + 71,500 = 14,786,188

model.summary()

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

# ===== 1. 이전 최상의 결과 (keras40_GAP04_cifar100.py) ===== <- GlobalAveragePooling2D 적용
# loss : 2.1279296875
# acc : 0.46209999918937683
# accuracy_score : 0.4621
# 소요 시간 : 227.22 초

# ===== 2. VGG16 동결 O (trainable=False) ===== <- Trainable params 71,500
# Epoch 43: early stopping (best epoch 18)
# loss : 2.4629876613616943
# acc : 0.38100001215934753
# accuracy_score : 0.381
# 소요 시간 : 749.54 초

# ===== 3. VGG16 동결 X (trainable=True) ===== <- Trainable params 14,786,188
# Epoch 28: early stopping (best epoch 3)
# loss : 4.6054582595825195
# acc : 0.009999999776482582
# accuracy_score : 0.01
# 소요 시간 : 369.26 초

# [결론] 이전 최상(GAP) 0.4621 > 동결 O 0.381 > 동결 X 0.01 -> 둘 다 이전보다 좋아지지 않았다.
#        동결 X 는 1 epoch 부터 loss 4.605(= ln100, 100개 중 찍기 수준)에 머물러 학습이 되지 않았다.
#
# cifar10 과 결과가 다른 이유
#  - 동결 O : cifar10 처럼 Dense 만 훈련하는데, 100종을 나누기에는 imagenet 특징 + Dense 71,500 개로 부족했다
#  - 동결 X : cifar10 에서는 0.8158 로 가장 좋았지만 cifar100 에서는 학습 자체가 안 됐다
#    acc 0.01 = 1/100 → 모든 이미지를 같은 클래스로 찍는 수준
#    loss 4.605 = ln(100) → 100개 클래스에 똑같은 확률(1/100) 을 주고 있다는 뜻
#    → adam 기본 학습률 0.001 로 VGG16 전체를 고치다가 imagenet 가중치가 크게 바뀌어 망가진 것으로 볼 수 있다
#      (동결 X 로 다시 해볼 때는 keras52 에서 배운 learning_rate 를 더 작게 줘서 비교해볼 수 있다)
#  - 시간 : 동결 O 가 epoch 당 약 17초 (749.54 / 43), 동결 X 가 약 13초 (369.26 / 28) 로
#    cifar10 (동결 O 약 4초, 동결 X 약 22초) 과 반대로 나왔다. 같은 조건에서 다시 재 보지 않아 원인은 확인하지 못했다
