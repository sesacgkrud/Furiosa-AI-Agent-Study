# keras52_optimizer11_mnist.py
# 이미지 DNN (다중 분류) - optimizer 의 learning_rate 를 직접 지정해 본다
# 증폭(keras51) 까지 끝낸 함수형 DNN 에 learning_rate 만 바꿔 붙인다
#
# [ 오늘 바뀌는 곳은 compile 한 줄이다 ]
#  before : model.compile(..., optimizer='adam')                  -> learning_rate 가 기본값 0.001 로 고정
#  after  : model.compile(..., optimizer=Adam(learning_rate=...))  -> 보폭을 내가 정한다
#
# [ learning_rate = 가중치를 한 번에 얼마나 크게 고칠지 정하는 보폭 ]
#  크면 (0.05 / 0.01)  : 최저점을 건너뛰고 그 주변에서 튕겨 다닌다
#  작으면 (0.00001)    : 한 걸음이 작아서 epoch 가 끝날 때까지 최저점에 닿지 못한다
#  데이터마다 맞는 값이 달라서 후보를 주석으로 적어 두고 하나씩 열어 가며 직접 비교했다
#  이 파일에서 고른 값 : learning_rate = 0.01
#
# keras51_augment1_fashion.py 베이스

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

datagen = ImageDataGenerator(
    # rescale=1./255,             # 픽셀값 0~255 를 0~1 로 (imshow 도 0~1 범위를 그대로 그린다)
    horizontal_flip=True,       # 좌우반전 - 사람 / 사물은 뒤집어도 같은 대상이다
    # vertical_flip=True,       # 상하반전 - 얼굴 사진에는 어울리지 않아 끔
    width_shift_range=0.1,      # 좌우로 최대 10% 이동
    # height_shift_range=0.1,   # 위아래로 이동
    rotation_range=15,          # -15 ~ +15 도 범위에서 회전
    # zoom_range=1.2,           # 확대 / 축소
    # shear_range=0.7,          # 한 점을 고정하고 기울이기
    fill_mode='nearest',        # 이동 / 회전으로 생긴 빈 자리를 가장 가까운 픽셀값으로 채운다
)

augment_size = 40000

print(x_train.shape[0])         # 60000
randidx = np.random.randint(x_train.shape[0], size=augment_size) # 60,000만 개 데이터 중 4만 개 랜덤 뽑기
print(randidx)                  # [17907 45366 39807 ... 46953 30618  1412] -> 벡터 형태 확인 가능
print(len(randidx))             # 40000 -> 뽑은 개수 확인

print(np.min(randidx), np.max(randidx)) # 0 ~ 59,999까지 랜덤한 값 40,000개를 뽑아 최솟값과 최댓값 출력 (매번 다름)

x_augmented = x_train[randidx].copy()   # 새로운 메모리 공간 할당해서 저장
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,) -> 이대로 사용하면 X -> 데이터 변환해서 원 데이터에 붙여야 함

x_augmented = x_augmented.reshape(  # 40000, 28, 28, 1
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 1
)

print(x_augmented.shape)            # (40000, 28, 28, 1) -> flow 는 4차원을 받는다

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,              # 원본 순서를 유지해야 y_augmented 와 짝이 맞는다
).next()[0]                     # [0] = x, [1] = y

print(x_augmented.shape)            # (40000, 28, 28, 1)
print(x_train.shape)                # (60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented)) / 255.
y_train = np.concatenate((y_train, y_augmented))

x_test = x_test / 255.

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

print(np.unique(y_train, return_counts=True))

x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)
print(x_train.shape, x_test.shape) # (100000, 784) (10000, 784)

ohe = OneHotEncoder(sparse_output=False)   # 희소행렬이 아니라 numpy 배열로 받는다

y_train = y_train.reshape(-1,1)    # OneHotEncoder 는 2차원만 받으므로 (100000,) -> (100000, 1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)   # train 으로 기준(클래스 목록)을 만든다
y_test = ohe.transform(y_test)         # test 는 그 기준으로 변환만

print(y_train.shape, y_test.shape) # (100000, 10) (10000, 10)

#2. 모델 구성 (함수형) - keras41_dnn2 와 층 / Dropout 위치가 똑같다
input1 = Input(shape=(784,))                      # 입력층 : 컬럼 784개 (Sequential 의 input_shape=(784,))
dense1 = Dense(512, activation='relu')(input1)    # (input1) -> input1 뒤에 연결
dense2 = Dense(256, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
drop1 = Dropout(0.4)(dense4)                      # dense4 출력의 40% 를 훈련 때마다 끈다

dense5 = Dense(64, activation='relu')(drop1)      # Dropout 다음 층은 drop1 에 연결
dense6 = Dense(64, activation='relu')(dense5)
dense7 = Dense(32, activation='relu')(dense6)
dense8 = Dense(32, activation='relu')(dense7)
drop2 = Dropout(0.3)(dense8)

dense9 = Dense(16, activation='relu')(drop2)
dense10 = Dense(16, activation='relu')(dense9)
drop3 = Dropout(0.2)(dense10)

output1 = Dense(10, activation='softmax')(drop3)

model = Model(inputs=input1, outputs=output1)   # 시작(input1) ~ 끝(output1) 범위를 정해서 모델 완성

model.summary()

#3. 컴파일, 훈련
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',      # val_loss 기준이면 loss 가 가장 낮은 epoch 를 고르는데, 그 epoch 가 acc 최고점은 아니다
    mode='max',             # acc 는 높을수록 좋으므로 max
    patience=25,            # acc 는 들쭉날쭉해서 loss 기준보다 patience 를 넉넉히 준다
    restore_best_weights=True,  # 멈춘 뒤 val_acc 가 가장 높았던 가중치로 되돌린다
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
          )

end_time = time.time()


#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)
y_predict=  np.argmax(y_predict, axis=1).reshape(-1,1) # reshape 없어도 됨
y_test = np.argmax(y_test, axis=1).reshape(-1,1)       # reshape 없어도 됨

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# ===== CPU 기록 =====
# loss : 0.26822808384895325
# acc : 0.9261999726295471
# accuracy_score : 0.9262
# 소요 시간 : 951.0 초

# ===== CPU 기록 =====  <- MaxPooling 적용
# loss : 0.2809465229511261
# acc : 0.9279000163078308
# accuracy_score : 0.9279
# 소요 시간 : 763.36 초

# ===== GPU 기록 =====
# loss : 0.26894062757492065
# acc : 0.9258000254631042
# accuracy_score : 0.9258
# 소요 시간 : 380.58 초

# ===== GPU 기록 =====  <- MaxPooling 적용
# loss : 0.2832645773887634
# acc : 0.9229999780654907
# accuracy_score : 0.923
# 소요 시간 : 116.61 초

# ===== GPU 기록 ===== <- GlobalAveragePooling2D 적용
# loss : 0.2701389491558075
# acc : 0.9247999787330627
# accuracy_score : 0.9248
# 소요 시간 : 118.45 초

# ===== GPU 기록 ===== <- DNN
# Epoch 75: early stopping
# loss : 0.43086478114128113
# acc : 0.866100013256073
# accuracy_score : 0.8661
# 소요 시간 : 77.2 초

# ===== GPU 기록 ===== <- 함수형 적용
# loss : 0.48146337270736694
# acc : 0.8787999749183655
# accuracy_score : 0.8788
# 소요 시간 : 102.52 초

# ===== GPU 기록 ===== <- 증폭 적용
# Epoch 84: early stopping
# loss : 0.385041207075119
# acc : 0.8790000081062317
# accuracy_score : 0.879
# 소요 시간 : 154.43 초

# ===== GPU 기록 ===== <- learning rate 적용
# loss : 0.8069843053817749
# acc : 0.7107999920845032
# accuracy_score : 0.7108
# 소요 시간 : 64.08 초

# [결론] 증폭 기록 0.879 -> learning_rate 0.01 로 0.7108. 크게 나빠졌다.
#        Dense 512 부터 시작하는 깊은 DNN 이라 보폭 0.01 은 너무 크다.
