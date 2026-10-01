# keras51_augment2_mnist.py
# mnist 데이터에 증폭(augmentation)을 적용해 훈련 데이터를 늘린다 - keras51_augment1_fashion.py 와 같은 흐름
#
# [ 증폭의 흐름 ]
#  1) 원본 6만 장에서 4만 장을 랜덤으로 뽑는다
#  2) 뽑은 이미지를 datagen.flow 에 통과시켜 변형본을 만든다
#  3) 원본 + 변형본을 concatenate 로 합쳐 10만 장으로 늘린다
#
# [ 숫자 이미지에 증폭을 쓸 때 ]
#  숫자는 좌우로 뒤집으면 모양이 달라진다 (2, 3, 7 등은 뒤집으면 실제로 쓰지 않는 모양이 된다)
#  fashion 의 옷처럼 뒤집어도 같은 대상인 데이터와 달리 horizontal_flip 이 도움이 되지 않을 수 있다
#  -> 결과도 함수형 기록(0.9784)과 증폭 적용(0.9783)이 거의 같았다
#
# [ 주의할 점 ]
#  - datagen 의 rescale 을 끈다 : 원본과 증폭본을 합친 뒤 한 번에 스케일링해야 범위가 어긋나지 않는다
#  - 증폭은 train 에만 한다. test 는 실제로 들어올 데이터 그대로여야 평가가 의미 있다

import numpy as np
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model                   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input   # Input -> 함수형 모델의 입력층
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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

########## 증폭할 이미지 뽑기 ##########
# keras50_flow2 에서는 한 장을 100번 복사해 변형했지만,
# 여기서는 원본 6만 장 중 4만 장을 랜덤으로 뽑아 변형한다 (서로 다른 사진이라 더 다양해진다)
augment_size = 40000

# randidx = np.random.randint(60000, size=augment_size) # 60,000만 개 데이터 중 4만 개 랜덤 뽑기

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

########## 뽑은 이미지를 변형 ##########
# batch_size 를 전체 장수로 주면 한 번에 다 변형된다 -> next() 로 그 한 묶음을 꺼낸다
# flow 결과는 (x, y) 튜플이라 [0] 으로 x 만 받는다 (y 는 변형되지 않으므로 y_augmented 를 그대로 쓴다)
x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,              # 원본 순서를 유지해야 y_augmented 와 짝이 맞는다
).next()[0]                     # [0] = x, [1] = y

########## 변환 완료 ##########
print(x_augmented.shape)            # (40000, 28, 28, 1)
print(x_train.shape)                # (60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

########## 원본 + 증폭 데이터 합치기 ##########
# 원본과 증폭본 모두 0~255 상태다 (datagen 의 rescale 을 꺼 뒀기 때문)
# 합친 뒤 한 번에 나눠야 두 덩어리의 범위가 같아진다
x_train = np.concatenate((x_train, x_augmented)) / 255.
y_train = np.concatenate((y_train, y_augmented))

# x_test 도 반드시 같은 방법으로 나눈다
# 훈련은 0~1 인데 평가만 0~255 로 들어가면 모델이 못 보던 범위라 acc 가 크게 떨어진다
x_test = x_test / 255.

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)


########## 2차원으로 reshape ##########
# Dense 는 (샘플, 컬럼) 2차원 입력 -> 28x28 그림 한 장을 784개 컬럼으로 편다
x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)
print(x_train.shape, x_test.shape) # (100000, 784) (10000, 784)

########## y 원핫 인코딩 ##########
ohe = OneHotEncoder(sparse_output=False)   # 희소행렬이 아니라 numpy 배열로 받는다

y_train = y_train.reshape(-1, 1)    # OneHotEncoder 는 2차원만 받으므로 (100000,) -> (100000, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)   # train 으로 기준(클래스 목록)을 만든다
y_test = ohe.transform(y_test)         # test 는 그 기준으로 변환만

print(y_train.shape, y_test.shape) # (100000, 10) (10000, 10)

#2. 모델 구성 (함수형) - keras41_dnn1 과 층 / Dropout 위치가 똑같다
# 변수 이름은 층마다 새로 붙인다 (dense1 / dense2 ...)
# 앞 층과 같은 이름을 다시 쓰면 그 변수가 덮어써져서 해당 층이 모델 연결에서 빠진다
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

# 다중 분류(원핫) 출력층은 softmax 가 반드시 있어야 칸별 확률(합 = 1)이 나온다
output1 = Dense(10, activation='softmax')(drop3)

model = Model(inputs=input1, outputs=output1)   # 시작(input1) ~ 끝(output1) 범위를 정해서 모델 완성

model.summary()
# Total params: 599,178  <- keras41_dnn1(Sequential) 과 같아야 제대로 변환한 것
#  -> 784 x 512 + 512 = 401,920 으로 첫 층 한 곳이 전체의 67% 를 차지한다

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

# 목표가 정확도라서 val_loss 가 아니라 val_acc 를 기준으로 잡는다
#  - loss 는 낮을수록 좋고 acc 는 높을수록 좋으므로 mode 를 'max' 로 같이 바꿔야 한다
es = EarlyStopping(
    monitor='val_acc',      # val_loss 기준이면 loss 가 가장 낮은 epoch 를 고르는데, 그 epoch 가 acc 최고점은 아니다
    mode='max',             # acc 는 높을수록 좋으므로 max
    patience=25,            # acc 는 들쭉날쭉해서 loss 기준보다 patience 를 넉넉히 준다
    restore_best_weights=True,  # 멈춘 뒤 val_acc 가 가장 높았던 가중치로 되돌린다
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=32,
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

# predict 결과는 (10000, 10) 짜리 확률 -> argmax 로 가장 큰 자리(예측 숫자)를 뽑는다
# y_test 도 원핫이라 똑같이 되돌려서 accuracy_score 에 넣는다
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_arg = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test_arg, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# ===== CPU 기록 =====
# Epoch 54: early stopping
# acc: 0.9938 - loss: 0.0334 
# loss : 0.03343440219759941
# acc : 0.9937999844551086
# accuracy_score : 0.9938
# 소요 시간 : 1861.89 초

# ===== GPU 기록 =====
# Epoch 70: early stopping
# loss: 0.0292 - acc: 0.9938
# loss : 0.029211144894361496
# acc : 0.9937999844551086
# accuracy_score : 0.9938
# 소요 시간 : 322.36 초

# ===== GPU 기록 ===== <- MaxPooling 적용
# loss : 0.025499815121293068
# acc : 0.9952999949455261
# accuracy_score : 0.9953
# 소요 시간 : 418.48 초

# ===== GPU 기록 ===== <- GlobalAveragePooling2D 적용
# loss : 0.017076997086405754
# acc : 0.9951000213623047
# accuracy_score : 0.9951
# 소요 시간 : 285.72 초

# ===== GPU 기록 ===== <- DNN
# Epoch 59: early stopping
# loss : 0.1566680520772934
# acc : 0.9785000085830688
# accuracy_score : 0.9785
# 소요 시간 : 223.62 초

# ===== GPU 기록 ===== <- 함수형 적용
# Epoch 78: early stopping
# loss : 0.2517959177494049
# acc : 0.9783999919891357
# accuracy_score : 0.9784
# 소요 시간 : 323.16 초

# ===== GPU 기록 ===== <- 증폭 적용
# Epoch 81: early stopping
# loss : 45.82192611694336
# acc : 0.9782999753952026
# accuracy_score : 0.9783
# 소요 시간 : 595.48 초