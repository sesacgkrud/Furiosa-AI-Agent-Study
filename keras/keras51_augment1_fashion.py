# keras51_augment1_fashion.py
# fashion_mnist 데이터에 증폭(augmentation)을 적용해 훈련 데이터를 늘린다 - keras50_flow2_next.py 베이스
#
# [ 증폭의 흐름 ]
#  1) 원본에서 augment_size 만큼 인덱스를 랜덤으로 뽑는다
#  2) 뽑은 이미지를 datagen.flow 에 통과시켜 변형본을 만든다
#  3) 원본 + 변형본을 concatenate 로 합쳐 훈련 데이터를 늘린다
#  -> 사진을 새로 찍지 않고도 "같은 대상의 조금 다른 모습" 을 학습시킬 수 있다 (과적합 완화)
#
# [ 주의할 점 ]
#  - datagen 의 rescale 을 끈다 : 원본과 증폭본을 합친 뒤 한 번에 스케일링해야 범위가 어긋나지 않는다
#  - flow 의 결과는 (x, y) 튜플이라 [0] 으로 x 만 꺼낸다
#  - 증폭은 train 에만 한다. test 는 실제로 들어올 데이터 그대로여야 평가가 의미 있다
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

#################### 여기부터 증폭이다 ####################

# 켜 둔 옵션은 사진을 조금씩 바꾸는 규칙이다. 꺼 둔 옵션은 이 사진에는 어울리지 않아 남겨만 둔다
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

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([ 9991,  9992, 10003, 10039,  9949, 10058,  9991,  9981, 10067,
#         9929], dtype=int64))

########## 2차원으로 reshape ##########
# Dense 는 (샘플, 컬럼) 2차원 입력 -> 28x28 그림 한 장을 784개 컬럼으로 편다
x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)
print(x_train.shape, x_test.shape) # (100000, 784) (10000, 784)

########## y 원핫 인코딩 ##########
ohe = OneHotEncoder(sparse_output=False)   # 희소행렬이 아니라 numpy 배열로 받는다

y_train = y_train.reshape(-1,1)    # OneHotEncoder 는 2차원만 받으므로 (100000,) -> (100000, 1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)   # train 으로 기준(클래스 목록)을 만든다
y_test = ohe.transform(y_test)         # test 는 그 기준으로 변환만

print(y_train.shape, y_test.shape) # (100000, 10) (10000, 10)


########## 실습 - 기존 스코어보다 높일 것 !! ##########

#2. 모델 구성 (함수형) - keras41_dnn2 와 층 / Dropout 위치가 똑같다
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
# Total params: 599,178  <- keras41_dnn2(Sequential) 과 같아야 제대로 변환한 것
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

# predict 결과는 (10000, 10) 짜리 확률 -> argmax 로 가장 큰 자리(예측 숫자)를 뽑는다
# y_test 도 원핫이라 똑같이 되돌려서 accuracy_score 에 넣는다
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