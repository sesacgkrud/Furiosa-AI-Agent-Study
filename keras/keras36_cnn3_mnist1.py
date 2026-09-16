# keras36_cnn3_mnist1.py
# mnist 손글씨 분류 - CNN 을 끝까지 돌려본 첫 모델 (스케일링 -> 4차원 reshape -> 원핫 -> Conv2D -> Flatten -> Dense)
# keras36_cnn2_mnist_imshow.py 베이스
#
# 이미지 데이터 스케일링 : 픽셀값 0 ~ 255 를 그대로 넣지 않고 작은 범위로 줄인다
#  방법 1) x / 255.            -> 0 ~ 1   (MinMax / MaxAbs 와 같은 결과)
#  방법 2) (x - 127.5) / 127.5 -> -1 ~ 1
# 이미지는 모든 컬럼(픽셀)의 범위가 0 ~ 255 로 같아서 scaler 없이 나눗셈만으로 스케일링이 된다

import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

########## 스케일링 1 ##########
# x_train = x_train/255. # . 의미 -> float 형태로 출력, 이미지 데이터라서 255로 나눈 것 -> 이 상태에서 Min-Max나 MaxAbs와 동일
# x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0

########## 스케일링 2 ##########
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5
print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

########## 4차원으로 reshape ##########
# Conv2D 의 입력은 (장수, 가로, 세로, 채널) 4차원이어야 한다
# mnist 는 흑백이라 로드하면 (60000, 28, 28) 3차원으로 들어오므로 채널 1 을 직접 붙여준다
# -1 = 장수는 알아서 계산 / 전체 값의 개수와 순서가 그대로라 reshape 가 가능하다
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)

########## y 원핫 인코딩 ##########
# 다중 분류라 y(0~9)를 10개 컬럼짜리 원핫으로 바꿔야 categorical_crossentropy 를 쓸 수 있다
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)   # sparse_output=False -> 희소행렬이 아니라 numpy 배열로 받는다
# y_train = ohe.fit_transform(y_train)     # y 가 (60000,) 1차원이라 그대로 넣으면 에러

########## 에러 발생 ##########
# ValueError: Expected 2D array, got 1D array instead:
# array=[5 0 4 ... 5 6 8].
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
# -> OneHotEncoder 는 2차원 입력만 받는다. (60000,) 를 (60000, 1) 로 바꿔주면 해결된다

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)


#2. 모델 구성
# Conv2D 는 padding 을 주지 않으면 (커널 크기 - 1) 만큼 가로 세로가 줄어든다
#   (3,3) 커널 -> 2씩 감소 / (2,2) 커널 -> 1씩 감소
# filters / kernel_size 는 앞의 두 자리 인자와 같은 것이라 이름을 써도 되고 순서로 넣어도 된다
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))               # 출력 : (26, 26, 64)  28 - 2
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu')) # 출력 : (24, 24, 32)  26 - 2
model.add(Dropout(0.2))                                             # Dropout 은 크기를 바꾸지 않고 훈련 때만 노드 일부를 끈다
model.add(Conv2D(32, kernel_size=(2,2), activation='relu'))         # 출력 : (23, 23, 32)  24 - 1
model.add(Conv2D(16, kernel_size=(2,2), activation='relu'))         # 출력 : (22, 22, 16)
model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=(2,2), activation='relu'))         # 출력 : (21, 21, 16)
model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=(2,2), activation='relu'))         # 출력 : (20, 20, 16)

# Flatten 이 없으면 Conv2D 의 4차원 출력이 그대로 Dense 로 들어가 마지막 출력이 (None, 20, 20, 10) 이 된다
# -> summary 는 보이지만 (60000, 10) 짜리 y 와 모양이 맞지 않아 fit 에서 에러가 난다
model.add(Flatten())                                                # (20, 20, 16) 4차원을 6400 짜리 2차원으로 펴준다. 값과 순서가 그대로라 reshape 와 같다

model.add(Dense(units=32, activation='relu'))                       # units = Dense 의 첫 번째 인자 이름 (Dense(32) 와 같다)
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))

model.add(Dense(10, activation='softmax'))                          # 0~9 열 개 클래스 -> 출력 10개 + softmax
model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 64)        640       
                                                                 
#  conv2d_1 (Conv2D)           (None, 24, 24, 32)        18464     
                                                                 
#  dropout (Dropout)           (None, 24, 24, 32)        0         
                                                                 
#  conv2d_2 (Conv2D)           (None, 23, 23, 32)        4128      
                                                                 
#  conv2d_3 (Conv2D)           (None, 22, 22, 16)        2064      
                                                                 
# =================================================================
# Total params: 25,296
# Trainable params: 25,296
# Non-trainable params: 0


# 위 결과는 Flatten / Dense 를 붙이기 전, Conv2D 4개까지만 쌓았을 때의 summary 다


# =================== Flatten, Dense 추가 후 결과 ===================
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 64)        640       
                                                                 
#  conv2d_1 (Conv2D)           (None, 24, 24, 32)        18464     
                                                                 
#  dropout (Dropout)           (None, 24, 24, 32)        0         
                                                                 
#  conv2d_2 (Conv2D)           (None, 23, 23, 32)        4128      
                                                                 
#  conv2d_3 (Conv2D)           (None, 22, 22, 16)        2064      
                                                                 
#  dropout_1 (Dropout)         (None, 22, 22, 16)        0         
                                                                 
#  conv2d_4 (Conv2D)           (None, 21, 21, 16)        1040      
                                                                 
#  dropout_2 (Dropout)         (None, 21, 21, 16)        0         
                                                                 
#  conv2d_5 (Conv2D)           (None, 20, 20, 16)        1040      
                                                                 
#  flatten (Flatten)           (None, 6400)              0         
                                                                 
#  dense (Dense)               (None, 32)                204832    
                                                                 
#  dropout_3 (Dropout)         (None, 32)                0         
                                                                 
#  dense_1 (Dense)             (None, 16)                528       
                                                                 
#  dense_2 (Dense)             (None, 10)                170       
                                                                 
# =================================================================
# Total params: 232,906
# Trainable params: 232,906
# Non-trainable params: 0


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

start_time = time.time()

model.fit(x_train, y_train, epochs=50, batch_size=128,
          verbose=1,
          validation_split=0.2,
          )                                # CPU / GPU 소요 시간을 비교하려고 EarlyStopping 없이 epochs 를 고정

end_time = time.time()


#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

# predict 결과는 (10000, 10) 짜리 확률이라 argmax 로 가장 큰 자리(=예측 숫자)를 뽑는다
# y_test 도 원핫이므로 똑같이 되돌려서 accuracy_score 에 넣는다
y_predict = model.predict(x_test)
y_predict=  np.argmax(y_predict, axis=1).reshape(-1,1) # reshape 없어도 됨
y_test = np.argmax(y_test, axis=1).reshape(-1,1)       # reshape 없어도 됨

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# ===== CPU 기록 =====   (같은 코드 / 같은 epochs=50 으로 장치만 바꿔 실행)
# loss : 0.04568600282073021
# acc : 0.989300012588501
# accuracy_score : 0.9893
# 소요 시간 : 557.11 초

# ===== GPU 기록 =====
# loss : 0.04483708366751671
# acc : 0.9897000193595886
# accuracy_score : 0.9897
# 소요 시간 : 141.31 초

# -> CPU 557초 / GPU 141초로 GPU 가 약 3.9배 빨랐다
#    Dense 만 쌓았던 keras35 에서는 CPU 가 더 빠른 경우가 많았는데, Conv2D 는 연산량이 커서 GPU 가 확실히 유리하다