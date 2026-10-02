# keras63_sparse1_mnist.py 베이스

import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

# 이번에는 reshape 주석 처리 -> (N, 28, 28) 3차원 그대로 넣고, 4차원은 모델 안의 Reshape 층으로 만든다
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

from tensorflow.keras.layers import Reshape

#2. 모델 구성
model = Sequential()
model.add(Dense(280, input_shape=(28, 28))) # (N, 28, 28) -> (N, 28, 280)
# Dense 는 마지막 축(28칸)에만 적용 -> 가로줄 28개마다 같은 가중치로 280칸을 만듦 -> param = 28 x 280 + 280 = 8120
model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 28, 280)           8120      
                                                                 
# =================================================================
# Total params: 8,120
# Trainable params: 8,120
# Non-trainable params: 0
# _________________________________________________________________

# Conv2D 는 4차원이 필요하므로 Reshape()
#  (28, 280) -> (28, 28, 10) : 28 x 280 = 28 x 28 x 10 = 7840 -> 값 개수가 같아야 바꿀 수 있음, 순서만 바꾸는 층이라 param 0
model.add(Reshape(target_shape=(28, 28, 10)))

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 10)))   # 채널 10 -> param = (3 x 3 x 10 + 1) x 32 = 2912
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=128, activation='relu'))

model.add(Dense(10, activation='softmax'))  # [sparse] 출력층은 그대로 10개 + softmax (예측은 여전히 10칸 확률)
model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 28, 280)           8120      
                                                                 
#  reshape (Reshape)           (None, 28, 28, 10)        0         
                                                                 
#  conv2d (Conv2D)             (None, 26, 26, 32)        2912      
                                                                 
#  conv2d_1 (Conv2D)           (None, 24, 24, 32)        9248      
                                                                 
#  conv2d_2 (Conv2D)           (None, 20, 20, 32)        25632     
                                                                 
#  dropout (Dropout)           (None, 20, 20, 32)        0         
                                                                 
#  conv2d_3 (Conv2D)           (None, 18, 18, 64)        18496     
                                                                 
#  conv2d_4 (Conv2D)           (None, 16, 16, 64)        36928     
                                                                 
#  conv2d_5 (Conv2D)           (None, 12, 12, 64)        102464    
                                                                 
#  dropout_1 (Dropout)         (None, 12, 12, 64)        0         
                                                                 
#  conv2d_6 (Conv2D)           (None, 10, 10, 128)       73856     
                                                                 
#  dropout_2 (Dropout)         (None, 10, 10, 128)       0         
                                                                 
#  flatten (Flatten)           (None, 12800)             0         
                                                                 
#  dense_1 (Dense)             (None, 256)               3277056   
                                                                 
#  dropout_3 (Dropout)         (None, 256)               0         
                                                                 
#  dense_2 (Dense)             (None, 128)               32896     
                                                                 
#  dense_3 (Dense)             (None, 10)                1290      
                                                                 
# =================================================================
# Total params: 3,588,898
# Trainable params: 3,588,898
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일, 훈련
# [sparse] categorical_crossentropy -> sparse_categorical_crossentropy
#  y 가 원핫이 아닌 정답 숫자여도 loss 안에서 원핫처럼 계산해 준다 (loss 값은 같다)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
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
y_predict = np.argmax(y_predict, axis=1)   # 예측은 (10000, 10) 확률 -> 그대로 argmax(axis=1)
# [sparse] y_test 는 이미 정답 숫자 (10000,) -> argmax 하지 않는다
#  - y_test 에 argmax(axis=1) : 1차원이라 AxisError: axis 1 is out of bounds for array of dimension 1
#  - 둘 다 axis=0 으로 바꾸면 : y_predict (10,) / y_test 숫자 1개 -> ValueError: inconsistent numbers of samples: [1, 10]

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')
