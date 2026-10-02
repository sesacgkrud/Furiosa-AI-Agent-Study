# keras1/keras43_hamsu01_mnist.py 베이스

import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM   # [LSTM] LSTM 추가
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# [LSTM] 베이스에서 주석 처리돼 있던 스케일링을 다시 켠다
#  LSTM 내부의 tanh / sigmoid 는 입력이 0 ~ 255 처럼 크면 값이 끝에 몰려 (포화) 학습이 거의 안 된다
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

# [LSTM] 2차원 reshape (-1, 784) -> 3차원 (N, timesteps, feature) = (N, 28, 28)
#  가로줄 28개를 시점 28개로 보고, 시점마다 그 줄의 픽셀 28개를 한 번에 읽는다
#  ((N, 784, 1) 로 픽셀을 하나씩 읽으면 시점이 784개라 훨씬 느리고 앞부분을 잊기 쉽다)
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)

#2. 모델 구성 (함수형)
# [LSTM] 입력층 shape (784,) -> (28, 28)
input1 = Input(shape=(28, 28))
# [LSTM] 입력층과 첫 Dense 사이에 LSTM 층을 추가
#  LSTM 은 마지막 시점의 출력만 내보내서 (None, 128) 2차원 -> 뒤의 Dense 층은 베이스 그대로 연결된다
lstm1 = LSTM(128)(input1)
dense1 = Dense(512, activation='relu')(lstm1)     # [LSTM] (input1) -> (lstm1) 뒤에 연결
dense2 = Dense(256, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
drop1 = Dropout(0.4)(dense4)

dense5 = Dense(64, activation='relu')(drop1)
dense6 = Dense(64, activation='relu')(dense5)
dense7 = Dense(32, activation='relu')(dense6)
dense8 = Dense(32, activation='relu')(dense7)
drop2 = Dropout(0.3)(dense8)

dense9 = Dense(16, activation='relu')(drop2)
dense10 = Dense(16, activation='relu')(dense9)
drop3 = Dropout(0.2)(dense10)

output1 = Dense(10, activation='softmax')(drop3)

model = Model(inputs=input1, outputs=output1)

model.summary()
# [LSTM] lstm1 param = 4 x (128 x (28 + 128) + 128) = 80,384
#  dense1 은 784 x 512 + 512 = 401,920 -> 128 x 512 + 512 = 66,048 로 줄었다

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

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_arg = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test_arg, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')
