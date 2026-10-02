# keras1/keras36_cnn3_mnist2.py 베이스

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

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)

# [sparse] y 원핫 인코딩을 하지 않는다 -> y 는 정답 숫자 0 ~ 9 그대로 (60000,) (10000,)
print(y_train.shape, y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
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
