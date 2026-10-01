# keras23_softmax4_digits.py
# 손글씨 숫자 0 ~ 9 분류 : 8 x 8 이미지를 펼친 64개 값이 특성이 된다
# 다중 분류 3종 세트
#  1) y 를 원핫 인코딩 : 0, 1, 2 를 [1,0,0] [0,1,0] [0,0,1] 로 바꾼다 (숫자 크기에 순서 의미가 생기지 않게)
#  2) 출력층 activation = softmax, 노드 수 = 클래스 개수 -> 칸별 확률이 나오고 다 더하면 1
#  3) loss = 'categorical_crossentropy'
# 정확도를 잴 때는 argmax 로 확률이 가장 큰 칸의 번호를 뽑아 정답 번호와 비교한다

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

#1. 데이터
datasets = load_digits()
# print(datasets.DESCR)
# print(datasets.feature_names) # (1797, 64)

x = datasets.data
y = datasets['target']
# print(x.shape) # (1797, 64)
# print(y.shape) # (1797,)
# print(y) # [0 1 2 ... 8 9 8]
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

########## OneHot Encording 1. (to_categorical) ##########
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y.shape) # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.6,
    random_state=99,
    shuffle=True,
    stratify=y,
)

print(x_train.shape, x_test.shape) # (1257, 64) (540, 64)
print(y_train.shape, y_test.shape) # (1257, 10) (540, 10)

#2. 모델 구성
model = Sequential()
model.add(Dense(50, input_dim=64, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=3000,
          batch_size=4,
          verbose=1,
          validation_split=0.3,
          callbacks=[es],
          )
end_time = time.time()

#4. 평가 예측
result = model.evaluate(x_test, y_test, )

print('loss :', result[0])
print('acc :', round(result[1], 3))

y_predict = model.predict(x_test)

y_test_arg = np.argmax(y_test, axis=1)
y_predict_arg = np.argmax(y_predict, axis=1)

accuracy_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', accuracy_score)
print('소요 시간 :', round(end_time - start_time), '초')

# [목표] acc = 1.00 합격
# loss : 0.12950600683689117
# acc : 0.965
# accuracy_score : 0.9652294853963839
# 소요 시간 : 34 초