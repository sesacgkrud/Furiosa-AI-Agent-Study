# keras23_softmax2_wine.py
# 와인 3종 분류 (13개 특성)
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
from sklearn.datasets import load_wine

#1. 데이터
datasets = load_wine()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
# print(x.shape) # (178, 13)
# print(y.shape) # (178,)

########## OneHot Encording 1. (to_categorical) ##########
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y.shape) # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=50,
    shuffle=True,
    stratify=y,
)

# print(x_train.shape, x_test.shape) # (124, 13) (54, 13)
# print(y_train.shape, y_test.shape) # (124, 3) (54, 3)

#2. 모델 구성
model = Sequential()
model.add(Dense(200, input_dim=13, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=2000,
          batch_size=32,
          verbose=1,
          validation_split=0.3,
          callbacks=[es],
          )
end_time = time.time()

#4. 평가 예측
result = model.evaluate(x_test, y_test, )
print('loss :', result[0])
print('acc :', round(result[1],3))

y_predict = model.predict(x_test)

y_test_arg = np.argmax(y_test, axis=1)
y_predict_arg = np.argmax(y_predict, axis=1)

accuracy_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', accuracy_score)
print('소요 시간 :', round(end_time - start_time), '초')

# [목표] acc = 0.95 이상 합격
# loss : 0.12328256666660309
# acc : 0.963
# accuracy_score : 0.9629629629629629
# 소요 시간 : 16 초