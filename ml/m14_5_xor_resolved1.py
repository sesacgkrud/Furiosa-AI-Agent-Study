# m11_4_xor_unresolved.py 베이스

# [목적]
# XOR 을 keras 로 풀어 보기 2 : 은닉층을 넣은 다층 퍼셉트론 (MLP) -> 해결 (resolved)
#   입력(2) -> 은닉층 Dense(10, relu) -> 출력층 Dense(1, sigmoid)
#
# [왜 풀리나]
#   - 은닉층이 입력을 새로운 10 개의 특성으로 바꾸고, 출력층이 그 특성 공간에서 직선으로 나눈다
#     -> 원래 입력 공간에서 보면 직선이 아니라 꺾인 경계가 되어 XOR 을 나눌 수 있다
#   - 핵심은 은닉층의 활성화 함수(relu). activation 을 주지 않으면 선형(linear) 이라
#     Dense 를 여러 개 쌓아도 (x·W1 + b1)·W2 + b2 = x·W + b 로 결국 층 하나(직선 하나)와 같다
#     -> 은닉층 Dense(10) 에 activation 없이 돌리면 m11_4 와 똑같이 acc 0.5
#
# [epochs]
#   데이터가 4 개뿐이라 한 epoch 에 가중치가 한 번만 갱신된다 (batch_size 기본 32 > 4)
#   -> adam 기본 learning rate 0.001 로는 epochs=100 에서 아직 acc 0.5 -> 500 으로 늘려 1.0 확인

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__) # 2.21.0

#1. 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 1, 1, 0])

#2. 모델
# model = Perceptron()
# model = LinearSVC()
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))   # 은닉층 : 비선형 활성화 함수가 있어야 XOR 을 푼다
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=500)

#4. 평가, 예측
y_predict = model.predict(x_data)
result = model.evaluate(x_data, y_data)
print("acc: ", result[1])

acc = accuracy_score(y_data, np.round(y_predict))
print("accuracy_score :", acc)

# acc:  1.0
# accuracy_score : 1.0
