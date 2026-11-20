# m11_3_xor.py 베이스

# [목적]
# XOR 을 keras 로 풀어 보기 1 : 은닉층 없는 Dense(1, sigmoid) -> 해결 안 됨 (unresolved)
#   Dense(1, input_dim=2, activation='sigmoid') 하나뿐인 모델 = 로지스틱 회귀 = 단층 퍼셉트론
#   -> 결국 직선 하나 (w1*x1 + w2*x2 + b = 0) 로 나누는 것이라 Perceptron · LinearSVC 와 마찬가지로 XOR 을 못 푼다
#
# [m11_3 과 다른 점]
#   - keras 모델은 list 가 아니라 numpy 배열로 넣는다 (np.array)
#   - 이진 분류라 loss='binary_crossentropy', 출력층 sigmoid
#   - predict 결과가 0 ~ 1 확률이라 np.round 로 0 / 1 로 바꿔서 accuracy_score 에 넣는다
#   - keras 의 evaluate 결과는 [loss, acc] 리스트 -> result[1] 이 acc

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(333)
np.random.seed(333)

#1. 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 1, 1, 0])

#2. 모델
# model = Perceptron()
# model = LinearSVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=100)

#4. 평가, 예측
y_predict = model.predict(x_data)
result = model.evaluate(x_data, y_data)
print("acc: ", result[1])

acc = accuracy_score(y_data, np.round(y_predict))
print("accuracy_score :", acc)

# acc:  0.75
# accuracy_score : 0.75
# -> 시드를 고정해도 실행마다 0.25 / 0.5 / 0.75 로 달라진다
# -> epochs 를 늘려도 직선 하나라서 최대 0.75 (4 개 중 3 개) 를 넘을 수 없다
