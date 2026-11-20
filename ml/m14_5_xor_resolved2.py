# m11_4_xor_unresolved.py 베이스

# [목적]
# XOR 풀기 3 : 신경망 대신 DecisionTreeClassifier -> 해결 (resolved)
#   결정 트리는 "x1 <= 0.5 ?" -> "x2 <= 0.5 ?" 처럼 축에 나란한 질문을 여러 번 나눠 내려간다
#   -> 직선 하나가 아니라 영역을 여러 칸으로 쪼개므로 XOR 같은 선형 분리 불가능 문제도 바로 풀린다
#
# [keras 모델과 다른 점]
#   - sklearn 모델이라 compile 이 없고 fit 에 epochs 도 없다 (fit 한 번으로 트리를 만든다)
#   - evaluate 대신 model.score (분류 모델이라 accuracy)
#   - predict 결과가 확률이 아니라 0 / 1 클래스라 np.round 를 해도 값이 그대로

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(333)
np.random.seed(333)

#1. 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0, 1, 1, 0])

#2. 모델
model = DecisionTreeClassifier()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
y_predict = model.predict(x_data)
result = model.score(x_data, y_data)
print("acc: ", result)

acc = accuracy_score(y_data, np.round(y_predict))
print("accuracy_score :", acc)

# acc:  1.0
# accuracy_score : 1.0
