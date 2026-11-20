# m11_2_or.py 베이스

# [목적]
# 논리 게이트 3 / 3 : XOR
#   두 입력이 서로 다를 때만 1 을 출력하는 XOR 게이트 (y = [0, 1, 1, 0])
#   -> (0,1) · (1,0) 은 1, (0,0) · (1,1) 은 0 이라 1 끼리 대각선 위치에 있다
#   -> 직선 하나로는 0 과 1 을 나눌 수 없다 (선형 분리 불가능)
#   -> Perceptron · LinearSVC 같은 선형 모델로는 풀 수 없다 = 단층 퍼셉트론의 한계 (XOR 문제)

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

#2. 모델
model = Perceptron()
# model = LinearSVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
y_predict = model.predict(x_data)
result = model.score(x_data, y_data)
print("model.score: ", result)

acc = accuracy_score(y_data, y_predict)
print("accuracy_score :", acc)

# model.score:  0.5
# accuracy_score : 0.5
# -> 4 개 중 2 개만 맞혔다 (Perceptron 은 random_state 를 주지 않으면 데이터를 섞는 순서가 달라 실행마다 바뀔 수 있다). 직선 하나로는 아무리 잘 그어도 4 개 중 3 개(0.75)까지만 맞힐 수 있다
