# m11_1_and.py 베이스

# [목적]
# 논리 게이트 2 / 3 : OR
#   입력 중 하나라도 1 이면 1 을 출력하는 OR 게이트 (y 만 [0, 1, 1, 1] 로 바뀜)
#   -> OR 도 직선 하나로 (0,0) 과 나머지 셋을 나눌 수 있어서 선형 모델로 1.0 이 나온다

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 1]

#2. 모델
# model = Perceptron()
model = LinearSVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
y_predict = model.predict(x_data)
result = model.score(x_data, y_data)
print("model.score: ", result)

acc = accuracy_score(y_data, y_predict)
print("accuracy_score :", acc)

# model.score:  1.0
# accuracy_score : 1.0
