# [목적]
# 퍼셉트론 · 선형 모델로 논리 게이트 문제 풀기 1 / 3 : AND
#   입력 두 개가 모두 1 일 때만 1 을 출력하는 AND 게이트를 학습시킨다
#   x1 x2 | y
#    0  0 | 0
#    0  1 | 0
#    1  0 | 0
#    1  1 | 1
#
# [흐름]
#   데이터 4 개 -> 모델 fit -> 같은 4 개로 predict -> model.score / accuracy_score 로 정확도 확인
#
# [모델]
#   Perceptron : 가장 단순한 신경망 (입력 * 가중치 + bias 가 0 보다 크면 1, 아니면 0)
#   LinearSVC  : 두 클래스를 직선(초평면) 하나로 나누는 선형 서포트 벡터 분류기
#   -> 둘 다 "직선 하나" 로 0 과 1 을 나눈다. AND 는 직선 하나로 나눌 수 있어서(선형 분리 가능) 1.0 이 나온다
#
# [model.score]
#   분류 모델의 score 는 accuracy (정답 개수 / 전체 개수) -> accuracy_score 와 같은 값

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 0, 0, 1]

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
