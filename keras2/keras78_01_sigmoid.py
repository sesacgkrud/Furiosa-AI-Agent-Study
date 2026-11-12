# [목적]
# 신경망 활성화 함수(activation function) 를 numpy 로 직접 만들어 그래프로 그려 본다. (keras78_01 ~ 08)
# 이번 파일은 그 첫 번째로 sigmoid 를 그린다.
#
# [sigmoid]
#   sigmoid(x) = 1 / (1 + e^-x)
#   - 어떤 값이 들어와도 출력이 0 ~ 1 사이로 눌린다 (x = 0 이면 0.5)
#   - x 가 크면 1 에, 작으면 0 에 가까워진다 → 이진 분류 출력층 (activation='sigmoid') 에서 확률처럼 쓴다
#   - 양 끝에서 기울기가 거의 0 이 되어, 층이 깊으면 기울기가 사라지는 문제(vanishing gradient) 가 생긴다
#
# [흐름]
#   1. sigmoid 함수 정의 : def 방식 → 같은 식을 lambda 한 줄로 다시 정의 (아래 lambda 가 덮어써서 lambda 가 쓰인다)
#   2. x = -5 ~ 4.9 를 0.1 간격으로 100 개 만든다
#   3. y = sigmoid(x) 를 계산해 plt.plot 으로 곡선을 그린다

import numpy as np
import matplotlib.pyplot as plt

def sigmoid (x):
    return 1 / (1 + np.exp(-x))    # 1 / (1 + e^-x)

sigmoid = lambda x : 1 / (1 + np.exp(-x))   # 위 def 와 같은 함수를 lambda 로
x = np.arange(-5, 5, 0.1)   # -5 부터 5 직전(4.9) 까지 0.1 간격
print(x)
print(len(x))   # 100

y = sigmoid(x)  # numpy 배열을 넣으면 원소마다 계산된 배열이 나온다
plt.plot(x, y)
plt.grid()      # 격자 표시
plt.show()
