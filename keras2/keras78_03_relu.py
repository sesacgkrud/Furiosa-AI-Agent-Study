# keras78_01_sigmoid.py 베이스

# [ReLU]
#   relu(x) = max(0, x)
#   - 0 보다 크면 그대로, 0 이하면 0 → 양수 쪽은 기울기가 1 이라 기울기 소실이 적다
#   - 계산이 단순해서 은닉층의 기본 활성화 함수로 쓴다 (activation='relu')
#   - 음수 입력은 모두 0 이 되어 기울기도 0 → 계속 음수만 받는 뉴런은 학습이 멈출 수 있다 (dying ReLU)
#   - np.maximum(0, x) : 0 과 x 를 원소마다 비교해 큰 값을 고른다

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def relu(x):
#     return np.maximum(0, x)

relu = lambda x : np.maximum(0, x)

y = relu(x)
plt.plot(x, y)
plt.grid()
plt.show()
