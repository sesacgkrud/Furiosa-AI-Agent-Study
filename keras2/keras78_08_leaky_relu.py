# keras78_07_selu.py 베이스

# [Leaky ReLU]
#   leaky_relu(x) = x            (x > 0)
#                 = alpha * x    (x <= 0)
#   - ReLU 는 음수를 모두 0 으로 만들지만, Leaky ReLU 는 음수에 작은 기울기(alpha) 를 남긴다 → dying ReLU 완화
#   - np.where(조건, 참일 때 값, 거짓일 때 값) : 원소마다 조건을 보고 값을 고른다
#   - 주석의 np.maximum(alpha * x, x) 도 같은 결과 (alpha < 1 이면 양수는 x, 음수는 alpha * x 가 더 크다)

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# 음수 구간의 기울기
alpha = 0.01

# def leaky_relu(x):
#     # return np.maximum(alpha * x, x)
#     return np.where(x > 0, x, alpha * x)

leaky_relu = lambda x : np.where(x > 0, x, alpha * x)

y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()
