# keras78_04_SiLu.py 베이스

# [ELU]
#   elu(x) = x                    (x > 0)
#          = alpha * (e^x - 1)    (x <= 0)
#   - 양수 쪽은 ReLU 와 같고, 음수 쪽은 0 대신 -alpha 를 향해 부드럽게 내려간다 (alpha = 1 이면 최소 -1 에 가까워짐)
#   - (x > 0) · (x < 0) 는 True / False 배열 → 곱하면 1 / 0 으로 계산되어, 조건에 맞는 쪽 식만 남는다
#   - x = 0 에서는 두 조건이 모두 False 라 0 (alpha * (e^0 - 1) = 0 과 같은 값)
#   - alpha 를 함수 인자로 받아서 lambda 에 인자 2 개 (x, alpha)

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
alpha = 1

# def elu(x, alpha):
#     return (x > 0) * x + (x < 0) * (alpha * (np.exp(x) - 1))

elu = lambda x, alpha : (x > 0) * x + (x < 0) * (alpha * (np.exp(x) - 1))

y = elu(x, alpha)
plt.plot(x, y)
plt.grid()
plt.show()
