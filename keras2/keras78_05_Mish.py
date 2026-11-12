# keras78_04_SiLu.py 베이스

# [Mish]
#   mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
#   - softplus(x) = ln(1 + e^x) : ReLU 를 부드럽게 만든 함수 (항상 0 보다 크다)
#   - SiLU 와 모양이 비슷하다 : 양수 쪽은 거의 x, 음수 쪽은 살짝 음수(최소 약 -0.31) 로 내려갔다가 0 에 가까워진다
#   - np.log 는 자연로그 ln

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def mish(x):
#     return x * np.tanh(np.log(1 + np.exp(x)))

mish = lambda x : x * np.tanh(np.log(1 + np.exp(x)))

y = mish(x)
plt.plot(x, y)
plt.grid()
plt.show()
