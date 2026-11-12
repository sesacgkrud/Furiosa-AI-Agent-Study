# keras78_03_relu.py 베이스

# [SiLU (Swish)]
#   silu(x) = x * sigmoid(x) = x * (1 / (1 + e^-x))
#   - 양수 쪽은 ReLU 처럼 거의 x, 음수 쪽은 0 으로 딱 자르지 않고 살짝 음수(최소 약 -0.28) 로 내려갔다가 0 에 가까워진다
#   - 0 근처에서 꺾이지 않고 부드럽게 이어진다 → 음수 입력에도 기울기가 남는다

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def silu(x):
#     return x * (1 / (1 + np.exp(-x)))

silu = lambda x : x * (1 / (1 + np.exp(-x)))

y = silu(x)
plt.plot(x, y)
plt.grid()
plt.show()
