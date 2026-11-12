# keras78_01_sigmoid.py 베이스

# [tanh]
#   tanh(x) = (e^x - e^-x) / (e^x + e^-x)
#   - 출력이 -1 ~ 1 사이 (x = 0 이면 0) → sigmoid 와 모양은 같지만 0 을 중심으로 대칭
#   - numpy 에 np.tanh 가 있어서 함수를 따로 만들지 않고 바로 쓴다
#   - LSTM · GRU 안의 기본 활성화 함수 (activation='tanh')
#   - matplotlib.pylab 은 pyplot 과 같은 plot · grid · show 를 쓸 수 있는 모듈

import numpy as np
import matplotlib.pylab as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()
