# [목적]
# 경사 하강법(gradient descent) 을 보기 전에, 최솟값을 찾을 2차 함수 f(x) = x² - 4x + 6 의 모양을 그래프로 확인한다.
#
# [f(x) = x² - 4x + 6]
#   - f(x) = (x - 2)² + 2 → 아래로 볼록, x = 2 에서 최솟값 f(2) = 2
#   - 신경망의 loss 를 가중치 x 에 대한 함수로 보면, 이 최솟값 지점을 찾아가는 것이 훈련이다
#
# [흐름]
#   1. np.linspace(-1, 6, 100) : -1 부터 6 까지 (6 포함) 같은 간격으로 100 개
#      (np.arange(시작, 끝, 간격) 은 간격을 정하고 끝을 포함하지 않지만, linspace 는 개수를 정하고 끝을 포함한다)
#   2. lambda 로 f(x) 를 만들어 곡선을 그린다 ("k-" : 검은 실선)
#   3. 최솟값 지점 (2, 2) 를 점으로 찍는다 ("sk" : 검은 네모 점 → color="red" 가 색을 빨강으로 덮어쓴다)

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 6, 100)
print(x, len(x))        # [-1. -0.92929293 ... 5.92929293 6.] 100

f = lambda x : x**2 - 4*x + 6
y = f(x)

plt.plot(x, y, "k-")
plt.plot(2, 2, "sk", color="red")   # 최솟값 지점 (2, 2)
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
