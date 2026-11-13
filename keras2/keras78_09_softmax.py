# [목적]
# 다중 분류 출력층에 쓰는 softmax 를 numpy 로 직접 만들고, 출력값을 원그래프(pie) 로 그려 비율을 확인한다. (keras78_01 ~ 08 활성화 함수에 이어서)
#
# [softmax]
#   softmax(x_i) = e^(x_i) / Σ e^(x_j)
#   - 각 값에 e^x 를 씌운 뒤 전체 합으로 나눈다 → 모든 출력이 0 ~ 1 이고 합이 1 (확률 분포)
#   - 값이 클수록 비율이 지수적으로 커진다 → 가장 큰 값의 확률이 두드러진다
#   - 다중 분류 출력층 activation='softmax' + categorical_crossentropy 와 짝, 예측은 np.argmax 로 가장 큰 자리를 고른다
#
# [흐름]
#   1. x = [1, 2, 3, 4] 를 softmax 에 넣는다
#      → y ≈ [0.0321, 0.0871, 0.2369, 0.6439] (합 1.0)
#   2. y 를 비율(ratio) 과 이름표(labels) 로 써서 plt.pie 로 그린다
#      - plt.pie(x, explode=None, labels=None, ...) : 두 번째 위치 인자는 explode(조각 띄우기) 라서
#        labels 는 labels=labels 처럼 이름을 붙여 넘겨야 이름표로 들어간다
#      - shadow=True : 그림자 / startangle=90 : 12 시 방향에서 시작해 반시계 방향으로 조각을 그린다

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 5)     # [1 2 3 4]

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x))

softmax = lambda x : np.exp(x) / np.sum(np.exp(x))
y = softmax(x)          # [0.0321 0.0871 0.2369 0.6439], 합 1.0

ratio = y
labels = y

plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.grid()
plt.show()
