# keras78_06_elu.py 베이스

# [SELU]
#   selu(x) = lmbda * x                    (x > 0)
#           = lmbda * alpha * (e^x - 1)    (x <= 0)
#   - ELU 에 lmbda(약 1.0507) 를 곱한 함수, 음수 쪽 최소값은 -lmbda * alpha ≈ -1.76 에 가까워진다
#   - alpha · lmbda 는 정해진 고정 상수 → 층을 지나도 출력의 평균 0 · 분산 1 이 유지되도록(자기 정규화) 맞춘 값
#   - lambda 는 파이썬 예약어라 변수 이름을 lmbda 로 쓴다

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# SELU 고정 상수 (논문 제안값)
alpha = 1.6732632423543772
lmbda = 1.0507009873554805

# def selu(x, alpha, lmbda):
#     return lmbda * ((x > 0) * x + (x <= 0) * (alpha * (np.exp(x) -1)))

# 람다 변환
selu = lambda x, alpha, lmbda : lmbda * ((x > 0) * x + (x <= 0) * (alpha * (np.exp(x) -1)))

y = selu(x, alpha, lmbda)
plt.plot(x, y)
plt.grid()
plt.show()
