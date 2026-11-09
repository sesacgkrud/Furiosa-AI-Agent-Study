# keras25_summary.py
# [목적] model.summary() : 층 구조와 파라미터(가중치) 개수를 표로 보여 준다
# 한 층의 파라미터 수 = (입력 개수 x 출력 개수) + 출력 개수(bias)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#2. 모델 구성
model = Sequential()
# 1 x 3 + 3 = 6
model.add(Dense(3, input_dim = 1))
# 3 x 4 + 4 = 16
model.add(Dense(4))
# 4 x 3 + 3 = 15
model.add(Dense(3))
# 3 x 1 + 1 = 4  → 총 41 개
model.add(Dense(1))

model.summary() # 파라미터 갯수 count -> bias 까지 포함