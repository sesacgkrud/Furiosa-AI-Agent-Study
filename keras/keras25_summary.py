from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary() # 파라미터 갯수 count -> bias 까지 포함