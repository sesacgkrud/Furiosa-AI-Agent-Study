# keras08_mlp1_2.py
# keras08_mlp1_1 과 같은 데이터를 (2, 5) 로 만든 뒤 전치(T)해서 (5, 2) 로 돌려 쓴다
# 전치 : 행과 열을 바꾼다 -> x.T 또는 x.transpose() (둘 다 결과는 같다)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]]) # (2, 5) -> 잘못된 예
# x = np.array([[1,6], [2,7], [3,8], [4,9], [5,10]]) # (5, 2) -> 올바른 예

## 데이터 전치
### 방법1
x = x.T # (5, 2)
### 방법2
# x = x.transpose()

y = np.array([1,2,3,4,5]) # (5,)

print(x.shape) # (5, 2) -> 행렬
print(y.shape) # (5,) -> 행렬

#2. 모델 구성
model = Sequential()
## Layer 구성
model.add(Dense(5, input_dim=2)) # 행렬 -> 2차원 -> input_dim=2 = 열의 갯수
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1)) # 벡터가 1개이기 때문에 1

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=3)

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict(np.array([[6, 11]])) # (1,2)
print("[6, 11]의 예측값 :", result)

# [6, 11]의 예측값 : [[5.908467]]