# keras08_mlp2_1.py
# 입력 특성 3개(컬럼 3개) 짜리 MLP 실습 -> input_dim=3
# 데이터를 행 단위로 쓰고 .T 로 (10, 3) 을 만드는 방식은 앞으로 계속 쓴다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9,8,7,6,5,4,3,2,1,0]
              ])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x = x.T

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=750, batch_size=5)

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict(np.array([[10, 1.3, 0]]))
print("[10, 1.3, 0]의 예측값 :", result)

# 목표 : 10.00...까지 합격 -> [10, 1.3, 0]의 예측값 : [[10.013668]]
# loss : 0.00015 이하 합격 -> loss : 5.540101483347826e-05