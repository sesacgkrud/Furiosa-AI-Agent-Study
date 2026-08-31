import tensorflow as tf # tensorflow를 가져오고 약어로 tf로 명명, AI에서 써야하는 기능을 대부분 가지고 있음
print(tf.__version__) # tensorflow 라이브러리 버전 출력

from tensorflow.keras.models import Sequential # Squential 순차적
from tensorflow.keras.layers import Dense # Dense 밀집된 형태의 layer 구성
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성
model = Sequential() # 클래스는 ()까지 입력해야 하는 것 기억!
model.add(Dense(1, input_dim=1)) # Dense(출력값, 입력값)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # mse 방식으로 오차 도출, optimizer='adam' 85% 이상 평타 (default 개념)
model.fit(x, y, epochs=1) # model.fit -> 훈련, epochs -> 반복 훈련 횟수

#4. 평가, 예측
result = model.predict(np.array([4]))
print("4의 예측값 :", result)