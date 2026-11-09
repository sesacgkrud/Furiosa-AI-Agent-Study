# keras01.py
# [목적] 딥러닝 첫 모델 : y = x 인 데이터로 Sequential 모델을 만들고 훈련 -> 예측까지 4단계를 한 번에 본다
# 4단계 : #1 데이터 -> #2 모델 구성 -> #3 컴파일, 훈련 -> #4 평가, 예측 (이 순서는 앞으로 모든 파일에서 똑같다)

import tensorflow as tf # tensorflow를 가져오고 약어로 tf로 명명, AI에서 써야하는 기능을 대부분 가지고 있음
print(tf.__version__) # tensorflow 라이브러리 버전 출력

from tensorflow.keras.models import Sequential # Squential 순차적
from tensorflow.keras.layers import Dense # Dense 밀집된 형태의 layer 구성
import numpy as np

#1. 데이터
# x, y 가 같은 값 -> 모델이 y = 1 * x + 0 (가중치 1, 편향 0) 에 가까워지도록 학습
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성
model = Sequential() # 클래스는 ()까지 입력해야 하는 것 기억!
model.add(Dense(1, input_dim=1)) # Dense(출력 노드 수, input_dim=입력 컬럼 수) -> 입력 1개 · 출력 1개 (y = wx + b 하나)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # mse 방식으로 오차 도출, optimizer='adam' 85% 이상 평타 (default 개념)
model.fit(x, y, epochs=300) # model.fit -> 훈련, epochs -> 반복 훈련 횟수

#4. 평가, 예측
# 훈련에 없던 4 를 넣어 예측 -> 4 에 가까울수록 잘 학습된 것 (입력은 numpy 배열로)
result = model.predict(np.array([4]))
print("4의 예측값 :", result)

# 4의 예측값 : [[4.10628]]