# keras02.py
# keras01 에 evaluate(평가) 를 추가한다
# evaluate : 훈련이 끝난 모델에 데이터를 넣어 loss 를 다시 계산한다 (여기서는 훈련 데이터를 그대로 넣어서 확인)

# from tensorflow.python.keras.models import Sequential # python을 입력하면 노란 줄이 없어지는데, python 버전에 따라 실행하면 에러가 발생하기도 함
# from tensorflow.python.keras.layers import Dense # python을 입력하면 노란 줄이 없어지는데, python 버전에 따라 실행하면 에러가 발생하기도 함

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300)

#4. 평가, 예측
loss = model.evaluate(x, y) # loss='mse'의 값
print("loss :", loss)
result = model.predict(np.array([1,2,3,4,5,6,7]))
print("7의 예측값 :", result)