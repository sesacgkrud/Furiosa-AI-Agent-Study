# keras16_validation4_split.py
# validation_split : x_val 을 따로 만들지 않고 fit 이 x_train 의 일부를 검증용으로 떼어 쓰게 한다
# 주의 : train 을 나눈 뒤 다시 떼어 가는 것이라 실제 훈련에 쓰이는 데이터는 그만큼 줄어든다

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.75,
    random_state=42,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(15))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4,
        #   validation_data=(x_val, y_val),
          validation_split=0.33, # 33%를 검증에 사용
          )

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)