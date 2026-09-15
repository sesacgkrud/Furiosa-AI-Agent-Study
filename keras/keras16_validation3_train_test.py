# keras16_validation3_train_test.py
# [실습] train_test_split 을 두 번 써서 train / val / test 로 나눈다
# 첫 번째로 train 과 나머지를 자르고, 남은 나머지를 다시 반으로 잘라 val 과 test 를 만든다

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

# [실습] train_test_split로 자르기
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.5,
    random_state=42,
)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test,
    test_size=0.5,
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
          validation_data=(x_val, y_val),
          )

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)