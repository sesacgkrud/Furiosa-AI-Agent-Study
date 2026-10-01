# keras16_validation1.py
# validation(검증) 데이터를 추가한다 : train 6개 / val 2개 / test 2개로 나눈다
# val_loss : 훈련 중에 매 epoch 마다 "안 배운 데이터" 로 재 보는 성적 -> 과적합을 미리 알아채는 기준이 된다
# test 는 마지막 평가에만 쓰고, 훈련 중에는 절대 보지 않는다

# keras09_train_test1.py 베이스

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6])
y_train = np.array([1,2,3,4,5,6])

x_val = np.array([7,8])
y_val = np.array([7,8])

x_test = np.array([9,10])
y_test = np.array([9,10])

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

###############################################################

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4,
        #   verbose=0, # 훈련되는 과정 생략, 자원 절약을 위한 방법
          verbose=1, # (deafult)
        #   verbose=2, # (progress bar 생략)
        #   나머지 -> epoch만 표시
          validation_data=(x_val, y_val),
          )

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test) # 검증을 테스트 데이터로 평가
print("loss :", loss)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)