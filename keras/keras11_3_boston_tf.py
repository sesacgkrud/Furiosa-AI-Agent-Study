# keras11_3_boston_tf.py
# 보스턴 집값 데이터(회귀) : 케라스가 주는 데이터는 처음부터 train / test 로 나뉘어서 들어온다
# 그래서 train_test_split 없이 바로 쓴다

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=16)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

# loss : 23.9038639068603