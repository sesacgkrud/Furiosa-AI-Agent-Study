from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=100,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(50, input_dim=8))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=16)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss :", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

# R2 기준 : 0.55 이상

# loss : 0.5828888416290283
# loss : 0.6091861724853516
# r2 : 0.5579915532789418