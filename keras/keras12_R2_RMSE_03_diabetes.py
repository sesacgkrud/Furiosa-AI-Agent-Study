# keras12_R2_RMSE_03_diabetes.py
# 당뇨 데이터에 r2 를 추가한다
# loss(mse) 가 2900 처럼 커도 r2 로 보면 실제 성능이 얼마나 되는지 알 수 있다

from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=321,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(30, input_dim=10))
model.add(Dense(15))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2000, batch_size=32)

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

# R2 기준 : 0.62 이상

# loss : 2913.689208984375
# loss : 2900.58203125
# r2 : 0.4605041617011295