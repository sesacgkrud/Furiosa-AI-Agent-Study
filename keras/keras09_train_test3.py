import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

## [찾아보기] train과 test 섞어서 7:3 나눈다.
## [힌트] Scikit-learn

x_train, x_test, y_train, y_test = train_test_split(
    x, y,

    # train_size + test_size <= 1.0 (O)
    # train_size + test_size > 1.0 (X) ValueError

    train_size=0.7, # train data 70%
    test_size=0.3, # 없어도 train_size 보고 default로 설정

    shuffle=True, # default = True -> 없어도 결과 동일 (섞음)

    # 훈련할 때마다 데이터셋이 달라지면 값이 매번 달라지기 때문에 필요
    random_state=1234, # 값 고정, 난수표에 있는 난수 대상으로 추출
)

print("x_train :", x_train) # x_train : [10  9  4  7  6  3  1]
print("x_test :", x_test) # x_test : [5 8 2]
print("y_train :", y_train) # y_train : [10  9  4  7  6  3  1]
print("y_test :", y_test) # y_test : [5 8 2]

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=4)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss :", loss)
result = model.predict(np.array([11]))
print("11의 예측값 :", result)

# loss : 3.623732069968355e-08
# loss : 3.396037939751295e-08
# 11의 예측값 : [[10.999699]]