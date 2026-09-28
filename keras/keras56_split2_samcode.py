import numpy as np

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0],
              ]).T

# print(a.shape) # (10, 2)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

size = 5

#################### [실습] ####################

bbb = split_x(a, size)
# print(bbb)
# print(bbb.shape)      # (6, 5, 2)

# x = bbb[:, :-1, :]
x = bbb[:, :-1]

# y = bbb[:, -1, 1]
y = bbb[:, -1, -1]

print(x)
print(y)                # [5 4 3 2 1 0]
print(x.shape, y.shape) # (6, 4, 2) (6,)
# x는 이미 (6, 4, 2) 3차원이라 LSTM에 넣을 때 reshape 할 필요가 없다
#   → 묶음 6개, 시점 4개, 시점마다 값 2개 (첫 번째 열 값, 두 번째 열 값)
#   예) x[0] = [[1 9] [2 8] [3 7] [4 6]] → y[0] = 5

# 스케일링 : x 값(0~10)을 10으로 나눠 0~1 근처로 맞춘다 (LSTM 안의 tanh 포화 방지)
x = x / 10

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#2. 모델 구성
model = Sequential()
# 직선 규칙 + 예측값 -1 이 훈련 y(0~5) 범위 밖 → linear 로 두어야 범위 밖까지 규칙을 이어간다
model.add(LSTM(10, input_shape=(4, 2), activation='linear'))
model.add(Dense(64))  # activation 기본값 linear
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss :', results)

# 마지막 4개 시점 [[7 3] [8 2] [9 1] [10 0]] 다음 칸의 두 번째 열 값 → -1
x_predict = np.array([[7, 3], [8, 2], [9, 1], [10, 0]]).reshape(1, 4, 2) # -1 맞춰보기
x_predict = x_predict / 10  # 훈련 데이터와 똑같이 스케일링
y_predict = model.predict(x_predict)

print('[-1] 예측 결과', y_predict)

# 목표 : -1
# 수정 전 : loss : 3.773192247535917e-07 / [-1] 예측 결과 [[-0.30748832]]
# 수정 후 : loss : 2.7887074338650564e-07 / [-1] 예측 결과 [[-0.99787235]]
