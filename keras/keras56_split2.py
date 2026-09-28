import numpy as np

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0],
              ]).T

print(a.shape) # (10, 2)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

size = 5
bbb = split_x(a, size)
print(bbb.shape) # (6, 5, 2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
x = bbb[:, :-1]    # 두 열 모두, 앞 4개 시점 → (6, 4, 2)
y = bbb[:, -1, 1]  # 두 번째 열에서 마지막 1개
print(x) # [[[1 9] [2 8] [3 7] [4 6]] ... [[6 4] [7 3] [8 2] [9 1]]]
print(y) # [5 4 3 2 1 0]
print(x.shape, y.shape) # (6, 4, 2) (6,)

# LSTM(RNN 계열)은 입력을 3차원 (batch, timesteps, feature) 로 받는다
#   - batch     : 데이터 묶음 개수          → x.shape[0] = 6
#   - timesteps : 한 묶음 안의 시간 순서 길이 → x.shape[1] = 4 (4개 시점을 순서대로 하나씩 읽음)
#   - feature   : 한 시점에 들어가는 값 개수  → x.shape[2] = 2 (첫 번째 열 값, 두 번째 열 값)
# split_x 결과가 이미 3차원이라 reshape 할 필요가 없다
#   (x = bbb[:, :-1, 0] 처럼 한 열만 뽑으면 (6, 4) 2차원이 되어 reshape(6, 4, 1) 이 필요했다)

# 스케일링 : x 값(0~10)을 10으로 나눠 0~1 근처로 맞춘다
#   LSTM 안의 tanh 는 입력이 크면 출력이 1 근처에 붙어버려(포화) 값 차이를 구분 못 한다
#   값을 작게 만들어 주면 포화가 줄어서 규칙을 더 잘 배운다
x = x / 10

#2. 모델 구성
model = Sequential()
# 이 데이터는 1씩 늘고 1씩 줄어드는 직선(선형) 규칙이다
# 그리고 예측할 -1 은 훈련 때 본 y(0~5) 범위 밖에 있다
#   - tanh 는 -1~1 사이에 갇히고, relu 는 0 아래를 잘라버려서 범위 밖 값을 잘 못 뻗어나간다
#   - activation='linear' 로 두면 직선 규칙을 그대로 이어가서 범위 밖도 잘 맞춘다
model.add(LSTM(10, input_shape=(4, 2), activation='linear'))
model.add(Dense(64))  # activation 을 안 쓰면 기본값이 linear
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss :', results)

# 훈련 데이터의 규칙 : 4개 시점 다음 칸의 두 번째 열 값이 y
#   [[1 9] [2 8] [3 7] [4 6]] → 5, ... , [[6 4] [7 3] [8 2] [9 1]] → 0
# 그러면 마지막 4개 시점 [[7 3] [8 2] [9 1] [10 0]] 다음 칸의 두 번째 열 값은 -1
# 모델은 훈련 때와 같은 3차원 모양만 받으므로 predict 할 데이터도 똑같이 맞춰야 한다
#   np.array([[7, 3], [8, 2], [9, 1], [10, 0]]) : (4, 2)
#   .reshape(1, 4, 2)                           : (1, 4, 2) → 묶음 1개, 시점 4개, 시점마다 값 2개
#   훈련 때 x 를 10으로 나눴으니 predict 할 데이터도 똑같이 10으로 나눠야 한다
x_predict = np.array([[7, 3], [8, 2], [9, 1], [10, 0]]).reshape(1, 4, 2) # -1 맞춰보기
x_predict = x_predict / 10
y_predict = model.predict(x_predict)

print('[-1] 예측 결과', y_predict)

# 목표 : -1
# 수정 전 : loss : 9.813317802809252e-08 / [-1] 예측 결과 [[-0.10447288]]
# 수정 후 : loss : 2.225139951406163e-06 / [-1] 예측 결과 [[-1.000485]]
