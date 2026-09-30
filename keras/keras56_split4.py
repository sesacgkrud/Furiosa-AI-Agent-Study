import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
a = np.array(range(1, 101))            # 1 ~ 100 (100,)
x_predict = np.array(range(96, 106))   # 96 ~ 105

size = 6

# [실습] 데이터를 reshape 한 후, split_x 함수로 시계열 데이터로 변환
#   (N, 10, 1) → (N, 5, 2)
#   한 시점에 값 1개씩 10개 보던 것을, 한 시점에 값 2개씩 5개 보도록 바꾼다

# 1차원 (100,) 을 2개씩 묶어 2차원 (50, 2) 로 만든다 → 한 행 = 한 시점, 두 열 = feature 2개
#   -1 은 "나머지는 알아서 계산" → 100 / 2 = 50
a = a.reshape(-1, 2)
print(a.shape)      # (50, 2)
# [[  1   2]
#  [  3   4]
#  ...
#  [ 99 100]]

# split_x 는 1차원(벡터)뿐 아니라 2차원(행렬)도 그대로 자를 수 있다
#   dataset[i : i+size] 는 "행" 을 기준으로 자르기 때문에 열(feature) 개수는 그대로 유지된다
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb.shape)    # (45, 6, 2) → 50 - 6 + 1 = 45 묶음, 묶음마다 6개 시점, 시점마다 값 2개
# bbb[0] = [[ 1  2] [ 3  4] [ 5  6] [ 7  8] [ 9 10] [11 12]]

# 앞 5개 시점은 x, 마지막 1개 시점은 y
x = bbb[:, :-1]     # (45, 5, 2)  예) [[1 2] [3 4] [5 6] [7 8] [9 10]]
y = bbb[:, -1]      # (45, 2)     예) [11 12] → 다음 시점의 값 2개를 한 번에 맞춘다
# split_x 결과가 이미 3차원이라 LSTM 에 넣기 위한 reshape 가 필요 없다
#   (한 값만 맞추고 싶으면 y = bbb[:, -1, -1] → (45,) 로 두고 마지막 Dense 를 1 로 두면 된다)
print(x.shape, y.shape)     # (45, 5, 2) (45, 2)

# x_predict 도 훈련과 똑같이 (2개씩 묶기 → 5개 시점씩 자르기) 해서 (N, 5, 2) 로 만든다
x_predict = x_predict.reshape(-1, 2)            # (5, 2)
x_predict = split_x(x_predict, size - 1)        # (1, 5, 2) → 5 - 5 + 1 = 1 묶음
print(x_predict)
# [[[ 96  97] [ 98  99] [100 101] [102 103] [104 105]]] → 다음 시점 [106 107] 이 정답

# 스케일링 : split3 과 같은 이유로 100으로 나누고, x_predict 도 똑같이 나눈다
x = x / 100
x_predict = x_predict / 100

#2. 모델 구성
model = Sequential()
# input_shape = (timesteps, feature) = (5, 2)
model.add(LSTM(64, input_shape=(5, 2), activation='linear'))   # (None, 5, 2) → (None, 64)
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(2))     # y 가 (45, 2) → 출력 노드도 2개 (다음 시점의 값 2개)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='loss',
    mode='min',
    patience=50,
    restore_best_weights=True,
)
model.fit(x, y, epochs=1000, batch_size=4, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)

y_predict = model.predict(x_predict)
print('[106 107] 예측 결과 :', y_predict)

# 목표 : [106 107]
# 결과 : loss : 0.010378451086580753 / [[105.77026 106.86232]]
