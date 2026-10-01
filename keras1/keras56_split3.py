import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
a = np.array(range(1, 101))            # 1 ~ 100 (100,)
x_predict = np.array(range(96, 106))   # 96 ~ 105 → 이걸로 101 ~ 106 을 찾자

size = 6    # 한 묶음 = x 5개 + y 1개

# 목표
#   loss 0.1 이하
#   [101, 102, 103, 104, 105, 106] 의 근사치가 나오면 됨

# split_x : 데이터를 size 만큼씩 한 칸씩 밀면서 자른다
#   range(len(dataset) - size + 1) → 마지막 묶음이 데이터 끝을 넘지 않는 개수만큼 반복
#   a 의 길이 100, size 6 → 100 - 6 + 1 = 95 묶음
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb.shape)    # (95, 6)
# [[  1   2   3   4   5   6]
#  [  2   3   4   5   6   7]
#  ...
#  [ 95  96  97  98  99 100]]

# 한 묶음에서 앞 5개는 x, 마지막 1개는 y
x = bbb[:, :-1]     # (95, 5)  예) [1 2 3 4 5]
y = bbb[:, -1]      # (95,)    예) 6

# LSTM 은 3차원 (batch, timesteps, feature) 을 받는다
#   batch 95개, 시점 5개, 시점마다 값 1개 → (95, 5, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape, y.shape)     # (95, 5, 1) (95,)

# x_predict 도 훈련 x 와 똑같은 모양으로 만들어야 predict 할 수 있다
#   96 ~ 105 (10개) 를 5개씩 자르면 10 - 5 + 1 = 6 묶음 → 예측값도 6개(101 ~ 106)
#   size 가 아니라 size - 1 (=5) 로 자르는 이유 : 예측할 데이터에는 y 칸이 없기 때문
x_predict = split_x(x_predict, size - 1)
print(x_predict)
# [[ 96  97  98  99 100]  → 101
#  [ 97  98  99 100 101]  → 102
#  ...
#  [101 102 103 104 105]] → 106
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)   # (6, 5, 1)

# 스케일링 : x 값(1 ~ 105)을 100으로 나눠 0 ~ 1 근처로 맞춘다
#   큰 값이 그대로 들어가면 LSTM 안의 tanh / sigmoid 가 포화(1 근처에 붙음)되어 값 차이를 구분 못 한다
#   훈련 x 를 나눴으면 x_predict 도 반드시 똑같이 나눠야 한다
#   y 는 그대로 둔다 → 예측값이 바로 101 ~ 106 으로 나온다
x = x / 100
x_predict = x_predict / 100

#2. 모델 구성
model = Sequential()
# 1씩 늘어나는 직선(선형) 규칙이고, 예측할 101 ~ 106 은 훈련 y(6 ~ 100) 범위 밖에 있다
#   → tanh / relu 는 범위 밖으로 뻗어나가지 못하므로 activation='linear' 로 둔다
model.add(LSTM(64, input_shape=(5, 1), activation='linear'))   # (None, 5, 1) → (None, 64)
model.add(Dense(16))    # activation 을 안 쓰면 기본값이 linear
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# 데이터가 95개뿐이라 validation 을 떼지 않고 훈련 loss 로 멈춘다
#   (시계열에서 validation_split 은 뒤쪽 20% = 가장 큰 값들을 떼어가서 훈련 범위가 더 좁아진다)
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
print('[101 ~ 106] 예측 결과 :\n', y_predict)

# 목표 : [101, 102, 103, 104, 105, 106]
# 결과 : loss : 0.0030726746190339327
#        [[100.97149 ] [101.98282 ] [102.98472 ] [104.0083  ] [105.00776 ] [106.007454]]
