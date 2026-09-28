import numpy as np

a = np.array(range(1, 11))
size = 5

print(a.shape) # (10,)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape) # (6, 5)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
x = bbb[:, :-1] # 앞 4개
y = bbb[:, -1]  # 마지막 1개
print(x) # [[1 2 3 4] [2 3 4 5] ... [6 7 8 9]]
print(y) # [ 5  6  7  8  9 10]
print(x.shape, y.shape) # (6, 4) (6,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (6, 4, 1)

# [수정 후 추가] 스케일링 : x 값(1~10)을 10으로 나눠 0~1 근처로 맞춘다 (LSTM 안의 tanh 포화 방지)
x = x / 10

#2. 모델 구성
model = Sequential()
# [수정 전]
# model.add(LSTM(10, input_shape=(4, 1)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

# [수정 후] 직선 규칙 + 예측값 11 이 훈련 y(5~10) 범위 밖 → linear 로 두어야 범위 밖까지 규칙을 이어간다
model.add(LSTM(10, input_shape=(4, 1), activation='linear'))
model.add(Dense(64))  # activation 기본값 linear
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=1000)  # [수정 전]
model.fit(x, y, epochs=3000)    # [수정 후] linear 모델은 천천히 수렴해서 더 오래 훈련

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss :', results)

x_predict = np.array([7, 8, 9, 10]).reshape(1, 4, 1) # 11 맞춰보기
x_predict = x_predict / 10  # [수정 후 추가] 훈련 데이터와 똑같이 스케일링
y_predict = model.predict(x_predict)

print('[11] 예측 결과', y_predict)

################ 수정 전 / 수정 후 비교 ################
# 목표 : 11
#
# | 항목          | 수정 전                  | 수정 후                               |
# |---------------|--------------------------|---------------------------------------|
# | x 스케일링    | 없음 (1~10 그대로)       | x / 10, x_predict / 10                |
# | LSTM 활성화   | tanh (기본값)            | activation='linear'                   |
# | Dense 활성화  | relu                     | 없음 (기본값 linear)                  |
# | epochs        | 1000                     | 3000                                  |
# | loss          | 0.00013087032129988074   | 0.0007227236055769026                 |
# | [11] 예측     | [[10.57232]]  (오차 0.43) | [[11.099059]]  (오차 0.10)           |
#
# 바꾼 이유
#   - 11 은 훈련 때 본 y(5~10) 범위 밖이라, 규칙을 범위 밖까지 이어가야 맞출 수 있다
#   - tanh 는 입력이 크면 1 근처에 붙어버리고(포화) → 스케일링으로 값을 작게 만들어 해결
#   - tanh / relu 는 곡선이라 범위 밖에서 휘어진다 → 1씩 커지는 직선 규칙이라 linear 가 잘 맞음
#   - linear 모델은 천천히 수렴해서 epochs 를 늘릴수록 11 에 가까워졌다
#     (같은 설정 3번씩 : 1000 → 약 11.17 / 2000 → 약 11.13 / 3000 → 약 11.06)
#   - 수정 전 loss 가 더 작은데도 예측은 더 틀렸다
#     → loss 는 훈련 데이터(5~10)에 대한 점수라 범위 밖 예측 성능을 보장하지 않는다
