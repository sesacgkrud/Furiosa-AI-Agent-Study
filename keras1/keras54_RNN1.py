# keras54_RNN1.py
# RNN(SimpleRNN) 입문 - "앞의 3개를 보고 다음 1개를 맞힌다"
#
# [ DNN / CNN 과 무엇이 다른가 ]
#  DNN  : 컬럼끼리 순서가 없다 (키, 몸무게, 나이 순서를 바꿔도 같은 데이터다)
#  CNN  : 가로 / 세로로 옆에 붙어 있는 픽셀끼리의 관계를 본다
#  RNN  : '앞에서 뒤로 흐르는 순서' 를 본다 -> 시계열 (주가, 날씨, 문장 등)
#         계산한 결과를 다음 칸으로 넘기면서 같은 층을 timesteps 번 반복한다
#
# [ RNN 은 3차원을 받는다 : (N, timesteps, features) ]
#  N         : 데이터 개수 (몇 묶음인가)
#  timesteps : 한 묶음 안에서 몇 칸을 보고 판단하는가   <- 여기서는 3 ([1,2,3] 처럼 3칸)
#  features  : 한 칸에 들어 있는 값이 몇 개인가          <- 여기서는 1 (숫자 하나)
#  그래서 (7, 3) 2차원을 (7, 3, 1) 로 reshape 해서 넣는다
#
# [ 데이터를 직접 잘라서 만든다 ]
#  원본은 1 ~ 10 이 이어진 숫자 하나뿐이다
#  이것을 x = 앞 3칸 / y = 그 다음 1칸 으로 밀어 가며 잘라 훈련 데이터를 만든다

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])    # 원본 시계열 (x, y 로 자르기 전)

# 한 칸씩 밀면서 3개 묶음을 만든다 -> 마지막 묶음이 [7,8,9] 라 7줄이 나온다
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])

y = np.array([4,5,6,7,8,9,10])           # 각 묶음의 '다음 값' 이 정답이다

print(x.shape, y.shape)                  # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1) # 2차원에서 3차원으로 변환
print(x.shape)                           # (7, 3, 1) -> (N, timesteps, features)

#2. 모델 구성
model = Sequential()
# units=10 -> RNN 이 기억해서 다음 칸으로 넘기는 값의 개수 (Dense 의 노드 수와 같은 자리)
# input_shape 에는 N 을 빼고 (timesteps, features) 만 적는다 - Dense 의 input_dim 과 같은 규칙
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))
model.add(SimpleRNN(10, input_shape=(3, 1)))

# 3차원으로 들어가서 1(벡터) 또는 2(Metrics)차원으로 나옴 -> 바로 Dense와 연결 가능
#  timesteps 를 다 돌고 '마지막 결과 하나' 만 내보내기 때문에 (N, units) 2차원이 된다
#  -> CNN 의 Flatten 같은 것이 따로 필요 없다
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))     # 숫자 하나를 맞히는 회귀라 출력층은 1칸, activation 없음

#3. 컴파일, 훈련
# 데이터가 7줄뿐이라 train_test_split / validation 없이 전부 훈련에 쓴다
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=520,)

#4. 평가, 예측
# 훈련에 쓴 데이터로 그대로 평가한다 (7줄짜리 연습용이라 나눌 것이 없다)
results = model.evaluate(x, y)
print('loss :', results)

# 예측도 훈련 때와 같은 3차원 모양으로 넣어야 한다
#  [8,9,10] 1묶음 / 3칸 / 값 1개 -> (1, 3, 1)
x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 결과 :', y_predict)

# loss : 5.951385173830204e-05
# 1/1 [==============================] - 0s 105ms/step
# [8,9,10]의 결과 : [[10.846822]]

# [결론] 정답 11 에 못 미치는 10.85 가 나왔다.
#        훈련 데이터의 y 가 4 ~ 10 까지뿐이라 모델은 10 보다 큰 값을 본 적이 없다.
#        RNN 이 규칙(+1)을 이해한 게 아니라 본 범위 안에서 흉내 낸 것에 가깝다.
