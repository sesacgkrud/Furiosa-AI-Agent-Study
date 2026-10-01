import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60],
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50, 60, 70])  # 80 맞춰보기

print(x.shape, y.shape)     # (13, 3) (13,)

# LSTM 입력은 3차원 (batch, timesteps, feature) → (13, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
x_predict = x_predict.reshape(1, 3, 1)      # predict 할 데이터도 똑같이 3차원 (묶음 1개)

#2. 모델 구성
# return_sequences : RNN 계열(SimpleRNN / LSTM / GRU) 이 "어떤 출력을 내보낼지" 정하는 옵션
#   False (기본값) : 마지막 시점의 출력 1개만 내보낸다 → 2차원 (batch, units)
#   True           : 모든 시점의 출력을 다 내보낸다   → 3차원 (batch, timesteps, units)
#
# RNN 층은 3차원만 입력으로 받는다
#   LSTM 뒤에 LSTM 을 또 쌓으려면 앞 LSTM 이 3차원을 내보내야 한다 → return_sequences=True
#   return_sequences 없이 쌓으면 에러가 난다
#     ValueError: Input 0 of layer "lstm_1" is incompatible with the layer:
#                 expected ndim=3, found ndim=2. Full shape received: (None, 10)
#   마지막 RNN 은 기본값(False) 으로 두어 2차원으로 만든 뒤 Dense 로 넘긴다
model = Sequential()
model.add(LSTM(10, input_shape=(3, 1), return_sequences=True))  # (None, 3, 1)  → (None, 3, 10)
model.add(LSTM(8))                                              # (None, 3, 10) → (None, 8)
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 3, 10)             480
#  lstm_1 (LSTM)               (None, 8)                 608
#  dense (Dense)               (None, 16)                144
#  dense_1 (Dense)             (None, 8)                 136
#  dense_2 (Dense)             (None, 1)                 9
# =================================================================
# Total params: 1,377
#
# LSTM 파라미터 = 4 × units × (feature + units + 1)
#   4 : LSTM 안의 게이트 4개 (forget / input / cell 후보 / output) 가 각각 가중치를 가진다
#   lstm   : 4 × 10 × (1  + 10 + 1) = 480
#   lstm_1 : 4 × 8  × (10 + 8  + 1) = 608   ← 앞 LSTM 의 units(10) 가 이 층의 feature 가 된다
# return_sequences=True 는 출력 "모양" 만 바꿀 뿐 파라미터 개수는 그대로다
#   (시점마다 같은 가중치를 반복해서 쓰기 때문)

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
results = model.evaluate(x, y)
print('loss :', results)

y_predict = model.predict(x_predict)
print('[80] 예측 결과 :', y_predict)

# LSTM 을 여러 층 쌓는다고 항상 좋아지는 건 아니다
#   층이 깊어질수록 파라미터가 늘고, 시계열 순서 정보가 흐려질 수 있다
#   실무에서는 보통 RNN 1 ~ 2층 + Dense 조합을 많이 쓴다

# 목표 : 80
# 결과 : loss : 0.004687579348683357 / [80] 예측 결과 : [[72.31362]]
#   loss 는 매우 작은데 80 에 못 미친다 → 80 은 훈련 y 의 최댓값(70) 보다 큰 "범위 밖" 값이기 때문
#   LSTM 안의 tanh 는 -1 ~ 1 사이로 값을 누르기 때문에 훈련 때 본 범위 밖으로 잘 뻗어나가지 못한다
#   (이 데이터는 앞 10개는 +1, 뒤 3개는 +10 규칙이라 split3 처럼 전부 linear 로 바꾸면 오히려 100 이상으로 튄다)
