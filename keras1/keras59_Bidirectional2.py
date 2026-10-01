import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60],
             ])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape) # (13, 3) (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)
y = y.reshape(y.shape[0], 1)
print(x.shape) # (13, 3, 1)
print(y.shape)

#2. 모델 구성
model = Sequential()
# model.add(Bidirectional(SimpleRNN(128, return_sequences=True), input_shape=(3, 1)))
#
# Bidirectional(SimpleRNN(64)),
# model.add(Dense(32, activation='relu'))
# Dropout(0.3),
# model.add(Dense(1))
#   [주석 이유]
#   1. Bidirectional(SimpleRNN(64)), / Dropout(0.3), 는 model.add() 없이 층만 만들어서 모델에 들어가지 않았다
#      → 실제 모델은 Bidirectional(SimpleRNN, return_sequences=True) → Dense(32) → Dense(1) 뿐이었다
#   2. 그래서 return_sequences=True 의 3차원 출력 (None, 3, 256) 이 그대로 Dense 로 넘어가
#      최종 출력이 (None, 3, 1) → 시점 3개마다 값을 하나씩 내는 모양이 되었다 (예측도 [[[72.0] ...]] 처럼 3개)
#      → y (13, 1) 과 모양이 달라 "마지막에 80 하나" 를 맞추는 문제가 제대로 학습되지 않았다
#   3. 1번만 고쳐서 model.add 로 제대로 넣어봐도 SimpleRNN 2층은 약 70 에서 멈췄다
#      SimpleRNN 은 게이트가 없어서 20 → 30 → 40 처럼 큰 폭으로 뛰는 규칙을 잘 못 잡는다
#   4. Dropout 은 데이터가 13개뿐이라 오히려 배울 것을 버리게 된다 → 뺀다

# [수정] Bidirectional(LSTM(128)) 1층 + Dense 여러 층
#   - SimpleRNN → LSTM : 게이트가 값을 얼마나 기억 / 반영할지 조절해서 +1 규칙과 +10 규칙을 같이 잡는다
#   - RNN 은 1층만 두고 return_sequences 는 기본값(False) → (None, 256) 2차원이 바로 Dense 로 들어간다
#   - units 128 → 정방향 128 + 역방향 128 = 256 개의 특징을 Dense 로 넘긴다
#   - Dense 를 점점 줄여가며 쌓아서 256 개 특징을 값 1개로 모은다
model.add(Bidirectional(LSTM(128), input_shape=(3, 1)))  # (None, 3, 1) → (None, 256)
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
# Bidirectional 파라미터 = LSTM 파라미터 × 2 = 4 × 128 × (1 + 128 + 1) × 2 = 133120

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss :', results)

x_predict = np.array([50, 60, 70]).reshape(1, 3, 1) # 80 맞춰보기
y_predict = model.predict(x_predict)

print('[80] 예측 결과', y_predict)

# 목표 : 79.5
# loss : 0.115653857588768
# [80] 예측 결과 [[78.97944]]

#################### Bidirectional 적용 ####################
# 수정 전 : loss : 0.0023651323281228542 / [80] 예측 결과 [[[72.0553  ] ...]]  (출력이 (1, 3, 1) 모양)
# 수정 후 : loss : 0.0008170997025445104 / [80] 예측 결과 [[79.8367]]
#   3번 실행 : 79.84 / 79.40 / 78.98