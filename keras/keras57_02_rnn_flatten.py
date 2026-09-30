import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60],
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50, 60, 70])  # 80 맞춰보기

x = x.reshape(x.shape[0], x.shape[1], 1)    # (13, 3, 1)
x_predict = x_predict.reshape(1, 3, 1)      # (1, 3, 1)

#2. 모델 구성
# return_sequences=True 로 나온 3차원 출력은 Dense 에 바로 넣지 않고 Flatten 으로 펴서 넘길 수 있다
#   LSTM(return_sequences=True) : (None, 3, 10) → 시점 3개 각각의 출력 10개
#   Flatten                    : (None, 30)    → 3 × 10 = 30 개를 한 줄로 편다 (CNN 에서 쓰던 것과 똑같음)
#
# keras57_01 (마지막 시점 출력만 사용) 과 비교
#   return_sequences=False : 마지막 시점 1개의 출력만 Dense 로 넘어간다
#   return_sequences=True + Flatten : 모든 시점의 출력을 다 Dense 로 넘긴다 → 중간 시점 정보도 쓸 수 있다
#
# 참고) Flatten 은 3차원 이상일 때만 의미가 있다
#   return_sequences=False 인 LSTM 뒤에 Flatten 을 두면 이미 2차원 (None, units) 이라 아무 변화가 없다
model = Sequential()
model.add(LSTM(10, input_shape=(3, 1), return_sequences=True))  # (None, 3, 1)  → (None, 3, 10)
model.add(Flatten())                                            # (None, 3, 10) → (None, 30)
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 3, 10)             480
#  flatten (Flatten)           (None, 30)                0      ← 모양만 바꾸므로 파라미터 0
#  dense (Dense)               (None, 16)                496    ← 30 × 16 + 16
#  dense_1 (Dense)             (None, 8)                 136
#  dense_2 (Dense)             (None, 1)                 9
# =================================================================
# Total params: 1,121

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

# 목표 : 80
# 결과 : loss : 0.2415105402469635 / [80] 예측 결과 : [[71.74753]]
#   keras57_01 과 같은 이유(80 은 훈련 y 범위 밖) 로 80 에 못 미친다
#   return_sequences + Flatten 은 모든 시점 정보를 Dense 로 넘겨주는 "구조" 의 차이일 뿐,
#   범위 밖 예측을 해결해 주는 방법은 아니다
