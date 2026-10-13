# keras71_trainable_weights1.py 베이스
# [변경] 동결한 모델을 실제로 compile · fit 해서, 동결 X / 동결 O 의 예측 결과를 비교한다
#  - 층을 Dense 30 → 20 → 20 → 20 → 10 → 10 → 1 로 늘렸다
#  - model.trainable = False 이면 fit 을 100 epoch 해도 가중치가 바뀌지 않는다

import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

print(tf.__version__)

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델 구성
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

######################## 동결 ########################
model.trainable = False
# model.trainable = True
######################## 동결 ########################

#3. 컴파일, 훈련
# [추가] 동결해도 compile · fit 은 에러 없이 돌아간다 (loss 만 계산되고 가중치는 갱신되지 않는다)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=100, verbose=1)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)

# 동결X
# [[1.0008212]
#  [2.0005221]
#  [3.0002232]
#  [3.9999237]
#  [4.9996247]]

# 동결O
# [[-0.3130919]
#  [-0.6261838]
#  [-0.9392756]
#  [-1.2523676]
#  [-1.565459 ]]

# [결론]
#  - 동결 X : 가중치가 훈련되어 y = x 에 맞게 1 ~ 5 가 나온다
#  - 동결 O : 처음 랜덤 가중치 그대로 예측한다 → 정답과 상관없는 값
#    bias 는 처음에 모두 0 이고 activation 도 없어서 예측값이 정확히 -0.3130919 x 꼴 (x=2 → -0.6261838 = 2배)
