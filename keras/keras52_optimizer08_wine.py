# keras52_optimizer08_wine.py
# 와인 (다중 분류) - optimizer 의 learning_rate 를 직접 지정해 본다
# 스케일러는 keras28 에서 비교해 고른 RobustScaler 를 그대로 쓴다
#
# [ 오늘 바뀌는 곳은 compile 한 줄이다 ]
#  before : model.compile(..., optimizer='adam')                  -> learning_rate 가 기본값 0.001 로 고정
#  after  : model.compile(..., optimizer=Adam(learning_rate=...))  -> 보폭을 내가 정한다
#
# [ learning_rate = 가중치를 한 번에 얼마나 크게 고칠지 정하는 보폭 ]
#  크면 (0.05 / 0.01)  : 최저점을 건너뛰고 그 주변에서 튕겨 다닌다
#  작으면 (0.00001)    : 한 걸음이 작아서 epoch 가 끝날 때까지 최저점에 닿지 못한다
#  데이터마다 맞는 값이 달라서 후보를 주석으로 적어 두고 하나씩 열어 가며 직접 비교했다
#  이 파일에서 고른 값 : learning_rate = 0.01
#
# keras28_Scaler08_wine.py 베이스

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets['target']

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=50,
    shuffle=True,
    stratify=y,
)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # -0.2222222222222222 Max : 1.2690763052208835

#2. 모델 구성
model = Sequential()
model.add(Dense(200, input_dim=13, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=2000,
          batch_size=32,
          verbose=1,
          validation_split=0.3,
          callbacks=[es],
          )
end_time = time.time()

#4. 평가 예측
result = model.evaluate(x_test, y_test, )
print('loss :', result[0])
print('acc :', round(result[1],3))

y_predict = model.predict(x_test)

y_test_arg = np.argmax(y_test, axis=1)
y_predict_arg = np.argmax(y_predict, axis=1)

accuracy_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', accuracy_score)
print('소요 시간 :', round(end_time - start_time), '초')

# [목표] acc = 0.95 이상 합격
# loss : 0.12328256666660309
# acc : 0.963
# accuracy_score : 0.9629629629629629
# 소요 시간 : 16 초

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# loss : 0.08898330479860306
# acc : 0.981
# accuracy_score : 0.9814814814814815
# 소요 시간 : 116 초

# [결론] 0.963 -> 0.981. 좋아진 것처럼 보이지만 그대로 믿으면 안 된다.
#        x_test 가 54개뿐이라 0.963 = 52/54, 0.981 = 53/54 -> 딱 1개 차이다.
#        seed 를 고정하지 않았으니 이 정도는 다시 돌리면 뒤집힐 수 있는 숫자다.
#        [참고] 소요 시간이 16초 -> 116초로 늘어난 것도 스케일링 자체가 느려서가 아니다.
#        스케일링 후 val_loss 가 더 오래 개선되니까 EarlyStopping(patience=50)이 늦게 걸린 것이다.
#        즉 '더 오래 학습할 수 있었다'는 뜻이라 오히려 좋은 신호에 가깝다.

# ========== ========== ========== ========== ========== <- StandardScaler 적용 후

# loss : 0.3034011721611023
# acc : 0.963
# accuracy_score : 0.9629629629629629
# 소요 시간 : 115 초

# ========== ========== ========== ========== ========== <- MaxAbsScaler 적용 후

# loss : 0.0894383043050766
# acc : 0.963
# accuracy_score : 0.9629629629629629
# 소요 시간 : 11 초

# ========== ========== ========== ========== ========== <- RobustScaler 적용 후

# loss : 0.1231844425201416
# acc : 0.963
# accuracy_score : 0.9629629629629629
# 소요 시간 : 115 초

# ========== ========== ========== ========== ========== <- learning rate 적용 후

# loss : 0.09649966657161713
# acc : 0.981
# accuracy_score : 0.9814814814814815
# 소요 시간 : 3 초

# [결론] RobustScaler 0.963 / 115초 -> learning_rate 0.01 로 0.981 / 3초.
#        acc 는 54개 중 1개 차이라 크지 않지만, 소요 시간이 115초 -> 3초로 줄었다.
#        보폭을 키우니 훨씬 적은 epoch 만에 최저점에 닿아 es 가 일찍 걸린 것이다.
