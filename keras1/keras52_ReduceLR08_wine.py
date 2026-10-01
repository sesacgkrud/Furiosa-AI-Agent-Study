# keras52_ReduceLR08_wine.py
# 와인 (다중 분류) - ReduceLROnPlateau 로 훈련 도중 learning_rate 를 줄인다
# 스케일러(RobustScaler)와 learning_rate 후보는 keras52_optimizer 파일 그대로다
#
# [ 오늘 추가되는 곳은 콜백 하나다 ]
#  keras52_optimizer 는 learning_rate 를 처음부터 끝까지 고정했다
#  하지만 한 값으로 두 가지를 다 할 수는 없다
#   초반에는 크게 움직여 빨리 내려가야 하고, 최저점 근처에서는 작게 움직여야 지나치지 않는다
#
# [ ReduceLROnPlateau : 훈련 도중 learning_rate 를 줄여 준다 ]
#  plateau = 고원. val_loss 가 더 이상 줄지 않고 평평해지는 구간을 말한다
#  monitor 가 patience 동안 좋아지지 않으면 learning_rate 에 factor 를 곱한다
#   factor=0.5 -> 0.0005 -> 절반 -> 또 절반 ... 으로 떨어진다
#  EarlyStopping 과 역할이 다르다 : es 는 '멈춘다', rlr 은 '보폭을 줄여 더 해 본다'
#   그래서 rlr 의 patience 를 es 보다 짧게 줘야 "줄여 보고 -> 그래도 안 되면 멈춘다" 순서가 된다
#  이 파일 : 시작 learning_rate = 0.0005 / es patience = 50 / rlr patience = 20
#
# keras28_Scaler08_wine.py 베이스

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
# learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009
learning_rate = 0.0005

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True,
)

# val_loss 가 patience 동안 좋아지지 않으면 learning_rate 에 factor 를 곱해 줄인다
# verbose=1 -> 줄어드는 순간 'ReduceLROnPlateau reducing learning rate to ...' 가 찍힌다
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=2000,
          batch_size=32,
          verbose=1,
          validation_split=0.3,
          callbacks=[es, rlr],
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

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후

# loss : 0.08551058173179626
# acc : 0.981
# accuracy_score : 0.9814814814814815
# 소요 시간 : 12 초

# [결론] 0.01 고정 0.981 / 3초 -> 시작 0.0005 + ReduceLROnPlateau 0.981 / 12초.
#        acc 는 같고 시간만 늘었다. 작게 시작해 줄여 나가도 도착점은 같았다.
