# keras52_ReduceLR10_digits.py
# 손글씨 숫자 (다중 분류) - ReduceLROnPlateau 로 훈련 도중 learning_rate 를 줄인다
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
#   factor=0.5 -> 0.005 -> 절반 -> 또 절반 ... 으로 떨어진다
#  EarlyStopping 과 역할이 다르다 : es 는 '멈춘다', rlr 은 '보폭을 줄여 더 해 본다'
#   그래서 rlr 의 patience 를 es 보다 짧게 줘야 "줄여 보고 -> 그래도 안 되면 멈춘다" 순서가 된다
#  이 파일 : 시작 learning_rate = 0.005 / es patience = 100 / rlr patience = 20
#
# keras28_Scaler10_digits.py 베이스

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

#1. 데이터
datasets = load_digits()

x = datasets.data
y = datasets['target']

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.6,
    random_state=99,
    shuffle=True,
    stratify=y,
)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : 0.0 Max : 2.6666666666666665

#2. 모델 구성
model = Sequential()
model.add(Dense(50, input_dim=64, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
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
          epochs=3000,
          batch_size=4,
          verbose=1,
          validation_split=0.3,
          callbacks=[es],
          )
end_time = time.time()

#4. 평가 예측
result = model.evaluate(x_test, y_test, )

print('loss :', result[0])
print('acc :', round(result[1], 3))

y_predict = model.predict(x_test)

y_test_arg = np.argmax(y_test, axis=1)
y_predict_arg = np.argmax(y_predict, axis=1)

accuracy_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', accuracy_score)
print('소요 시간 :', round(end_time - start_time), '초')

# [목표] acc = 1.00 합격
# loss : 0.12950600683689117
# acc : 0.965
# accuracy_score : 0.9652294853963839
# 소요 시간 : 34 초

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# loss : 0.12810786068439484
# acc : 0.974
# accuracy_score : 0.9735744089012517
# 소요 시간 : 34 초

# [결론] 0.9652 -> 0.9736. 예상대로 거의 변화가 없다.
#        digits 는 모든 컬럼이 '픽셀 밝기 0~16' 으로 이미 단위가 같아서
#        스케일링으로 바로잡을 컬럼 간 범위 차이가 애초에 없다.
#        x_test 540개 기준으로 521개 -> 526개, 5개 차이라 실행할 때마다 생기는 편차 범위다.
#        소요 시간도 34초로 동일하다.
#        효과가 없는 것을 확인한 대조군 역할의 데이터다.

# ========== ========== ========== ========== ========== <- StandardScaler 적용 후

# loss : 0.23934650421142578
# acc : 0.944
# accuracy_score : 0.9443671766342142
# 소요 시간 : 30 초

# ========== ========== ========== ========== ========== <- MaxAbsScaler 적용 후

# loss : 0.16283389925956726
# acc : 0.969
# accuracy_score : 0.9694019471488178
# 소요 시간 : 34 초

# ========== ========== ========== ========== ========== <- RobustScaler 적용 후

# loss : 0.20674990117549896
# acc : 0.937
# accuracy_score : 0.9374130737134909
# 소요 시간 : 29 초

# ========== ========== ========== ========== ========== <- learning rate 적용 후

# loss : 0.4236033856868744
# acc : 0.907
# accuracy_score : 0.9068150208623088
# 소요 시간 : 55 초

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후 (CPU)

# loss : 0.34086424112319946
# acc : 0.949
# accuracy_score : 0.9485396383866481
# 소요 시간 : 40 초

# [결론] 0.01 고정 0.9068 -> 시작 0.005 + ReduceLROnPlateau 0.9485 / 40초 (CPU).
#        튕겨 다니던 것이 lr 이 절반씩 줄면서 잡혔다.
#        es patience=100 / rlr patience=20 이라 줄여 볼 기회가 여러 번 온다.
