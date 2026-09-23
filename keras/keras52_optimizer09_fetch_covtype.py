# keras52_optimizer09_fetch_covtype.py
# 산림 피복 (다중 분류) - optimizer 의 learning_rate 를 직접 지정해 본다
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
#  이 파일에서 고른 값 : learning_rate = 0.0005
#
# keras28_Scaler09_fetch_covtype.py 베이스

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype

#1. 데이터
datasets = fetch_covtype()

x = datasets.data
y = datasets['target']

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42,
    shuffle=True,
    stratify=y,
)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : 0.0 Max : 1.0050359712230217

#2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim=54, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(8, activation='softmax'))

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
    patience=20,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=1000,
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

# [목표] acc = 0.93 이상 합격
# [참고] 데이터 50만개, 시간 측정, batch_size 작게 주지 말 것

# 시간 오래 걸림, 이 예제의 경우 0부터 시작하는 to_categorical 적합하지 않음 (다른 방법으로 시도하기)
# loss : 0.34686315059661865
# acc : 0.861
# accuracy_score : 0.8611735817881403
# 소요 시간 : 1463 초

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# loss : 0.1859578788280487
# acc : 0.936
# accuracy_score : 0.9361001468698366
# 소요 시간 : 1110 초

# [결론] 0.861 -> 0.936. 이번 실습에서 스케일링 효과가 가장 확실하게 나온 데이터다.
#        covtype 은 Elevation(1859~3858), Horizontal_Distance(0~7000) 같은 큰 값 컬럼과
#        Wilderness_Area / Soil_Type 처럼 0 아니면 1 인 컬럼이 한 테이블에 섞여 있다.
#        스케일링 전에는 값이 큰 컬럼 쪽으로만 gradient 가 크게 튀어서 학습이 제대로 안 됐던 것.
#        데이터가 58만개라 test 도 17만개 -> 이 정도 표본이면 0.861 -> 0.936 은 우연이 아니다.
#        [참고] 시간이 1463초 -> 1110초로 줄어든 것도 같은 이유다. 더 빨리 수렴해서 EarlyStopping 이 일찍 걸렸다.
#        목표였던 acc 0.93 도 스케일링만으로 넘겼다.

# ========== ========== ========== ========== ========== <- StandardScaler 적용 후

# loss : 0.17114242911338806
# acc : 0.936
# accuracy_score : 0.9360657242518817
# 소요 시간 : 866 초

# ========== ========== ========== ========== ========== <- MaxAbsScaler 적용 후

# loss : 0.22837351262569427
# acc : 0.934
# accuracy_score : 0.9338397282908023
# 소요 시간 : 1091 초

# ========== ========== ========== ========== ========== <- RobustScaler 적용 후

# loss : 0.15137867629528046
# acc : 0.943
# accuracy_score : 0.9426060216633009
# 소요 시간 : 664 초

# ========== ========== ========== ========== ========== <- learning rate 적용 후

# loss : 0.15072979032993317
# acc : 0.945
# accuracy_score : 0.9452565632458234
# 소요 시간 : 1231 초

# [결론] RobustScaler 0.9426 / 664초 -> learning_rate 0.0005 로 0.9453 / 1231초.
#        기본값 0.001 보다 작게 줬더니 acc 는 조금 올랐지만 시간이 두 배가 됐다.
#        데이터가 58만개라 test 도 17만개 -> 0.0027 차이도 우연은 아니다.
