# keras52_ReduceLR05_kaggle_bike.py
# 캐글 자전거 (회귀, [방법 2] - validation_split 사용) - ReduceLROnPlateau 적용
# https://www.kaggle.com/competitions/bike-sharing-demand/data
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
#   factor=0.5 -> 0.01 -> 절반 -> 또 절반 ... 으로 떨어진다
#  EarlyStopping 과 역할이 다르다 : es 는 '멈춘다', rlr 은 '보폭을 줄여 더 해 본다'
#   그래서 rlr 의 patience 를 es 보다 짧게 줘야 "줄여 보고 -> 그래도 안 되면 멈춘다" 순서가 된다
#  이 파일 : 시작 learning_rate = 0.01 / es patience = 20 / rlr patience = 20
#   -> 둘이 같아서 lr 이 줄어드는 바로 그 epoch 에 es 도 같이 걸린다
#
# keras28_Scaler05_kaggle_bike.py 베이스

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = "./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv) # [10886 rows x 11 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [6493 rows x 8 columns]

submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

#################### x, y 분리 ####################

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# print(x) # [10886 rows x 8 columns]

y = train_csv['count']
print(y, y.shape) # Name: count, Length: 10886, dtype: int64 (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,     # [변경] 원본은 비율 생략(기본값 0.75) -> 의도를 명확히 하려고 명시
    random_state=100,   # shuffle 은 기본값 True -> 여기서 날짜 순서가 섞인다
)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : 0.0 Max : 1.0425531914893618

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu')) # 통상적으로 마지막에는 relu 넣지 않음

#3. 컴파일, 훈련
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
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

hist = model.fit(
    x_train, y_train,
    epochs=500,
    batch_size=16,
    validation_split=0.2,             # [추가] x_train 의 20%를 검증용으로 사용 -> val_loss 출력
    # validation_data=(x_val, y_val), # [방법 1] 로 x_val 을 만들었다면 위 줄 대신 이 줄을 쓴다
    callbacks=[es, rlr],
)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
# 평가는 훈련에도 검증에도 쓰지 않은 x_test 로만 한다.
loss = model.evaluate(x_test, y_test, )

print("loss(mse) :", loss)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)     # [변경] print 를 감싸고 있던 불필요한 괄호 제거

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

print("========================= history =========================")
print(hist)
print("========================= history =========================")
print(hist.history)
print("========================= loss =========================")
print(hist.history['loss'])
print("========================= val_loss =========================")
print(hist.history['val_loss'])
print("========================= The End =========================")


import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic' # 글자 깨짐 해결

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss') # y값만 넣으면 시간 순으로 그림
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')

plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title('California(캘리포니아) Loss')

plt.xlabel('epochs')
plt.xlabel('loss')

plt.grid() # 격자 표시 추가
plt.show()

# loss: 22175.3379 - val_loss: 22727.3770
# loss: 21894.7402

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# loss(mse) : 22418.43359375
# r2 : 0.2820294499397278
# mse : 22418.431640625
# RMSE : 149.72785859894276

# [결론] 21894 -> 22418 로 거의 그대로다(약 2% 차이).
#        bike 의 x 컬럼은 season(1~4), weather(1~4), temp(0~41), humidity(0~100) 정도라
#        원래 범위 차이가 크지 않아서 스케일링으로 얻을 게 별로 없다.
#        게다가 seed 를 고정하지 않아서 이 정도 차이는 실행할 때마다 생기는 편차 범위다.
#        -> '좋아졌다/나빠졌다'라고 결론 내리면 안 되는 케이스.

# ========== ========== ========== ========== ========== <- StandardScaler 적용 후

# loss(mse) : 21298.0078125
# r2 : 0.31791210174560547
# mse : 21298.005859375
# RMSE : 145.9383632201451

# ========== ========== ========== ========== ========== <- MaxAbsScaler 적용 후

# loss(mse) : 21775.37109375
# r2 : 0.3026239275932312
# mse : 21775.375
# RMSE : 147.5648162672932

# ========== ========== ========== ========== ========== <- RobustScaler 적용 후

# loss(mse) : 22058.591796875
# r2 : 0.2935537099838257
# mse : 22058.58984375
# RMSE : 148.5213447412526

# ========== ========== ========== ========== ========== <- learning rate 적용 후

# loss(mse) : 21586.302734375
# r2 : 0.3086792230606079
# mse : 21586.298828125
# RMSE : 146.92276483964287

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후

# loss(mse) : 21324.048828125
# r2 : 0.3170781135559082
# mse : 21324.046875
# RMSE : 146.02755519079267

# [결론] 0.01 고정 RMSE 146.92 -> ReduceLROnPlateau RMSE 146.03. 거의 같다.
#        bike 는 스케일링 때도 효과가 없던 데이터라 여기서도 크게 달라지지 않는다.
