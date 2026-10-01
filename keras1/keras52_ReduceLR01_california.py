# keras52_ReduceLR01_california.py
# 캘리포니아 집값 (회귀) - ReduceLROnPlateau 로 훈련 도중 learning_rate 를 줄인다
# 스케일링은 keras27 그대로다 (split 전에 전체 x 로 fit 하는 방식이라 데이터 누수가 있다)
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
#  이 파일 : 시작 learning_rate = 0.01 / es patience = 40 / rlr patience = 20
#
# keras27_Scaler01_california.py 베이스

from sklearn.datasets import fetch_california_housing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

'''
MinMaxScaler Algorithm
-> (원값 - Min) / (Max - Min)
-> 0 ~ 1 사이로 수렴
'''

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) # sklearn 에서 fit -> 실행하다 로 생각
x = scaler.transform(x) # x에 있는 모든 데이터는 0~1 사이로 수렴
print(x)
print('Min :', np.min(x), 'Max :', np.max(x)) # Min : 0.0 Max : 1.0000000000000002 -> 부동 소숫점 연산에 의한 오차 발생 (문제X)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.75,
    random_state=100,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

########## ReduceLROnPlateau ##########
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=40,
    verbose=1,
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

hist = model.fit(x_train, y_train,
                 epochs=500,
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[es, rlr],
                 verbose=1,)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

from sklearn.metrics import r2_score, mean_squared_error

print("loss(mse) :", loss)
y_predict = model.predict(x_test).ravel()

r2 = r2_score(y_test, y_predict)

print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# loss(mse) : 0.25228360295295715
# r2 : 0.8086916350028439
# mse : 0.2522835515735914
# RMSE : 0.5022783606463566

# ===== learning rate 변경 =====
# loss(mse) : 0.2797977328300476
# r2 : 0.7878274481224297
# mse : 0.2797977230891145
# RMSE : 0.5289590939657948

# ===== ReduceLROnPlateau 변경 =====
# loss(mse) : 0.26496830582618713
# r2 : 0.7990727130026423
# mse : 0.2649682860051271
# RMSE : 0.5147507027728346

# [결론] 0.01 고정 RMSE 0.5290 -> ReduceLROnPlateau 0.5148, r2 0.7878 -> 0.7991.
#        시작은 0.01 로 똑같은데 결과가 좋아졌다. 초반에는 크게 움직여 빨리 내려가고,
#        더 이상 안 줄어드는 구간에서 보폭을 절반씩 줄여 최저점에 더 가까이 붙은 것이다.
#        es patience=40 / rlr patience=20 이라 lr 을 줄여 볼 기회가 먼저 온다.
