# keras52_ReduceLR04_dacon_ddareung.py
# 따릉이 (회귀, [방법 1] - x_val 을 직접 만들어 validation_data 로 넘긴다) - ReduceLROnPlateau 적용
# https://dacon.io/competitions/open/235576/overview/description
# 스케일러(RobustScaler)와 learning_rate 후보는 keras52_optimizer 파일 그대로다 (x_train / x_test / x_val 모두 변환)
#
# [ 오늘 추가되는 곳은 콜백 하나다 ]
#  keras52_optimizer 는 learning_rate 를 처음부터 끝까지 고정했다
#  하지만 한 값으로 두 가지를 다 할 수는 없다
#   초반에는 크게 움직여 빨리 내려가야 하고, 최저점 근처에서는 작게 움직여야 지나치지 않는다
#
# [ ReduceLROnPlateau : 훈련 도중 learning_rate 를 줄여 준다 ]
#  plateau = 고원. val_loss 가 더 이상 줄지 않고 평평해지는 구간을 말한다
#  monitor 가 patience 동안 좋아지지 않으면 learning_rate 에 factor 를 곱한다
#   factor=0.5 -> 0.0008 -> 절반 -> 또 절반 ... 으로 떨어진다
#  EarlyStopping 과 역할이 다르다 : es 는 '멈춘다', rlr 은 '보폭을 줄여 더 해 본다'
#   그래서 rlr 의 patience 를 es 보다 짧게 줘야 "줄여 보고 -> 그래도 안 되면 멈춘다" 순서가 된다
#  이 파일 : 시작 learning_rate = 0.0008 / es patience = 20 / rlr patience = 20
#   -> 둘이 같아서 lr 이 줄어드는 바로 그 epoch 에 es 도 같이 걸린다
#
# keras28_Scaler04_dacon_ddareung.py 베이스

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = "./_data/ddareung/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission = pd.read_csv(path + "submission.csv", index_col=0)

########## 결측치 처리 1. 삭제 ##########

train_csv = train_csv.dropna()

# ★★★ train_csv를 x와 y로 분리 ★★★
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,     # [변경] 원본은 비율 생략(기본값 0.75) -> 의도를 명확히 하려고 명시
    random_state=222,
)

# 2단계 : train -> train(80%) / val(20%)
x_train, x_val, y_train, y_val = train_test_split(   # [추가] x_val, y_val 을 여기서 만든다
    x_train, y_train,   # ★ x_test 가 아니라 위에서 만든 x_train 을 다시 자른다
    train_size=0.8,     # 전체 기준으로는 0.8 * 0.2 = 16% 가 val 이 된다
    random_state=77,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

# scaler.fit(x_train) # sklearn 에서 fit -> 실행하다 로 생각
# x_train = scaler.transform(x_train) # x에 있는 모든 데이터는 0~1 사이로 수렴
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴
x_val = scaler.transform(x_val)

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : -0.022222222222222227 Max : 1.0
print('Min :', np.min(x_val), 'Max :', np.max(x_val)) # val 도 0~1 근처로 들어왔는지 같이 확인

#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009
learning_rate = 0.0008

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
    epochs=1000,
    batch_size=16,
    validation_data=(x_val, y_val),   # [추가] 위에서 만든 val 세트를 넘긴다 -> val_loss 출력
    # validation_split=0.2,           # [방법 2] 는 x_val 없이 이 줄 하나로 끝낸다
    callbacks=[es, rlr],
)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
# 평가는 훈련에도 검증에도 쓰지 않은 x_test 로만 한다.
loss = model.evaluate(x_test, y_test, )

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
plt.plot(hist.history['loss'][2:], c='red', label='loss') # y값만 넣으면 시간 순으로 그림
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')

plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title('ddareung Loss')

plt.xlabel('epochs')
plt.xlabel('loss')

plt.grid() # 격자 표시 추가
plt.show()

# loss: 3104.0977 - val_loss: 3415.0693
# loss: 3192.954

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후 (x_val 누락 상태의 잘못된 결과)

# loss(mse) : 15935.1015625
# r2 : -1.1632139132191313        <- r2 가 음수 = 평균값만 찍는 모델보다 못하다는 뜻
# mse : 15935.10207654089
# RMSE : 126.23431418018197

# (위 결과는 x_val 스케일링 없이 나온 값이라 스케일링 성능 비교에는 쓰지 않는다)

# ========== ========== ========== ========== ========== <- x_val 까지 스케일링한 최종 결과

# loss(mse) : 1995.118408203125
# r2 : 0.7291596889111879
# mse : 1995.1184565098156
# RMSE : 44.66674889120335

# [결론] 3192 -> 1995 (약 37% 개선), r2 도 0.729 로 정상 범위에 들어왔다.
#        x_val 한 줄을 빼먹었을 때 r2 -1.16 이었던 것과 비교하면,
#        스케일링은 x_train / x_test / x_val 중 하나만 빠져도 결과가 통째로 망가진다는 것을 알 수 있다.
#        따릉이는 hour(0~23), 온도, 습도, 미세먼지 농도처럼 컬럼 단위가 제각각이라 효과가 크다.

# ========== ========== ========== ========== ========== <- StandardScaler 적용 후

# loss(mse) : 2522.88916015625
# r2 : 0.6575140348819886
# mse : 2522.8891052279823
# RMSE : 50.22836952587633

# ========== ========== ========== ========== ========== <- MaxAbsScaler 적용 후

# loss(mse) : 2175.078369140625
# r2 : 0.7047298300457296
# mse : 2175.0786039356653
# RMSE : 46.63773798047741

# ========== ========== ========== ========== ========== <- RobustScaler 적용 후

# loss(mse) : 1870.7774658203125
# r2 : 0.7460391594998328
# mse : 1870.7775001280418
# RMSE : 43.25248547919578

# ========== ========== ========== ========== ========== <- learning rate 적용 후

# loss(mse) : 1759.0279541015625
# r2 : 0.7612093218542999
# mse : 1759.0279943769463
# RMSE : 41.940767689408666

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후

# loss(mse) : 2280.103515625
# r2 : 0.6904725251642657
# mse : 2280.1036350868394
# RMSE : 47.750430731950885

# [결론] 0.0018 고정 RMSE 41.94 -> 시작 0.0008 + ReduceLROnPlateau RMSE 47.75 로 나빠졌다.
#        시작값이 0.0008 로 이미 작은데 거기서 또 절반씩 줄어드니 최저점까지 못 갔다.
#        ReduceLROnPlateau 는 "시작값을 넉넉히 주고 줄여 나간다" 가 전제다.
