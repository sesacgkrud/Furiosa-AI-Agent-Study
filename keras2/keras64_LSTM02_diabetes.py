# keras1/keras42_cnn2_diabetes.py 베이스

import numpy as np
import time

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM    # [LSTM] Conv2D, Flatten -> LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras64/'

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=77,
    shuffle=True,
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.7,
    random_state=77,
    shuffle=True,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# [LSTM] Conv2D 용 4차원 (N, 5, 2, 1) -> LSTM 용 3차원 (N, timesteps, feature) = (N, 10, 1)
#  컬럼 10개를 시점 10개로 보고, 시점마다 값 1개씩 순서대로 읽는다 (x_val 도 똑같이 바꿔야 한다)
x_train = x_train.reshape(-1, 10, 1)
x_val = x_val.reshape(-1, 10, 1)
x_test = x_test.reshape(-1, 10, 1)
print(x_train.shape, x_val.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
# [LSTM] Conv2D 3층 + Flatten -> LSTM 1층
#  LSTM 은 마지막 시점의 출력만 내보내서 (None, 64) 2차원 -> Flatten 없이 바로 Dense 에 연결된다
model.add(LSTM(64, input_shape=(10, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=path + 'keras64_mcp2.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=100,
                 batch_size=32,
                 validation_data=(x_val, y_val),
                 callbacks=[es, mcp],
                 verbose=1,
                 )

end_time = time.time()

print("소요 시간 :", round(end_time - start_time, 2), "초")

print("========== ========== ========== ========== ==========")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# ===== CPU 기록 =====
# 소요 시간 : 8.13 초
# loss : 3353.86865234375
# r2 : 0.4153950162220976
# mse : 3353.868579771466
# RMSE : 57.91259431049058

# ===== GPU 기록 =====
# 소요 시간 : 6.36 초
# loss : 3307.038330078125
# r2 : 0.4235579227575228
# mse : 3307.0381275713275
# RMSE : 57.50685287486464

# ===== GPU 기록 ===== <- Conv2D 적용
# 소요 시간 : 8.1 초
# loss : 3156.532470703125
# r2 : 0.4497921663401562
# mse : 3156.5327304102047
# RMSE : 56.1830288468876

# ===== GPU 기록 ===== <- LSTM 적용
# loss : 3273.363037109375
# r2 : 0.4294277388201354
# mse : 3273.3630953569027
# RMSE : 57.21331222151802