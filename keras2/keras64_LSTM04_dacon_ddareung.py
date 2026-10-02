# keras1/keras42_cnn4_dacon_ddareung.py 베이스

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout    # [LSTM] Conv2D, Flatten -> LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_data/ddareung/'
path_save = './_save/keras64/'

#1. 데이터
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=222,
    shuffle=True,
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.8,
    random_state=77,
    shuffle=True,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# [LSTM] Conv2D 용 4차원 (N, 3, 3, 1) -> LSTM 용 3차원 (N, timesteps, feature) = (N, 9, 1)
#  컬럼 9개를 시점 9개로 보고, 시점마다 값 1개씩 순서대로 읽는다 (x_val 도 똑같이 바꿔야 한다)
x_train = x_train.reshape(-1, 9, 1)
x_val = x_val.reshape(-1, 9, 1)
x_test = x_test.reshape(-1, 9, 1)
print(x_train.shape, x_val.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
# [LSTM] Conv2D 3층 + Flatten -> LSTM 1층
#  LSTM 은 마지막 시점의 출력만 내보내서 (None, 64) 2차원 -> Flatten 없이 바로 Dense 에 연결된다
model.add(LSTM(64, input_shape=(9, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
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
    filepath=path_save + 'keras64_mcp4.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=1000,
                 batch_size=16,
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

# ===== 이전 기록 =====
# Sequential + Dropout (keras33_dropout04, 2026-09-14)  loss(mse) : 2411.2744140625 / r2 : 0.6726659193650188

# ===== 실행 결과 (2026-09-14, 함수형 + Dropout + MCP + r2/mse/rmse, keras33 과 같은 구성) =====
# loss(mse) : 2727.140869140625
# r2 : 0.6297865601223622
# mse : 2727.140815111746
# RMSE : 52.22203380864964
# (Epoch 94: early stopping / Restoring model weights from the end of the best epoch: 74.)

# ===== GPU 기록 ===== <- Conv2D 적용
# Epoch 115: early stopping
# 소요 시간 : 22.04 초
# loss : 2130.4580078125
# r2 : 0.7107871256030921
# mse : 2130.458133244115
# RMSE : 46.15688608695473

# ===== GPU 기록 ===== <- LSTM 적용
# Epoch 168: early stopping
# 소요 시간 : 35.75 초
# loss : 2532.836669921875
# r2 : 0.6561636632515002
# mse : 2532.836484745838
# RMSE : 50.32729363621531