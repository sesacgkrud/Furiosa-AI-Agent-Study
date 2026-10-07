import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_data/ddareung/'
path_save = './_save/keras68/'

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

x_train = x_train.reshape(-1, 9, 1)
x_val = x_val.reshape(-1, 9, 1)
x_test = x_test.reshape(-1, 9, 1)
print(x_train.shape, x_val.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(9, 1)))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))   # MaxPooling1D 로 시점 축 9 -> 4
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))   # 과적합 방지용 Dropout
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))   # Dense 사이에도 Dropout
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=path_save + 'keras68_conv1d_04_ddacon_ddareung.keras',
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

# ===== GPU 기록 ===== <- Conv1D 적용
# Epoch 112: early stopping
# 소요 시간 : 22.68 초
# loss : 2184.4794921875
# r2 : 0.7034536456639313
# mse : 2184.479490398244
# RMSE : 46.738415574324335