import numpy as np
import time

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras68/'

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

x_train = x_train.reshape(-1, 10, 1)
x_val = x_val.reshape(-1, 10, 1)
x_test = x_test.reshape(-1, 10, 1)
print(x_train.shape, x_val.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(10, 1)))   # (None, 10, 64)
model.add(Conv1D(64, 3, padding='same', activation='relu'))   # Conv1D 한 층 추가 (None, 10, 64)
model.add(MaxPooling1D(pool_size=2))                          # MaxPooling1D 로 시점 축을 절반으로 (None, 5, 64)
model.add(Conv1D(32, 2, padding='same', activation='relu'))   # 줄어든 시점 위에서 Conv1D 한 번 더 (None, 5, 32)
model.add(Dropout(0.2))                                       # 데이터가 적어서(약 216개) 과적합 방지용 Dropout
model.add(Flatten())                                          # 3차원을 2차원으로 펴서 Dense 에 연결 (None, 160)

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))                                       # Dense 사이에도 Dropout
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,   # Dropout 으로 val_loss 가 출렁여서 patience 20 -> 30
    restore_best_weights=True,
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=path + 'keras68_conv1d_02_diabetes.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=300,
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

# ===== GPU 기록 ===== <- Conv1D 적용
# 소요 시간 : 10.0 초
# loss : 2937.562255859375
# r2 : 0.48796043779911313
# mse : 2937.562023791979
# RMSE : 54.19928065751407
