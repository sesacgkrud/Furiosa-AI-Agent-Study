# keras1/keras42_cnn1_california.py 베이스

import numpy as np
import time

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM    # [LSTM] Conv2D, Flatten -> LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras64/'

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.75,
    random_state=100,
    shuffle=True,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) # (15480, 8)
print(x_test.shape)  # (5160, 8)

# [LSTM] Conv2D 용 4차원 (N, 4, 2, 1) -> LSTM 용 3차원 (N, timesteps, feature) = (N, 8, 1)
#  컬럼 8개를 시점 8개로 보고, 시점마다 값 1개씩 순서대로 읽는다
x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)
print(x_train.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
# [LSTM] Conv2D 3층 + Flatten -> LSTM 1층
#  LSTM 은 마지막 시점의 출력만 내보내서 (None, 64) 2차원 -> Flatten 없이 바로 Dense 에 연결된다
model.add(LSTM(64, input_shape=(8, 1)))
model.add(Dense(128, activation='relu'))
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
    filepath=path + 'keras64_mcp1.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=100,
                 batch_size=32,
                 validation_split=0.2,
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
# 소요 시간 : 39.3 초
# loss : 0.26291415095329285
# r2 : 0.8006303903866829
# mse : 0.2629141344124486
# RMSE : 0.5127515328231097

# ===== GPU 기록 =====
# 소요 시간 : 59.3 초
# loss : 0.26191774010658264
# r2 : 0.801385968725164
# mse : 0.26191773267786206
# RMSE : 0.5117789881168062

# ===== GPU 기록 ===== <- Conv2D 적용
# Epoch 79: early stopping
# 소요 시간 : 238.17 초
# loss : 0.2639758884906769
# r2 : 0.799825255464116
# mse : 0.2639758877642382
# RMSE : 0.5137858384232075

# ===== GPU 기록 ===== <- LSTM 적용
# loss : 0.26240935921669006
# r2 : 0.801013249513262
# mse : 0.26240924765432294
# RMSE : 0.5122589654211266