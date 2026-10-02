# keras1/keras42_cnn3_boston.py 베이스

import numpy as np
import time

from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM    # [LSTM] Conv2D, Flatten -> LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras64/'

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# [LSTM] Conv2D 용 4차원 (N, 13, 1, 1) -> LSTM 용 3차원 (N, timesteps, feature) = (N, 13, 1)
#  컬럼 13개를 시점 13개로 보고, 시점마다 값 1개씩 순서대로 읽는다
x_train = x_train.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)
print(x_train.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
# [LSTM] Conv2D 3층 + Flatten -> LSTM 1층
#  LSTM 은 마지막 시점의 출력만 내보내서 (None, 64) 2차원 -> Flatten 없이 바로 Dense 에 연결된다
model.add(LSTM(64, input_shape=(13, 1)))
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
    filepath=path + 'keras64_mcp3.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=100,
                 batch_size=32,
                 validation_split=0.3,
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
# 소요 시간 : 7.96 초
# loss : 24.035165786743164
# r2 : 0.7112679690358323
# mse : 24.0351654483954
# RMSE : 4.902567230379956

# ===== GPU 기록 =====
# 소요 시간 : 5.92 초
# loss : 20.36505889892578
# r2 : 0.7553565692056211
# mse : 20.365060694411476
# RMSE : 4.512766412569066

# ===== GPU 기록 ===== <- Conv2D 적용
# 소요 시간 : 8.37 초
# loss : 18.92754554748535
# r2 : 0.7726252752728436
# mse : 18.927547142418646
# RMSE : 4.35058009263347

# ===== GPU 기록 ===== <- LSTM 적용
# loss : 24.399946212768555
# r2 : 0.7068858902287075
# mse : 24.39994655281753
# RMSE : 4.939630204055515