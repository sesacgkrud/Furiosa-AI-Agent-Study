# keras1/keras42_cnn5_kaggle_bike.py 베이스

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout    # [LSTM] Conv2D, Flatten -> LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_data/kaggle_bike/'
path_save = './_save/keras64/'

#1. 데이터
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=100,
    shuffle=True,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

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
    filepath=path_save + 'keras64_mcp5.keras',
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
# 소요 시간 : 24.9 초
# loss(mse) : 22356.04296875
# r2 : 0.2840275764465332
# mse : 22356.041015625
# RMSE : 149.5193666908237

# ===== GPU 기록 =====
# 소요 시간 : 42.76 초
# loss(mse) : 22352.484375
# r2 : 0.28414130210876465
# mse : 22352.490234375
# RMSE : 149.50749223492113

# ===== GPU 기록 ===== <- Conv2D 적용
# Epoch 71: early stopping
# 소요 시간 : 45.25 초
# loss : 21706.837890625
# r2 : 0.3048190474510193
# mse : 21706.833984375
# RMSE : 147.33239285498286

# ===== GPU 기록 ===== <- LSTM 적용
# 소요 시간 : 70.85 초
# loss : 20865.236328125
# r2 : 0.3317719101905823
# mse : 20865.23828125
# RMSE : 144.4480469970086