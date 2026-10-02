# keras1/keras42_cnn9_fetch_covtype.py 베이스

import numpy as np
import time

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM    # [LSTM] Conv2D, Flatten -> LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

path = './_save/keras64/'

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42,
    shuffle=True,
    stratify=y,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# [LSTM] Conv2D 용 4차원 (N, 6, 9, 1) -> LSTM 용 3차원 (N, timesteps, feature) = (N, 54, 1)
#  컬럼 54개를 시점 54개로 보고, 시점마다 값 1개씩 순서대로 읽는다
x_train = x_train.reshape(-1, 54, 1)
x_test = x_test.reshape(-1, 54, 1)
print(x_train.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
# [LSTM] Conv2D 3층 + Flatten -> LSTM 1층
#  LSTM 은 마지막 시점의 출력만 내보내서 (None, 64) 2차원 -> Flatten 없이 바로 Dense 에 연결된다
model.add(LSTM(64, input_shape=(54, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
    filepath=path + 'keras64_mcp9.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=10,
                 batch_size=32,
                 validation_split=0.3,
                 callbacks=[es, mcp],
                 verbose=1,
                 )

end_time = time.time()

print("소요 시간 :", round(end_time - start_time, 2), "초")

print("========== ========== ========== ========== ==========")

#4. 평가 예측
result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('acc :', round(result[1], 3))

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

y_test_arg = np.argmax(y_test, axis=1)
y_predict_arg = np.argmax(y_predict, axis=1)
acc_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', acc_score)

# ===== CPU 기록 =====
# 소요 시간 : 136.02 초
# loss : 0.2725818455219269
# acc : 0.889
# r2 : 0.6215999397276787
# mse : 0.020001761734731602
# RMSE : 0.14142758477302653
# accuracy_score : 0.8892337525243253

# ===== GPU 기록 =====
# 소요 시간 : 234.58 초
# loss : 0.2714778184890747
# acc : 0.892
# r2 : 0.6270689964294434
# mse : 0.01972629874944687
# RMSE : 0.1404503426462423
# accuracy_score : 0.8924063704791628

# ===== GPU 기록 ===== <- Conv2D 적용
# 소요 시간 : 248.45 초
# loss : 0.23483967781066895
# acc : 0.906
# r2 : 0.6672987341880798
# mse : 0.017069324851036072
# RMSE : 0.13064962629504942
# accuracy_score : 0.9060664127042408

# ===== GPU 기록 ===== <- LSTM 적용
# 소요 시간 : 499.69 초
# loss : 0.43571028113365173
# acc : 0.822
# r2 : 0.469210147857666
# mse : 0.031942229717969894
# RMSE : 0.1787238924094087
# accuracy_score : 0.8223907655590234