# keras1/keras42_cnn6_cancer.py 베이스

import numpy as np
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout    # [LSTM] Conv2D, Flatten -> LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras64/'

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

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

# [LSTM] Conv2D 용 4차원 (N, 5, 6, 1) -> LSTM 용 3차원 (N, timesteps, feature) = (N, 30, 1)
#  컬럼 30개를 시점 30개로 보고, 시점마다 값 1개씩 순서대로 읽는다
x_train = x_train.reshape(-1, 30, 1)
x_test = x_test.reshape(-1, 30, 1)
print(x_train.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
# [LSTM] Conv2D 3층 + Flatten -> LSTM 1층
#  LSTM 은 마지막 시점의 출력만 내보내서 (None, 64) 2차원 -> Flatten 없이 바로 Dense 에 연결된다
model.add(LSTM(64, input_shape=(30, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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
    filepath=path + 'keras64_mcp6.keras',
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

y_predict_arg = np.round(y_predict)
acc_score = accuracy_score(y_test, y_predict_arg)
print('accuracy_score :', acc_score)

# ===== CPU 기록 =====
# 소요 시간 : 8.34 초
# loss : 0.13056117296218872
# acc : 0.9649
# r2 : 0.8715246319770813
# mse : 0.030087867751717567
# RMSE : 0.17345854764674346
# acc_score : 0.9649122807017544

# ===== GPU 기록 =====
# 소요 시간 : 6.49 초
# loss : 0.1108744665980339
# acc : 0.9591
# r2 : 0.8735296726226807
# mse : 0.029618313536047935
# RMSE : 0.17209971974424576
# acc_score : 0.9590643274853801

# ===== GPU 기록 ===== <- Conv2D 적용
# Epoch 51: early stopping
# 소요 시간 : 5.99 초
# loss : 0.14266230165958405
# acc : 0.965
# r2 : 0.8670377135276794
# mse : 0.031138665974140167
# RMSE : 0.1764615141444167
# accuracy_score : 0.9649122807017544

# ===== GPU 기록 ===== <- LSTM 적용
# Epoch 70: early stopping
# 소요 시간 : 7.91 초
# loss : 0.24084244668483734
# acc : 0.912
# r2 : 0.6972483992576599
# mse : 0.07090192288160324
# RMSE : 0.26627414985612713
# accuracy_score : 0.9122807017543859