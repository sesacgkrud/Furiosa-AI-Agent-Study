import numpy as np
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras68/'

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

x_train = x_train.reshape(-1, 30, 1)
x_test = x_test.reshape(-1, 30, 1)
print(x_train.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(30, 1)))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))   # MaxPooling1D 한 번 더 : 시점 축 15 -> 7
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 이진 분류 + binary_crossentropy 라 출력을 0~1 로 만드는 sigmoid 추가

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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
    filepath=path + 'keras68_conv1d_06_cancer.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=300,   # EarlyStopping 이 멈춰주므로 epochs 100 -> 300
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

# ===== GPU 기록 ===== <- Conv1D 적용
# Epoch 40: early stopping
# 소요 시간 : 5.28 초
# loss : 0.11899332702159882
# acc : 0.953
# r2 : 0.8446009159088135
# mse : 0.0363931767642498
# RMSE : 0.19076995770888508
# accuracy_score : 0.9532163742690059