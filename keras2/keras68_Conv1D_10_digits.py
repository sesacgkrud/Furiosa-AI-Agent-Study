# keras1/keras42_cnn10_digits.py 베이스

import numpy as np
import time

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

path = './_save/keras68/'

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.6,
    random_state=99,
    shuffle=True,
    stratify=y,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 8)
x_test = x_test.reshape(-1, 8, 8)
print(x_train.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(8, 8)))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=path + 'keras68_conv1d_10_digits.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=300,
                 batch_size=4,
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
# 소요 시간 : 29.17 초
# loss : 0.27403485774993896
# acc : 0.964
# r2 : 0.92851891938515
# mse : 0.006412560070352405
# RMSE : 0.08007846196295484
# accuracy_score : 0.9638386648122392

# ===== GPU 기록 =====
# 소요 시간 : 58.05 초
# loss : 0.24361562728881836
# acc : 0.955
# r2 : 0.9139814376831055
# mse : 0.0077177248895168304
# RMSE : 0.08785058275001272
# accuracy_score : 0.9554937413073713

# ===== GPU 기록 ===== <- Conv2D 적용
# 소요 시간 : 132.79 초
# loss : 0.2607406973838806
# acc : 0.971
# r2 : 0.9462674856185913
# mse : 0.004828672856092453
# RMSE : 0.06948865271461559
# accuracy_score : 0.9707927677329624

# ===== GPU 기록 ===== <- LSTM 적용
# 소요 시간 : 67.96 초
# loss : 0.12751145660877228
# acc : 0.972
# r2 : 0.9470843076705933
# mse : 0.004755221772938967
# RMSE : 0.06895811607736226
# accuracy_score : 0.972183588317107

# ===== GPU 기록 ===== <- Conv1D 적용
# Epoch 172: early stopping
# 소요 시간 : 107.26 초
# loss : 0.05903158709406853
# acc : 0.979
# r2 : 0.9675434827804565
# mse : 0.002915413584560156
# RMSE : 0.05399456995439593
# accuracy_score : 0.9791376912378303