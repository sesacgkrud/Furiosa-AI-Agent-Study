import numpy as np
import time

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

path = './_save/keras68/'

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=50,
    shuffle=True,
    stratify=y,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)
print(x_train.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(13, 1)))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True,
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=path + 'keras68_conv1d_08_wine.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=300,
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
# 소요 시간 : 8.02 초
# loss : 0.1516449749469757
# acc : 0.963
# r2 : 0.9148545485493251
# mse : 0.01852531937049567
# RMSE : 0.13610774911993684
# accuracy_score : 0.9629629629629629

# ===== GPU 기록 =====
# 소요 시간 : 4.65 초
# loss : 0.0985560491681099
# acc : 0.981
# r2 : 0.9433357119560242
# mse : 0.01232890784740448
# RMSE : 0.111035615220543
# accuracy_score : 0.9814814814814815

# ===== GPU 기록 ===== <- Conv2D 적용
# 소요 시간 : 6.49 초
# loss : 0.16413834691047668
# acc : 0.963
# r2 : 0.9251127243041992
# mse : 0.016293292865157127
# RMSE : 0.1276451834780973
# accuracy_score : 0.9629629629629629

# ===== GPU 기록 ===== <- LSTM 적용
# 소요 시간 : 7.37 초
# loss : 0.2549792528152466
# acc : 0.926
# r2 : 0.8337950706481934
# mse : 0.03616282716393471
# RMSE : 0.19016526276882093
# accuracy_score : 0.9259259259259259

# ===== GPU 기록 ===== <- Conv1D 적용
# Epoch 146: early stopping
# 소요 시간 : 7.88 초
# loss : 0.026557112112641335
# acc : 0.981
# r2 : 0.9677550792694092
# mse : 0.00701557332649827
# RMSE : 0.08375901937402484
# accuracy_score : 0.9814814814814815