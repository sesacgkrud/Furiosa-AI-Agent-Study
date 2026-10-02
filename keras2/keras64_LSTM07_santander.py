# keras1/keras42_cnn7_santander.py 베이스

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout    # [LSTM] Conv2D, Flatten -> LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

path = './_data/kaggle_santander/'
path_save = './_save/keras64/'

#1. 데이터
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape)

num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=777,
    shuffle=True,
    stratify=np.argmax(y, axis=1),
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# [LSTM] Conv2D 용 4차원 (N, 10, 20, 1) -> LSTM 용 3차원 (N, timesteps, feature) = (N, 200, 1)
#  컬럼 200개를 시점 200개로 보고, 시점마다 값 1개씩 순서대로 읽는다
#  시점이 200개라 LSTM 이 한 샘플마다 200번 순서대로 계산한다 -> Conv2D 보다 훈련이 훨씬 느리다
x_train = x_train.reshape(-1, 200, 1)
x_test = x_test.reshape(-1, 200, 1)
print(x_train.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
# [LSTM] Conv2D 3층 + Flatten -> LSTM 1층
#  LSTM 은 마지막 시점의 출력만 내보내서 (None, 64) 2차원 -> Flatten 없이 바로 Dense 에 연결된다
model.add(LSTM(64, input_shape=(200, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
    filepath=path_save + 'keras64_mcp7.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=10,
                 batch_size=32,
                 validation_split=0.2,
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
# 소요 시간 : 65.38 초
# loss : 0.26788538694381714
# acc : 0.9085
# r2 : 0.1891651251119561
# mse : 0.07328847213823145
# RMSE : 0.27071843701202075
# acc_score : 0.9085166666666666

# ===== GPU 기록 =====
# 소요 시간 : 77.38 초
# loss : 0.2606703042984009
# acc : 0.9087
# r2 : 0.19318419694900513
# mse : 0.07292849570512772
# RMSE : 0.2700527646685509
# acc_score : 0.9087333333333333

# ===== GPU 기록 ===== <- Conv2D 적용
# 소요 시간 : 84.21 초
# loss : 0.24611467123031616
# acc : 0.911
# r2 : 0.2405877709388733
# mse : 0.06864365935325623
# RMSE : 0.2619993499099878
# accuracy_score : 0.91095

# ===== GPU 기록 ===== <- LSTM 적용
# 소요 시간 : 369.84 초
# loss : 0.3194146156311035
# acc : 0.9
# r2 : 0.014550715684890747
# mse : 0.08907526731491089
# RMSE : 0.29845479945028675
# accuracy_score : 0.8995166666666666