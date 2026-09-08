# https://dacon.io/competitions/open/235576/overview/description
# [방법 1] train_test_split 을 2번 써서 x_val 을 직접 만들고 validation_data 로 넘기는 방식

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = "./_data/ddareung/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission = pd.read_csv(path + "submission.csv", index_col=0)

########## 결측치 처리 1. 삭제 ##########

train_csv = train_csv.dropna()

# ★★★ train_csv를 x와 y로 분리 ★★★
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,     # [변경] 원본은 비율 생략(기본값 0.75) -> 의도를 명확히 하려고 명시
    random_state=222,
)

# 2단계 : train -> train(80%) / val(20%)
x_train, x_val, y_train, y_val = train_test_split(   # [추가] x_val, y_val 을 여기서 만든다
    x_train, y_train,   # ★ x_test 가 아니라 위에서 만든 x_train 을 다시 자른다
    train_size=0.8,     # 전체 기준으로는 0.8 * 0.2 = 16% 가 val 이 된다
    random_state=77,
)

# print(x_train.shape, x_val.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
)

hist = model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=16,
    validation_data=(x_val, y_val),   # [추가] 위에서 만든 val 세트를 넘긴다 -> val_loss 출력
    # validation_split=0.2,           # [방법 2] 는 x_val 없이 이 줄 하나로 끝낸다
    callbacks=[es],
)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
# 평가는 훈련에도 검증에도 쓰지 않은 x_test 로만 한다.
loss = model.evaluate(x_test, y_test, )

print("loss(mse) :", loss)
y_predict = model.predict(x_test).ravel()

r2 = r2_score(y_test, y_predict)

print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

print("========================= history =========================")
print(hist)
print("========================= history =========================")
print(hist.history)
print("========================= loss =========================")
print(hist.history['loss'])
print("========================= val_loss =========================")
print(hist.history['val_loss'])
print("========================= The End =========================")


import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic' # 글자 깨짐 해결

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:], c='red', label='loss') # y값만 넣으면 시간 순으로 그림
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')

plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title('ddareung Loss')

plt.xlabel('epochs')
plt.xlabel('loss')

plt.grid() # 격자 표시 추가
plt.show()

# loss: 3104.0977 - val_loss: 3415.0693
# loss: 3192.954