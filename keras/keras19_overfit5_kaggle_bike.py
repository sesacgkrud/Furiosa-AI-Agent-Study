# keras19_overfit5_kaggle_bike.py
# 과적합 확인 : loss 와 val_loss 를 같이 그려서 val_loss 가 올라가기 시작하는 지점을 찾는다

# https://www.kaggle.com/competitions/bike-sharing-demand/data
# [방법 2] validation_split 으로 fit 이 x_train 에서 알아서 val 을 떼어가게 하는 방식

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = "./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv) # [10886 rows x 11 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [6493 rows x 8 columns]

submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

#################### x, y 분리 ####################

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# print(x) # [10886 rows x 8 columns]

y = train_csv['count']
print(y, y.shape) # Name: count, Length: 10886, dtype: int64 (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,     # [변경] 원본은 비율 생략(기본값 0.75) -> 의도를 명확히 하려고 명시
    random_state=100,   # shuffle 은 기본값 True -> 여기서 날짜 순서가 섞인다
)

# print(x_train.shape, x_test.shape)   # (8708, 8) (2178, 8)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu')) # 통상적으로 마지막에는 relu 넣지 않음

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(
    x_train, y_train,
    epochs=500,
    batch_size=16,
    validation_split=0.2,             # [추가] x_train 의 20%를 검증용으로 사용 -> val_loss 출력
    # validation_data=(x_val, y_val), # [방법 1] 로 x_val 을 만들었다면 위 줄 대신 이 줄을 쓴다
)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
# 평가는 훈련에도 검증에도 쓰지 않은 x_test 로만 한다.
loss = model.evaluate(x_test, y_test, )

print("loss(mse) :", loss)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)     # [변경] print 를 감싸고 있던 불필요한 괄호 제거

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
plt.plot(hist.history['loss'], c='red', label='loss') # y값만 넣으면 시간 순으로 그림
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')

plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title('California(캘리포니아) Loss')

plt.xlabel('epochs')
plt.xlabel('loss')

plt.grid() # 격자 표시 추가
plt.show()

# loss: 21725.1992 - val_loss: 22583.5430
# loss: 21853.4062