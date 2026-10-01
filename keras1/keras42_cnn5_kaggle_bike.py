import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_data/kaggle_bike/'       # 데이터 폴더
path_save = './_save/keras34/'      # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

#1. 데이터
train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # 제출을 안 하므로 test.csv 는 읽지 않는다

# casual + registered = count 라서 입력에 넣으면 정답을 알려주는 셈 -> 같이 뺀다
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)  # 입력 8개 컬럼
y = train_csv['count']                                          # 정답 = 대여 수
print(x.shape, y.shape) # (10886, 8) (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=100,
    shuffle=True,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

# Conv2D 는 (행, 열, 채널) 4차원 입력이 필요하다 -> 컬럼 8개를 4 x 2 x 1 로 바꾼다
x_train = x_train.reshape(-1, 4, 2, 1)  # (8708, 4, 2, 1)
x_test = x_test.reshape(-1, 4, 2, 1)    # (2178, 4, 2, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', activation='relu', input_shape=(4, 2, 1)))  # 출력 : (4, 2, 64)  param 320 = (2x2x1+1)x64
model.add(Conv2D(64, (2,2), activation='relu'))                                         # 출력 : (3, 1, 64)  param 16448 = (2x2x64+1)x64
model.add(Conv2D(32, (2,1), activation='relu'))                                         # 출력 : (2, 1, 32)  param 4128 = (2x1x64+1)x32  <- 가로가 1칸이라 (2,1)
model.add(Flatten())                                                                    # 출력 : (64,)  2x1x32
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))                                                                 # 훈련 때마다 20% 를 끈다 (평가 / 예측 때는 전부 사용)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))                                                  # 회귀 -> 출력 1개, 대여 수는 음수가 없어서 relu

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,                # val_loss 가 20 epoch 동안 안 좋아지면 멈춘다
    restore_best_weights=True,  # 멈춘 뒤 val_loss 가 가장 낮았던 가중치로 되돌린다
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',                # val_loss 는 낮을수록 좋으므로 auto(=min)
    save_best_only=True,        # 최고 기록이 갱신될 때만 덮어쓴다 -> 마지막에 남는 파일 = 최고 epoch 모델
    filepath=path_save + 'keras34_mcp5.keras',
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

#4. 평가 예측 (훈련에도 검증에도 안 쓴 x_test 로만)
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)                # 1 에 가까울수록 좋다
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)     # loss 가 mse 라서 위 loss 와 거의 같은 값
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  # mse 에 루트 -> '대여 수' 단위로 오차를 본다

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