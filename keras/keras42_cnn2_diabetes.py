import numpy as np
import time

from sklearn.datasets import load_diabetes   # 당뇨병 진행도 데이터 (회귀)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras34/'   # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

# 1단계 : 전체 -> train(70%) / test(30%)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=77,
    shuffle=True,
)

# 2단계 : train -> train(70%) / val(30%)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.7,
    random_state=77,
    shuffle=True,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)
x_val = scaler.transform(x_val)         # val 도 transform 만 (빠뜨리면 검증에만 원본 단위가 들어간다)

# Conv2D 는 (행, 열, 채널) 4차원 입력이 필요하다 -> 컬럼 10개를 5 x 2 x 1 로 바꾼다
x_train = x_train.reshape(-1, 5, 2, 1)  # (216, 5, 2, 1)
x_val = x_val.reshape(-1, 5, 2, 1)      # (93, 5, 2, 1)
x_test = x_test.reshape(-1, 5, 2, 1)    # (133, 5, 2, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', activation='relu', input_shape=(5, 2, 1)))  # 출력 : (5, 2, 64)  param 320 = (2x2x1+1)x64
model.add(Conv2D(64, (2,2), activation='relu'))                                         # 출력 : (4, 1, 64)  param 16448 = (2x2x64+1)x64
model.add(Conv2D(32, (2,1), activation='relu'))                                         # 출력 : (3, 1, 32)  param 4128 = (2x1x64+1)x32  <- 가로가 1칸이라 (2,1)
model.add(Flatten())                                                                    # 출력 : (96,)  3x1x32
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))                                                                     # 회귀 -> 출력 1개, 활성화 함수 없음(linear)

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
    filepath=path + 'keras34_mcp2.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=100,
                 batch_size=32,
                 validation_data=(x_val, y_val),    # 직접 나눈 val 세트로 val_loss 계산
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
    return np.sqrt(mean_squared_error(y_test, y_predict))  # mse 에 루트 -> y(진행도 점수)와 같은 단위로 오차를 본다

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# ===== CPU 기록 =====
# 소요 시간 : 8.13 초
# loss : 3353.86865234375
# r2 : 0.4153950162220976
# mse : 3353.868579771466
# RMSE : 57.91259431049058

# ===== GPU 기록 =====
# 소요 시간 : 6.36 초
# loss : 3307.038330078125
# r2 : 0.4235579227575228
# mse : 3307.0381275713275
# RMSE : 57.50685287486464

# ===== GPU 기록 ===== <- Conv2D 적용
# 소요 시간 : 8.1 초
# loss : 3156.532470703125
# r2 : 0.4497921663401562
# mse : 3156.5327304102047
# RMSE : 56.1830288468876