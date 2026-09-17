import numpy as np
import time

from tensorflow.keras.datasets import boston_housing   # 보스턴 집값 데이터 (회귀)
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras34/'   # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()  # train / test 가 이미 나뉘어 있다
print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

# Conv2D 는 (행, 열, 채널) 4차원 입력이 필요하다 -> 컬럼 13개를 13 x 1 x 1 로 바꾼다
# 13 은 소수라 직사각형으로 못 나눈다 -> 세로로 세우고 커널을 (2,1) 로 쓴다
x_train = x_train.reshape(-1, 13, 1, 1)  # (404, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)    # (102, 13, 1, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,1), padding='same', activation='relu', input_shape=(13, 1, 1)))  # 출력 : (13, 1, 64)  param 192 = (2x1x1+1)x64
model.add(Conv2D(64, (2,1), activation='relu'))                                          # 출력 : (12, 1, 64)  param 8256 = (2x1x64+1)x64
model.add(Conv2D(32, (2,1), activation='relu'))                                          # 출력 : (11, 1, 32)  param 4128 = (2x1x64+1)x32
model.add(Flatten())                                                                     # 출력 : (352,)  11x1x32
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))                                                                      # 회귀 -> 출력 1개, 활성화 함수 없음(linear)

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
    filepath=path + 'keras34_mcp3.keras',
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

#4. 평가 예측 (훈련에도 검증에도 안 쓴 x_test 로만)
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)                # 1 에 가까울수록 좋다
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)     # loss 가 mse 라서 위 loss 와 거의 같은 값
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  # mse 에 루트 -> y(집값, 천 달러)와 같은 단위로 오차를 본다

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# ===== CPU 기록 =====
# 소요 시간 : 7.96 초
# loss : 24.035165786743164
# r2 : 0.7112679690358323
# mse : 24.0351654483954
# RMSE : 4.902567230379956

# ===== GPU 기록 =====
# 소요 시간 : 5.92 초
# loss : 20.36505889892578
# r2 : 0.7553565692056211
# mse : 20.365060694411476
# RMSE : 4.512766412569066

# ===== GPU 기록 ===== <- Conv2D 적용
# 소요 시간 : 8.37 초
# loss : 18.92754554748535
# r2 : 0.7726252752728436
# mse : 18.927547142418646
# RMSE : 4.35058009263347