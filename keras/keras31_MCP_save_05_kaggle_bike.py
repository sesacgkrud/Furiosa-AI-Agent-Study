# keras31_MCP_save_05_kaggle_bike.py
# https://www.kaggle.com/competitions/bike-sharing-demand/data
# 구조 : keras30_ModelCheckPoint1.py 기준 / 데이터, 모델 : keras28_Scaler05_kaggle_bike.py
# 짝 파일 : keras32_MCP_load_05_kaggle_bike.py
#   이 파일이 저장한 keras31_mcp5.keras 를 keras32 가 불러와서 평가하면 아래 결과와 소수점까지 같아야 한다.
#   EarlyStopping 과 ModelCheckpoint 가 둘 다 val_loss 가 가장 낮은 epoch 를 고르고,
#   restore_best_weights=True 가 훈련이 끝날 때 그 가중치로 되돌리므로 '훈련 끝난 model' = '저장된 파일' 이다.
# [방법 2] validation_split 으로 fit 이 x_train 에서 알아서 val 을 떼어가게 하는 방식

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = "./_data/kaggle_bike/"   # 데이터 폴더
path_save = './_save/keras31/'  # 모델 저장 폴더

#1. 데이터
train_csv = pd.read_csv(path + "train.csv", index_col=0)   # [10886 rows x 11 columns] / 제출을 안 하므로 test.csv 는 읽지 않는다

# casual + registered = count 라서 입력에 넣으면 정답을 알려주는 셈 -> 같이 뺀다
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)  # [10886 rows x 8 columns]
y = train_csv['count']
print(x.shape, y.shape) # (10886, 8) (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=100,
)

scaler = RobustScaler()                 # keras28 에서 4종 비교 후 이번 실습은 RobustScaler 로 통일
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

# keras32 에서도 같은 값이 찍혀야 #1 전처리가 똑같다는 확인이 된다
print('Min :', np.min(x_train), 'Max :', np.max(x_train))
print('Min :', np.min(x_test), 'Max :', np.max(x_test))

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))  # keras28 그대로. 출력층 relu 는 음수를 0 으로 잘라서 한번 0 에 갇히면 학습이 멈출 수 있다(dying ReLU)

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
    monitor='val_loss',         # es 와 같은 기준으로 '최고 epoch' 를 고른다
    mode='auto',                # val_loss 는 낮을수록 좋으므로 auto(=min)
    save_best_only=True,        # 최고 기록이 갱신될 때만 덮어쓴다 -> 마지막에 남는 파일 = 최고 epoch 모델
    filepath=path_save + 'keras31_mcp5.keras',  # 파일 번호 규칙 : 05 -> mcp5 (다른 파일을 덮어쓰지 않게)
    verbose=1,
)

hist = model.fit(x_train, y_train,
                 epochs=500,
                 batch_size=16,
                 validation_split=0.2,  # [방법 2] x_train 의 뒤쪽 20% 를 검증용으로 떼어 쓴다
                 callbacks=[es, mcp],   # callbacks 에 넣어야 실제로 동작한다
                 verbose=1,
                 )

print("========== ========== ========== ========== ==========")

#4. 평가 예측 (훈련에도 검증에도 안 쓴 x_test 로만)
loss = model.evaluate(x_test, y_test)
print("loss(mse) :", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)                # 1 에 가까울수록 좋다
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)     # loss 가 mse 라서 위 loss 와 거의 같은 값
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  # mse 에 루트 -> '대여 수' 단위로 오차를 본다

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# ===== 실행 결과 (2026-09-14, keras31 실행 직후 keras32 실행) =====
# loss(mse) : 21296.765625
# r2 : 0.31795167922973633
# mse : 21296.76953125
# RMSE : 145.93412737002268
# -> keras32 (load) 실행 결과와 소수점까지 같다. (Min / Max 출력도 같음 = #1 전처리 동일)
# keras31 은 가중치 초기값을 고정하지 않아서 다시 실행하면 값이 달라지고 .keras 파일도 덮어써진다.
# 그러니 비교할 때는 항상 keras31 을 실행한 "직후" 에 keras32 를 실행할 것.
