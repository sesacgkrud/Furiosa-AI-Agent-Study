# keras31_MCP_save_02_diabetes.py
# 구조 : keras30_ModelCheckPoint1.py 기준 / 데이터, 모델 : keras28_Scaler02_diabetes.py
# 짝 파일 : keras32_MCP_load_02_diabetes.py
#   이 파일이 저장한 keras31_mcp2.keras 를 keras32 가 불러와서 평가하면 아래 결과와 소수점까지 같아야 한다.
#   EarlyStopping 과 ModelCheckpoint 가 둘 다 val_loss 가 가장 낮은 epoch 를 고르고,
#   restore_best_weights=True 가 훈련이 끝날 때 그 가중치로 되돌리므로 '훈련 끝난 model' = '저장된 파일' 이다.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras31/'   # 모델 저장 폴더 (keras/ 가 아니라 furiosa_study 폴더에서 실행하는 기준)

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=77,
)

# x_val 은 아래 fit 에서 쓰지 않는다. (validation_split 을 쓰기 때문)
# 그래도 지우면 안 된다 : 이 split 으로 x_train 이 전체의 49% 로 줄어들고, scaler 도 이 x_train 으로 fit 하기 때문이다.
# 지우면 훈련 데이터 양과 scaler 기준이 달라져서 keras28 과 비교할 수 없고, keras32 와도 값이 달라진다.
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.7,
    random_state=77,
)

scaler = RobustScaler()                 # keras28 에서 4종 비교 후 이번 실습은 RobustScaler 로 통일
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

# keras32 에서도 같은 값이 찍혀야 #1 전처리가 똑같다는 확인이 된다
print('Min :', np.min(x_train), 'Max :', np.max(x_train))
print('Min :', np.min(x_test), 'Max :', np.max(x_test))

#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

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
    filepath=path + 'keras31_mcp2.keras',   # keras32 가 불러와야 하므로 고정 이름 사용 (날짜/epoch 이름은 미리 알 수 없음)
    verbose=1,
)

hist = model.fit(x_train, y_train,
                 epochs=500,
                 batch_size=16,
                 validation_split=0.3,
                 callbacks=[es, mcp],   # callbacks 에 넣어야 실제로 동작한다
                 verbose=1,
                 )

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
    return np.sqrt(mean_squared_error(y_test, y_predict))  # mse 에 루트 -> y 와 같은 단위로 오차를 본다

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# ===== 실행 결과 (2026-09-14, keras31 실행 직후 keras32 실행) =====
# loss : 3291.81494140625
# r2 : 0.42621144855681947
# mse : 3291.8148964138336
# RMSE : 57.374340052098496
# -> keras32 (load) 실행 결과와 소수점까지 같다. (Min / Max 출력도 같음 = #1 전처리 동일)
# keras31 은 가중치 초기값을 고정하지 않아서 다시 실행하면 값이 달라지고 .keras 파일도 덮어써진다.
# 그러니 비교할 때는 항상 keras31 을 실행한 "직후" 에 keras32 를 실행할 것.
