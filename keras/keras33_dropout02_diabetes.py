# keras33_dropout02_diabetes.py
# 구조 : keras33_dropout01_california.py 기준 / 데이터, 모델 : keras31_MCP_save_02_diabetes.py + Dropout 적용
# Dropout : 훈련할 때마다 층 출력의 일부 노드를 랜덤으로 꺼서(0 으로 만들어서) 특정 노드에만 의존하지 않게 한다 -> 과적합 방지
#           evaluate / predict 때는 자동으로 꺼지고 모든 노드를 다 사용한다

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras33/'   # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

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

# x_val 은 fit 에서 쓰지 않지만(validation_split 사용), 이 split 으로 x_train 이 49% 로 줄어든다.
# 지우면 훈련 데이터 양이 달라져서 Dropout 적용 전(keras31_02)과 비교할 수 없으므로 그대로 둔다.
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.7,
    random_state=77,
)

scaler = RobustScaler()                 # keras28 에서 4종 비교 후 RobustScaler 로 통일
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

print('Min :', np.min(x_train), 'Max :', np.max(x_train))
print('Min :', np.min(x_test), 'Max :', np.max(x_test))

#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dropout(0.2))     # 바로 앞 층 출력의 20% 를 훈련 때마다 랜덤으로 끈다

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1))         # 출력층 : 회귀라 activation 없음, Dropout 도 넣지 않는다
model.summary()             # Dropout 층은 파라미터 0 개 (가중치 없이 끄기만 한다)

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
    filepath=path + 'keras33_mcp2.keras',   # 파일 번호 규칙 : 02 -> mcp2
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

# ===== 이전 기록 =====
# Dropout 적용 전 (keras31_MCP_save_02)  loss : 3291.81494140625 / r2 : 0.42621144855681947 / RMSE : 57.374340052098496

# ===== 실행 결과 (2026-09-14, Dropout + MCP + r2/mse/rmse 추가 후) =====
# loss : 3112.50927734375
# r2 : 0.45746572147891085
# mse : 3112.509678624538
# RMSE : 55.78987075289329
# (Epoch 96: early stopping / Restoring model weights from the end of the best epoch: 76.)
