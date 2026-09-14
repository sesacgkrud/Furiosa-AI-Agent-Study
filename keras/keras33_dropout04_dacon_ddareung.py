# keras33_dropout04_dacon_ddareung.py
# https://dacon.io/competitions/open/235576/overview/description
# 구조 : keras33_dropout01_california.py 기준 / 데이터, 모델 : keras31_MCP_save_04_dacon_ddareung.py + Dropout 적용
# Dropout : 훈련할 때마다 층 출력의 일부 노드를 랜덤으로 꺼서(0 으로 만들어서) 특정 노드에만 의존하지 않게 한다 -> 과적합 방지
#           evaluate / predict 때는 자동으로 꺼지고 모든 노드를 다 사용한다
# [방법 1] train_test_split 을 2번 써서 x_val 을 직접 만들고 validation_data 로 넘기는 방식

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = "./_data/ddareung/"      # 데이터 폴더
path_save = './_save/keras33/'  # 모델 저장 폴더

#1. 데이터
train_csv = pd.read_csv(path + "train.csv", index_col=0)   # 제출을 안 하므로 test.csv / submission.csv 는 읽지 않는다

train_csv = train_csv.dropna()  # 결측치가 있는 행 삭제

x = train_csv.drop(['count'], axis=1)   # 입력 9개 컬럼
y = train_csv['count']                  # 정답 = 대여 수

# 1단계 : 전체 -> train(80%) / test(20%)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=222,
)

# 2단계 : train -> train(80%) / val(20%)  (전체 기준 val = 16%)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.8,
    random_state=77,
)

scaler = RobustScaler()                 # keras28 에서 4종 비교 후 RobustScaler 로 통일
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)
x_val = scaler.transform(x_val)         # val 도 transform 만. 빠뜨리면 검증에만 원본 단위가 들어가 EarlyStopping 이 엉뚱하게 멈춘다

print('Min :', np.min(x_train), 'Max :', np.max(x_train))
print('Min :', np.min(x_test), 'Max :', np.max(x_test))

#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=9, activation='relu'))
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
    filepath=path_save + 'keras33_mcp4.keras',  # 파일 번호 규칙 : 04 -> mcp4
    verbose=1,
)

hist = model.fit(x_train, y_train,
                 epochs=1000,
                 batch_size=16,
                 validation_data=(x_val, y_val),    # [방법 1] 직접 만든 val 세트로 val_loss 계산
                 callbacks=[es, mcp],               # callbacks 에 넣어야 실제로 동작한다
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

# ===== 이전 기록 =====
# Dropout 적용 전 (keras31_MCP_save_04)  loss(mse) : 2351.59130859375 / r2 : 0.6807679617592108 / RMSE : 48.49320926571903

# ===== 실행 결과 (2026-09-14, Dropout + MCP + r2/mse/rmse 추가 후) =====
# loss(mse) : 2411.2744140625
# r2 : 0.6726659193650188
# mse : 2411.274241615284
# RMSE : 49.104727283788925
# (Epoch 153: early stopping / Restoring model weights from the end of the best epoch: 133.)
