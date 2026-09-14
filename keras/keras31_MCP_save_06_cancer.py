# keras31_MCP_save_06_cancer.py
# 구조 : keras30_ModelCheckPoint1.py 기준 / 데이터, 모델 : keras28_Scaler06_cancer.py
# 짝 파일 : keras32_MCP_load_06_cancer.py
#   이 파일이 저장한 keras31_mcp6.keras 를 keras32 가 불러와서 평가하면 아래 결과와 소수점까지 같아야 한다.
#   EarlyStopping 과 ModelCheckpoint 가 둘 다 val_loss 가 가장 낮은 epoch 를 고르고,
#   restore_best_weights=True 가 훈련이 끝날 때 그 가중치로 되돌리므로 '훈련 끝난 model' = '저장된 파일' 이다.

import numpy as np
from sklearn.datasets import load_breast_cancer   # 유방암 데이터 (이진 분류 : 0 악성 / 1 양성)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras31/'   # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42,
    stratify=y,     # 0 / 1 비율을 train 과 test 에 똑같이 나눈다
)

scaler = RobustScaler()                 # keras28 에서 4종 비교 후 이번 실습은 RobustScaler 로 통일
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

# keras32 에서도 같은 값이 찍혀야 #1 전처리가 똑같다는 확인이 된다
print('Min :', np.min(x_train), 'Max :', np.max(x_train))
print('Min :', np.min(x_test), 'Max :', np.max(x_test))

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=30, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 이진 분류 출력층은 sigmoid -> 0~1 사이 확률

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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
    filepath=path + 'keras31_mcp6.keras',   # 파일 번호 규칙 : 06 -> mcp6
    verbose=1,
)

hist = model.fit(x_train, y_train,
                 epochs=500,
                 batch_size=32,
                 validation_split=0.3,
                 callbacks=[es, mcp],   # callbacks 에 넣어야 실제로 동작한다 (mcp 가 빠지면 파일이 저장되지 않는다)
                 verbose=1,
                 )

print("========== ========== ========== ========== ==========")

#4. 평가 예측 (훈련에도 검증에도 안 쓴 x_test 로만)
loss = model.evaluate(x_test, y_test)   # [loss, acc] 리스트로 나온다
print("loss :", loss[0])
print("acc :", round(loss[1], 4))

y_predict = model.predict(x_test)       # sigmoid 확률 (0~1)

# 분류 성능은 acc 로 판단한다. r2 / mse / rmse 는 '확률 vs 정답(0, 1)' 의 오차로 참고용이다.
# acc 는 반올림 후 맞힌 개수라 둔감하지만, 이 값들은 확률 그대로라 save / load 가 소수점까지 같은지 확인하기 좋다.
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

y_predict = np.round(y_predict)         # 0.5 기준 반올림 -> 0 또는 1 (안 하면 accuracy_score 가 ValueError)
acc_score = accuracy_score(y_test, y_predict)
print("acc_score :", acc_score)

# ===== 실행 결과 (2026-09-14, keras31 실행 직후 keras32 실행) =====
# loss : 0.09741392731666565
# acc : 0.9649
# r2 : 0.8792529106140137
# mse : 0.02827797643840313
# RMSE : 0.16816056743007002
# acc_score : 0.9649122807017544
# -> keras32 (load) 실행 결과와 소수점까지 같다. (Min / Max 출력도 같음 = #1 전처리 동일)
# keras31 은 가중치 초기값을 고정하지 않아서 다시 실행하면 값이 달라지고 .keras 파일도 덮어써진다.
# 그러니 비교할 때는 항상 keras31 을 실행한 "직후" 에 keras32 를 실행할 것.
