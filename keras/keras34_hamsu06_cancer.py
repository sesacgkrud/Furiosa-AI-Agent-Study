# keras34_hamsu06_cancer.py
# 구조 : keras34_hamsu00.py (함수형 모델) 기준 / 데이터, 모델 : keras33_dropout06_cancer.py 를 함수형으로 변환
# 함수형 모델 : Input 으로 입력층을 만들고, '층(이전 층)' 형태로 하나씩 연결한 뒤 Model(inputs, outputs) 로 범위를 정한다
#               층 구성이 같으면 Sequential 과 파라미터 수도 같은 똑같은 모델이다 (만드는 방법만 다르다)
# Dropout : 훈련할 때마다 층 출력의 일부 노드를 랜덤으로 꺼서 특정 노드에만 의존하지 않게 한다 -> 과적합 방지
#           evaluate / predict 때는 자동으로 꺼지고 모든 노드를 다 사용한다

import numpy as np
from sklearn.datasets import load_breast_cancer   # 유방암 데이터 (이진 분류 : 0 악성 / 1 양성)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Model                   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input   # Input -> 함수형 모델의 입력층
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras34/'   # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

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

scaler = RobustScaler()                 # keras28 에서 4종 비교 후 RobustScaler 로 통일
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

print('Min :', np.min(x_train), 'Max :', np.max(x_train))
print('Min :', np.min(x_test), 'Max :', np.max(x_test))

#2. 모델 구성 (함수형) - keras33 과 층 / Dropout 위치 / 출력층 activation 이 똑같다
input1 = Input(shape=(30,))                     # 입력층 : 컬럼 30개 (Sequential 의 input_dim=30)
dense1 = Dense(32, activation='relu')(input1)   # (input1) -> input1 뒤에 연결
drop1 = Dropout(0.3)(dense1)                    # dense1 출력의 30% 를 훈련 때마다 끈다

dense2 = Dense(16, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)

# 변수 이름은 층마다 새로 붙인다 (dense3 / drop3)
# 원래는 dense2 / drop2 이름을 다시 쓰고 (drop1) 에 연결해서 16 층이 모델에서 빠져 있었다
dense3 = Dense(8, activation='relu')(drop2)     # 바로 앞 층인 drop2 에 연결해야 32 -> 16 -> 8 순서가 된다
drop3 = Dropout(0.3)(dense3)

# 이진 분류 출력층은 sigmoid 가 반드시 있어야 0~1 확률이 나온다
# (원래는 빠져 있어서 출력이 음수 / 1 초과로 나왔고, 반올림한 값이 0, 1 이 아니라 acc_score 가 0.33 으로 떨어졌다)
output1 = Dense(1, activation='sigmoid')(drop3)

model = Model(inputs=input1, outputs=output1)   # 시작(input1) ~ 끝(output1) 범위를 정해서 모델 완성
model.summary()                                 # keras33 Sequential 과 Total params 가 같아야 제대로 변환한 것

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
    filepath=path + 'keras34_mcp6.keras',   # 파일 번호 규칙 : 06 -> mcp6
    verbose=1,
)

hist = model.fit(x_train, y_train,
                 epochs=500,
                 batch_size=32,
                 validation_split=0.3,
                 callbacks=[es, mcp],   # callbacks 에 넣어야 실제로 동작한다
                 verbose=1,
                 )

print("========== ========== ========== ========== ==========")

#4. 평가 예측 (훈련에도 검증에도 안 쓴 x_test 로만)
loss = model.evaluate(x_test, y_test)   # [loss, acc] 리스트로 나온다
print("loss :", loss[0])
print("acc :", round(loss[1], 4))

y_predict = model.predict(x_test)       # sigmoid 확률 (0~1)

# 분류 성능은 acc 로 판단한다. r2 / mse / rmse 는 '확률 vs 정답(0, 1)' 의 오차로 참고용이다.
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

# ===== 이전 기록 =====
# Sequential + Dropout (keras33_dropout06, 2026-09-14)    loss : 0.0966392457485199 / acc_score : 0.9473684210526315
# 함수형 (이전 실행, sigmoid 누락 + 16 층 연결 빠진 상태)  loss : 0.19262878596782684 / acc : 0.883 / acc_score : 0.3333333333333333

# ===== 실행 결과 (2026-09-14, 함수형 + Dropout + MCP + r2/mse/rmse, keras33 과 같은 구성으로 수정 후) =====
# loss : 0.10246408730745316
# acc : 0.9532
# r2 : 0.860884964466095
# mse : 0.03257958963513374
# RMSE : 0.18049817072517313
# acc_score : 0.9532163742690059
# (Epoch 74: early stopping / Restoring model weights from the end of the best epoch: 54.)
