# keras35_gpu_test02_diabetes.py
# keras34 와 코드는 그대로 두고 CPU / GPU 로 각각 실행해서 소요 시간을 비교하는 파일
# 파일 아래쪽에 CPU 기록 / GPU 기록을 따로 남긴다
# 데이터가 작으면 CPU 가 더 빠를 수 있다 (GPU 로 데이터를 보내는 시간이 계산 시간보다 크기 때문)
# --- 아래는 원본(keras34_hamsu02_diabetes.py) 의 설명 ---
# 구조 : keras34_hamsu00.py (함수형 모델) 기준 / 데이터, 모델 : keras33_dropout02_diabetes.py 를 함수형으로 변환
# 함수형 모델 : Input 으로 입력층을 만들고, '층(이전 층)' 형태로 하나씩 연결한 뒤 Model(inputs, outputs) 로 범위를 정한다
#               층 구성이 같으면 Sequential 과 파라미터 수도 같은 똑같은 모델이다 (만드는 방법만 다르다)
#               단, 같은 것은 '구조(Total params)' 다. 가중치 초기값 / Dropout 이 끄는 노드 / 배치 섞기가 매번 랜덤이라
#               시드를 고정하지 않으면 같은 코드를 두 번 돌려도 loss, acc 는 조금씩 다르게 나온다
#               (random_state 는 train_test_split 의 데이터 분할만 고정한다)
# Dropout : 훈련할 때마다 층 출력의 일부 노드를 랜덤으로 꺼서 특정 노드에만 의존하지 않게 한다 -> 과적합 방지
#           evaluate / predict 때는 자동으로 꺼지고 모든 노드를 다 사용한다

import numpy as np
import time

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Model                   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input   # Input -> 함수형 모델의 입력층
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras34/'   # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

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
# 지우면 훈련 데이터 양이 달라져서 keras31 / keras33 과 비교할 수 없으므로 그대로 둔다.
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

#2. 모델 구성 (함수형) - keras33 과 층 / Dropout 위치가 똑같다
input1 = Input(shape=(10,))                     # 입력층 : 컬럼 10개 (Sequential 의 input_dim=10)
dense1 = Dense(128, activation='relu')(input1)  # (input1) -> input1 뒤에 연결
drop1 = Dropout(0.2)(dense1)                    # dense1 출력의 20% 를 훈련 때마다 끈다

dense2 = Dense(64, activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)

dense3 = Dense(32, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)

output1 = Dense(1)(drop3)                       # 출력층 : 회귀라 activation 없음

model = Model(inputs=input1, outputs=output1)   # 시작(input1) ~ 끝(output1) 범위를 정해서 모델 완성
model.summary()                                 # keras33 Sequential 과 Total params 가 같아야 제대로 변환한 것

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=20,                # val_loss 가 20 epoch 동안 안 좋아지면 멈춘다
#     restore_best_weights=True,  # 멈춘 뒤 val_loss 가 가장 낮았던 가중치로 되돌린다
#     verbose=1,
# )

mcp = ModelCheckpoint(
    monitor='val_loss',         # es 와 같은 기준으로 '최고 epoch' 를 고른다
    mode='auto',                # val_loss 는 낮을수록 좋으므로 auto(=min)
    save_best_only=True,        # 최고 기록이 갱신될 때만 덮어쓴다 -> 마지막에 남는 파일 = 최고 epoch 모델
    filepath=path + 'keras34_mcp2.keras',   # 파일 번호 규칙 : 02 -> mcp2
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=100,
                 batch_size=32,
                 validation_split=0.3,
                 callbacks=[mcp],   # callbacks 에 넣어야 실제로 동작한다
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
    return np.sqrt(mean_squared_error(y_test, y_predict))  # mse 에 루트 -> y 와 같은 단위로 오차를 본다

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