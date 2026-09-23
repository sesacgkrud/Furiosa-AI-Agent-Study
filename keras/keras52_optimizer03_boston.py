# keras52_optimizer03_boston.py
# 보스턴 집값 (회귀) - optimizer 의 learning_rate 를 직접 지정해 본다
# 스케일러는 keras28 에서 비교해 고른 RobustScaler 를 그대로 쓴다
#
# [ 오늘 바뀌는 곳은 compile 한 줄이다 ]
#  before : model.compile(..., optimizer='adam')                  -> learning_rate 가 기본값 0.001 로 고정
#  after  : model.compile(..., optimizer=Adam(learning_rate=...))  -> 보폭을 내가 정한다
#
# [ learning_rate = 가중치를 한 번에 얼마나 크게 고칠지 정하는 보폭 ]
#  크면 (0.05 / 0.01)  : 최저점을 건너뛰고 그 주변에서 튕겨 다닌다
#  작으면 (0.00001)    : 한 걸음이 작아서 epoch 가 끝날 때까지 최저점에 닿지 못한다
#  데이터마다 맞는 값이 달라서 후보를 주석으로 적어 두고 하나씩 열어 가며 직접 비교했다
#  이 파일에서 고른 값 : learning_rate = 0.01
#
# keras28_Scaler03_boston.py 베이스

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
import numpy as np

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0000000000000002
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : -0.0019120458891013214 Max : 1.1478180091225068

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=13, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
)
hist = model.fit(
    x_train, y_train,
    epochs=500,
    batch_size=16,
    validation_split=0.3,
    callbacks=[es],
)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
# 평가는 훈련에도 검증에도 쓰지 않은 x_test 로만 한다.
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

print("========================= history =========================")
print(hist)
print("========================= history =========================")
print(hist.history)
print("========================= loss =========================")
print(hist.history['loss'])
print("========================= val_loss =========================")
print(hist.history['val_loss'])
print("========================= The End =========================")


import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic' # 글자 깨짐 해결

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][10:], c='red', label='loss') # y값만 넣으면 시간 순으로 그림
plt.plot(hist.history['val_loss'][10:], c='blue', label='val_loss')

plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title('Boston Loss')

plt.xlabel('epochs')
plt.xlabel('loss')

plt.grid() # 격자 표시 추가
plt.show()

# loss: 10.2613 - val_loss: 23.3625
# loss: 22.8219

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# loss: 4.7181 - val_loss: 13.1078
# loss : 18.74143409729004

# [결론] 22.8 -> 18.7 (약 18% 개선).
#        boston 은 CRIM(0~89), TAX(187~711), B(0~396) 처럼 컬럼 범위가 제각각이라 효과가 있다.

# ========== ========== ========== ========== ========== <- StandardScaler 적용 후

# loss : 20.864587783813477

# ========== ========== ========== ========== ========== <- MaxAbsScaler 적용 후

# loss : 20.24062728881836

# ========== ========== ========== ========== ========== <- RobustScaler 적용 후

# loss : 21.350570678710938

# ========== ========== ========== ========== ========== <- learning rate 적용 후

# loss : 19.099597930908203

# [결론] RobustScaler 21.35 -> learning_rate 0.01 로 19.10. 약 10% 좋아졌다.
#        boston 은 404개로 작아서 기본값 0.001 로는 덜 내려간 상태에서 es 가 걸렸던 것이다.
