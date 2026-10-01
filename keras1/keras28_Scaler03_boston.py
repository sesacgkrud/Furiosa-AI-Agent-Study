# keras28_Scaler03_boston.py
# 보스턴 집값 (회귀)
# 스케일러 4종 비교 (MinMax / Standard / MaxAbs / Robust) - 파일 아래쪽에 스케일러별 결과를 기록해 둔다
# 순서가 중요하다 : train_test_split 을 먼저 하고 -> scaler.fit(x_train) -> x_test 는 transform 만
#  fit 은 변환 기준(Min/Max, 평균, 중앙값 등)을 구하는 단계라 x_train 으로만 해야 한다
#  x_test 로 fit 하면 아직 보면 안 되는 평가 데이터의 정보가 기준에 섞인다 (데이터 누수)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
import numpy as np

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

# scaler.fit(x_train) # sklearn 에서 fit -> 실행하다 로 생각
# x_train = scaler.transform(x_train) # x에 있는 모든 데이터는 0~1 사이로 수렴
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
model.compile(loss='mse', optimizer='adam')

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