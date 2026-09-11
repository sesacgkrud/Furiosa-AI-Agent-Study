from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

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

# [수정] 이 x_val / y_val 은 아래 fit 에서 실제로 쓰지 않는다. (validation_split=0.3 으로 대체됨)
#        그런데 '안 쓰니까 지우자' 하면 안 된다.
#        이 split 때문에 x_train 이 전체의 0.7 x 0.7 = 49% 로 줄어드는데,
#        지우면 훈련 데이터 양 자체가 달라져서 베이스(keras20)와의 성능 비교가 성립하지 않는다.
#        즉 지금은 'val 을 만드는 줄'이 아니라 'x_train 을 한 번 더 자르는 줄'로 남아있는 것이다.
#        x_val 을 진짜로 쓰고 싶다면 keras28_Scaler04 처럼 스케일링까지 해서 validation_data 로 넘길 것.
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.7,
    random_state=77,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

# scaler.fit(x_train) # sklearn 에서 fit -> 실행하다 로 생각
# x_train = scaler.transform(x_train) # x에 있는 모든 데이터는 0~1 사이로 수렴
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : -0.051724137931034475 Max : 1.2758620689655171

#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
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
hist = model.fit(x_train, y_train,
                 epochs=500,
                 batch_size=16,
                 validation_split=0.3,
                 callbacks=[es],
)

#4. 평가 예측
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
plt.title('Diabetes Loss')

plt.xlabel('epochs')
plt.xlabel('loss')

plt.grid() # 격자 표시 추가
plt.show()

# loss: 2520.1035 - val_loss: 3677.6003
# loss : 2874.47216796875

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# loss: 2501.8716 - val_loss: 3561.1721
# loss : 2971.2080078125

# [결론] 2874 -> 2971 로 오히려 살짝 나빠졌다. 하지만 이건 실패가 아니다.
#        diabetes 는 sklearn 이 이미 각 컬럼을 정규화해서 배포하는 데이터셋이라
#        (평균 0, 컬럼 제곱합 1) 스케일링을 더 해도 얻을 게 없다.
#        '스케일이 이미 고른 데이터에는 효과가 없다'를 확인한 것 자체가 결과다.

# ========== ========== ========== ========== ========== <- StandardScaler 적용 후

# loss : 3279.496337890625

# ========== ========== ========== ========== ========== <- MaxAbsScaler 적용 후

# loss : 3041.59130859375

# ========== ========== ========== ========== ========== <- RobustScaler 적용 후

# loss : 3232.843505859375