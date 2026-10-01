# keras52_optimizer02_diabetes.py
# 당뇨 (회귀) - optimizer 의 learning_rate 를 직접 지정해 본다
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
#  이 파일에서 고른 값 : learning_rate = 0.005
#
# keras28_Scaler02_diabetes.py 베이스

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
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
learning_rate = 0.005
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

# ========== ========== ========== ========== ========== <- learning rate 적용 후

# loss : 3203.0986328125

# [결론] RobustScaler 3232 -> learning_rate 0.005 로 3203. 사실상 그대로다.
#        diabetes 는 442개뿐이라 이 정도 차이는 실행할 때마다 생기는 편차 범위다.
