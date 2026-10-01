# keras52_ReduceLR02_diabetes.py
# 당뇨 (회귀) - ReduceLROnPlateau 로 훈련 도중 learning_rate 를 줄인다
# 스케일러(RobustScaler)와 learning_rate 후보는 keras52_optimizer 파일 그대로다
#
# [ 오늘 추가되는 곳은 콜백 하나다 ]
#  keras52_optimizer 는 learning_rate 를 처음부터 끝까지 고정했다
#  하지만 한 값으로 두 가지를 다 할 수는 없다
#   초반에는 크게 움직여 빨리 내려가야 하고, 최저점 근처에서는 작게 움직여야 지나치지 않는다
#
# [ ReduceLROnPlateau : 훈련 도중 learning_rate 를 줄여 준다 ]
#  plateau = 고원. val_loss 가 더 이상 줄지 않고 평평해지는 구간을 말한다
#  monitor 가 patience 동안 좋아지지 않으면 learning_rate 에 factor 를 곱한다
#   factor=0.5 -> 0.0001 -> 절반 -> 또 절반 ... 으로 떨어진다
#  EarlyStopping 과 역할이 다르다 : es 는 '멈춘다', rlr 은 '보폭을 줄여 더 해 본다'
#   그래서 rlr 의 patience 를 es 보다 짧게 줘야 "줄여 보고 -> 그래도 안 되면 멈춘다" 순서가 된다
#  이 파일 : 시작 learning_rate = 0.0001 / es patience = 20 / rlr patience = 20
#   -> 둘이 같아서 lr 이 줄어드는 바로 그 epoch 에 es 도 같이 걸린다
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
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009
learning_rate = 0.0001

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
)

# val_loss 가 patience 동안 좋아지지 않으면 learning_rate 에 factor 를 곱해 줄인다
# verbose=1 -> 줄어드는 순간 'ReduceLROnPlateau reducing learning rate to ...' 가 찍힌다
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,
)

hist = model.fit(x_train, y_train,
                 epochs=500,
                 batch_size=16,
                 validation_split=0.3,
                 callbacks=[es, rlr],
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

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후

# loss : 3296.05859375

# [결론] 0.005 고정 3203 -> 시작 0.0001 + ReduceLROnPlateau 3296. 오히려 살짝 나빠졌다.
#        시작값을 0.0001 로 너무 작게 잡아서 줄일 여지가 거의 없었다.
#        es 와 rlr 의 patience 가 둘 다 20 이라 lr 이 줄어드는 순간 훈련도 같이 끝난다.
