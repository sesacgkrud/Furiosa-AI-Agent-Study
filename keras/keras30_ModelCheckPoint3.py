# keras30_ModelCheckPoint1.py 베이스

from sklearn.datasets import fetch_california_housing

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

path = './_save/keras30/'

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.75,
    random_state=100,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

# scaler.fit(x_train)                   # x_train 의 컬럼별 변환 기준(중앙값, IQR)을 찾아서 저장만 한다
# x_train = scaler.transform(x_train)   # 저장해둔 기준을 공식에 대입해 실제로 값을 바꾼다
x_train = scaler.fit_transform(x_train) # 위 두 줄(fit + transform)을 한 줄로

x_test = scaler.transform(x_test) # test 는 transform 만! fit 을 또 하면 평가 데이터 기준이 섞인다 (데이터 누수)

# [수정] 아래 주석은 MinMaxScaler 를 쓰던 시절의 값이라 지금 출력과 맞지 않아 실제 값으로 고쳤다.
#        RobustScaler = (원값 - 중앙값) / IQR 이라서 0~1 로 수렴하지 않고 이상치가 그대로 남는다.
print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : -7.2267 Max : 1455.8382
print('Min :', np.min(x_test), 'Max :', np.max(x_test))   # Min : -7.6732 Max : 586.3726
# 최대 이상치(1455.8)가 x_test(586.4) 쪽이 아니라 x_train 쪽에 있다
# -> 그래서 이 조합은 훈련 loss 가 평가 loss 보다 높게 나오기도 한다

# #2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # weights만 불러왔기 때문에 있어야 함

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30, # 참고 기다리는 횟수
    restore_best_weights=True, # 지나쳤더라도 최솟값 반환
    verbose=1,
)

#################### MCP SAVE 파일명 만들기 ####################
import datetime
date = datetime.datetime.now() # 현재 시간 반환
print(date) # 2026-09-14 11:40:57.930454
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime('%m%d_%H%M')
print(date) # 0914_1147 -> 생성된 날짜
print(type(date)) # <class 'str'>

path = './_save/keras30/'
filename = '{epoch:04d}_{val_loss:.4f}.keras'
filepath = ''.join([path, 'k30_', date, '_', filename])

## 내가 생각하는 파일명 예)
# './_save/keras30/' + 'k30_' + '0914_1147' + '530-0.001.keras'

#################### #################### ####################

mcp = ModelCheckpoint(
    monitor='val_loss', # loss도 상관 없음, 선택의 문제, 통상적으로 epoch 하면서 검증하니 val_loss 사용
    mode='auto', # 낮을수록 좋기 때문에 auto, min 선택
    save_best_only=True,
    # filepath=path + 'keras30_mcp1.keras',
    filepath=filepath,
    verbose=1,
)

hist = model.fit(x_train, y_train,      # 여기에 훈련시킨 성능이 담겨 있음
                 epochs=1000,
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[es, mcp],
                 verbose=1,
                 )

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)
#4. 평가 예측
# [수정] 원본에는 여기에 loss = model.evaluate(x, y) 가 있었는데 두 가지 문제로 삭제했다.
#        1) x 는 스케일링을 안 한 원본이다. 모델은 0~1 로 변환된 값으로 학습했는데
#           평가에만 원본(MedInc 0~15, Population 수천)을 넣으니 loss 가 89923 처럼 터졌다.
#           -> 모델이 나쁜 게 아니라 자와 대상의 단위가 안 맞은 것. 성능 비교와 무관한 숫자다.
#        2) x 안에는 x_train 이 그대로 들어있다. (시험지에 답안지가 섞인 상태)
#           평가는 훈련에도 검증에도 안 쓴 x_test 로만 해야 하므로 위의 evaluate 하나면 충분하다.
#        전체 데이터로 굳이 확인하고 싶다면 scaler.transform(x) 로 같은 기준으로 변환해서 넣어야 한다.

# print("========================= history =========================")
# print(hist)
# print("========================= history =========================")
# print(hist.history)
# print("========================= loss =========================")
# print(hist.history['loss'])
# print("========================= val_loss =========================")
# print(hist.history['val_loss'])
# print("========================= The End =========================")


import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic' # 글자 깨짐 해결

plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss') # y값만 넣으면 시간 순으로 그림
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')

plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title('California(캘리포니아) Loss')

plt.xlabel('epochs')
plt.xlabel('loss')

plt.grid() # 격자 표시 추가
plt.show()

# ===== 실행 결과 (2026-09-11) =====
# loss : 1.7271162271499634
# 가중치 초기값을 고정하지 않아서 돌릴 때마다 값이 달라진다. (관측값 1.1605 / 1.7271)
# 저장 파일도 마지막 실행 것으로 덮어써지므로,
# keras29_4 는 이 파일을 실행한 '직후'에 돌려야 같은 값이 나온다.
#
# [수정] 아래에 있던 스케일러별 loss 비교 기록은 keras28_Scaler01_california.py 에서
#        측정한 값이라 이 파일과 무관해서 지웠다. 스케일러 비교는 keras28 파일에 있다.

# ========== ========== ========== ========== ========== <- weights.h5 적용 후

# loss: 3.6324
# loss : 3.632439374923706

# ========== ========== ========== ========== ========== <- ModelCheckPoint 적용 후 (+model.fit -> verbose=1 적용)

# loss: 2.1273
# loss : 2.1273159980773926

# r2 추가
# mse 추가
# rmse 추가