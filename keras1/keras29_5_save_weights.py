# keras29_5_save_weights.py
# save_weights() : 가중치(숫자)만 저장한다. 모델 구조는 저장되지 않는다
# 확장자는 .weights.h5 로 끝나야 한다
# fit 앞에서 저장하면 초기 가중치(save1), fit 뒤에서 저장하면 훈련된 가중치(save2)가 된다

# keras29_3_save_model2.py 베이스

from sklearn.datasets import fetch_california_housing

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

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

# RobustScaler = (원값 - 중앙값) / IQR 이라서 MinMaxScaler 처럼 0~1 로 수렴하지 않고 이상치가 그대로 남는다
print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : -7.2267 Max : 1455.8382
print('Min :', np.min(x_test), 'Max :', np.max(x_test))   # Min : -7.6732 Max : 586.3726
# 최대 이상치(1455.8)가 x_test(586.4) 쪽이 아니라 x_train 쪽에 있다
# -> 그래서 이 조합은 훈련 loss 가 평가 loss 보다 높게 나오기도 한다

# #2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(50))
model.add(Dense(1))

# model.summary()

########## [추가] 저장 위치를 훈련 '아래'로 옮긴다 ##########
# keras29_1 은 여기서 save 를 했지만, 이 파일은 fit 아래에서 저장한다. (아래 save 주석 참고)
path = './_save/keras29/'
# model.save(path + 'keras29_1_save_model.h5') # 예전 방식
model.save_weights(path + 'keras29_5_save1.weights.h5') # 최근 방식
# model = load_model(path + 'keras29_1_save_model.keras')

model.summary() # 아직 훈련 전이라 Total params 6,001 (가중치만)

# exit()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20, # 참고 기다리는 횟수
    restore_best_weights=True, # 지나쳤더라도 최솟값 반환
)

hist = model.fit(x_train, y_train,
                 epochs=10000,
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[es],
                 )

########## [추가] 훈련이 끝난 뒤에 저장 ##########
# keras29_1 과 달리 save 가 fit '아래'에 있다.
# 그래서 구조뿐 아니라 훈련된 가중치 + 옵티마이저 상태까지 저장된다. (6,001 -> 18,005)
# EarlyStopping 의 restore_best_weights=True 로 되돌린 '최적 가중치'가 저장되는 것이다.
# model.save(path + 'keras29_3_save_model.keras')
model.save_weights(path + 'keras29_5_save2.weights.h5')

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

#4. 평가 예측
# 평가는 훈련에도 검증에도 안 쓴 x_test 로만 한다 -> 위의 evaluate 하나면 충분하다
#  1) x 는 스케일링을 안 한 원본이다. 모델은 변환된 값으로 학습했는데
#     평가에만 원본(MedInc 0~15, Population 수천)을 넣으면 loss 가 89923 처럼 커진다
#     -> 모델이 나쁜 게 아니라 자와 대상의 단위가 안 맞는 것이라 성능 비교와 무관한 숫자다
#  2) x 안에는 x_train 이 그대로 들어있다 (시험지에 답안지가 섞인 상태)
#  전체 데이터로 굳이 확인하고 싶다면 scaler.transform(x) 로 같은 기준으로 변환해서 넣는다

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
plt.plot(hist.history['loss'], c='red', label='loss') # y값만 넣으면 시간 순으로 그림
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')

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
# 스케일러별 loss 비교 기록은 keras28_Scaler01_california.py 에 있다

# ========== ========== ========== ========== ========== <- weights.h5 적용 후

# loss: 3.6324
# loss : 3.632439374923706