# keras28_Scaler01_california.py
# keras27_Scaler01_california.py 베이스 (스케일링 순서를 바로잡은 기준 파일)
# 스케일러 4종 비교 (MinMax / Standard / MaxAbs / Robust) - 파일 아래쪽에 스케일러별 결과를 기록해 둔다
# 순서가 중요하다 : train_test_split 을 먼저 하고 -> scaler.fit(x_train) -> x_test 는 transform 만
#  fit 은 변환 기준(Min/Max, 평균, 중앙값 등)을 구하는 단계라 x_train 으로만 해야 한다
#  x_test 로 fit 하면 아직 보면 안 되는 평가 데이터의 정보가 기준에 섞인다 (데이터 누수)

from sklearn.datasets import fetch_california_housing

from tensorflow.keras.models import Sequential
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
# (스케일러를 MinMaxScaler 로 되돌리면 다시 Min 0.0 / Max 1.0 근처가 나온다)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(50))
model.add(Dense(1))

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

# loss : 0.63498455286026
# loss : 0.6710934638977051

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# loss : 0.503523051738739
# loss : 0.5084022879600525

# [결론] 0.635 -> 0.504 (약 20% 개선). california 는 컬럼별 값 범위 차이가 커서 스케일링 효과가 크다.

# ========== ========== ========== ========== ========== <- StandardScaler 적용 후

# loss : 0.5425562858581543

# ========== ========== ========== ========== ========== <- MaxAbsScaler 적용 후

# loss : 0.5589894652366638

# ========== ========== ========== ========== ========== <- RobustScaler 적용 후

# loss : 0.6695248484611511
# loss : 1.0735976696014404  <- 같은 코드 재실행 (2026-09-11)

# [주의] 같은 코드인데 0.6695 / 1.0736 으로 크게 벌어진다.
#        가중치 초기값 시드를 고정하지 않아서 실행할 때마다 생기는 편차인데,
#        이 편차가 스케일러 간 차이(0.504 ~ 0.670)보다 커서
#        한 번씩만 돌린 결과로 "어느 스케일러가 제일 좋다"를 단정할 수 없다.
#        제대로 비교하려면 시드를 고정하거나, 스케일러마다 여러 번 돌려 평균을 봐야 한다.