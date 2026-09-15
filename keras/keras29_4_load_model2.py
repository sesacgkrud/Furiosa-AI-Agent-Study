# keras29_4_load_model2.py
# keras29_3 이 저장한 "훈련까지 끝난 모델" 을 불러온다
# 모델 구성도 컴파일도 fit 도 필요 없이 load_model 한 줄이면 바로 평가할 수 있다
# -> 저장했을 때와 loss 가 소수점까지 같으면 제대로 저장/복원된 것이다

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
# model = Sequential()
# model.add(Dense(100, input_dim=8))
# model.add(Dense(50))
# model.add(Dense(1))

# model.summary()

########## [추가] 훈련까지 끝난 모델 불러오기 ##########
# keras29_3 이 fit 아래에서 저장한 파일이라 가중치가 들어있다 -> 훈련을 다시 할 필요가 없다.
# 그래서 아래 #3. 컴파일, 훈련 이 통째로 주석 처리되어 있다.
path = './_save/keras29/'
# model.save(path + 'keras29_1_save_model.h5') # 예전 방식
# model.save(path + 'keras29_1_save_model.keras') # 최근 방식
model = load_model(path + 'keras29_3_save_model.keras')

model.summary() # Total params 18,005 / Optimizer params 12,004
                # keras29_1(6,001)보다 큰 이유 - 가중치 외에 옵티마이저 상태까지 저장됐다

# exit()

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=20, # 참고 기다리는 횟수
#     restore_best_weights=True, # 지나쳤더라도 최솟값 반환
# )

# hist = model.fit(x_train, y_train,
#                  epochs=10000,
#                  batch_size=32,
#                  validation_split=0.2,
#                  callbacks=[es],
#                  )

# model.save(path + 'keras29_3_save_model.keras') # 최근 방식

print("========== ========== ========== ========== ==========")

# model.compile() 을 안 했는데도 평가가 되는 이유
# -> .keras 파일에는 가중치뿐 아니라 컴파일 정보(loss='mse', optimizer='adam')까지 저장돼 있다
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

#4. 평가 예측
# 평가는 훈련에도 검증에도 안 쓴 x_test 로만 한다 -> 위의 evaluate 하나면 충분하다
#  1) x 는 스케일링을 안 한 원본이다. 모델은 변환된 값으로 학습했는데
#     평가에만 원본(MedInc 0~15, Population 수천)을 넣으면 loss 가 89923 처럼 커진다
#     -> 모델이 나쁜 게 아니라 자와 대상의 단위가 안 맞는 것이라 성능 비교와 무관한 숫자다
#  2) x 안에는 x_train 이 그대로 들어있다 (시험지에 답안지가 섞인 상태)
#  전체 데이터로 굳이 확인하고 싶다면 scaler.transform(x) 로 같은 기준으로 변환해서 넣는다

# print("========================= history =========================")
# print(hist)
# print("========================= history =========================")
# print(hist.history)
# print("========================= loss =========================")
# print(hist.history['loss'])
# print("========================= val_loss =========================")
# print(hist.history['val_loss'])
# print("========================= The End =========================")


# [주의] 아래 그래프 블록은 이 파일에서는 동작하지 않는다.
#        훈련을 안 해서 hist 가 없고 plot 이 전부 주석이라 빈 창만 뜨면서
#        UserWarning: No artists with labels found to put in legend 경고가 출력된다.
#        로드 전용 파일에는 필요 없으니 통째로 지워도 된다. (일단 그대로 둠)
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
# 바로 앞서 실행한 keras29_3 의 loss 와 소수점 끝까지 같다.
# evaluate 는 학습이 아니라 계산이라 무작위 요소가 없고,
# random_state=100 으로 분할 고정 + 스케일러 변환 고정 + 가중치는 파일에 저장된 값이라
# 입력이 같으면 출력도 항상 같다. (두 번 돌려도 동일한 값)
#
# 스케일러별 loss 비교 기록은 keras28_Scaler01_california.py 에 있다
