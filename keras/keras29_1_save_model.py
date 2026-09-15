# keras29_1_save_model.py
# model.save() : 모델을 파일로 저장한다 (구조 + 가중치 + 컴파일 정보가 통째로 들어간다)
# 여기서는 fit 앞에서 저장하므로 "구조만 잡아 둔 훈련 전 모델" 이 저장된다
# 확장자 : .keras(최근 방식) / .h5(예전 방식)

# keras28_Scaler01_california.py 베이스

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

# #2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(50))
model.add(Dense(1))

model.summary()

########## [추가] 모델 저장 - model.save() ##########
# path 는 폴더까지만 적고 반드시 '/' 로 끝낸다.
# './_save/keras29' 처럼 슬래시를 빼면 뒤의 파일명과 그대로 이어붙어서
# _save/keras29keras29_1_save_model.keras 라는 엉뚱한 파일이 만들어진다.
path = './_save/keras29/'
# model.save(path + 'keras29_1_save_model.h5') # 예전 방식 (.h5)
model.save(path + 'keras29_1_save_model.keras') # 최근 방식 (.keras)

# 여기서는 컴파일/훈련 '전'에 저장하므로 모델 "구조"만 담긴다. (Total params 6,001)
exit() # 아래 훈련 코드는 실행하지 않고 여기서 끝낸다

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

# ===== 실행 결과 =====
# 이 파일은 exit() 로 저장까지만 하고 끝나므로 loss 가 나오지 않는다.
# 저장 결과 : _save/keras29/keras29_1_save_model.keras
#            Total params 6,001 - 훈련 전이라 구조와 초기 가중치만 들어있다.
#
# 스케일러별 loss 비교 기록은 keras28_Scaler01_california.py 에 있다
