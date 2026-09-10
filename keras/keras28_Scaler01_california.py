# keras20_EarlyStopping1_california.py 베이스

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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) # sklearn 에서 fit -> 실행하다 로 생각
x_train = scaler.transform(x_train) # x에 있는 모든 데이터는 0~1 사이로 수렴
x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0000000000000004
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : -0.0012367054167697258 Max : 1.0000000000000004

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
# [수정] 원본에는 여기에 loss = model.evaluate(x, y) 가 있었는데 두 가지 문제로 삭제했다.
#        1) x 는 스케일링을 안 한 원본이다. 모델은 0~1 로 변환된 값으로 학습했는데
#           평가에만 원본(MedInc 0~15, Population 수천)을 넣으니 loss 가 89923 처럼 터졌다.
#           -> 모델이 나쁜 게 아니라 자와 대상의 단위가 안 맞은 것. 성능 비교와 무관한 숫자다.
#        2) x 안에는 x_train 이 그대로 들어있다. (시험지에 답안지가 섞인 상태)
#           평가는 훈련에도 검증에도 안 쓴 x_test 로만 해야 하므로 위의 evaluate 하나면 충분하다.
#        전체 데이터로 굳이 확인하고 싶다면 scaler.transform(x) 로 같은 기준으로 변환해서 넣어야 한다.

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
# [수정] 아래에 있던 'loss : 89923.1484375' 는 스케일링 안 한 x 를 넣어서 나온 값이라 삭제했다.

# [결론] 0.635 -> 0.504 (약 20% 개선). california 는 컬럼별 값 범위 차이가 커서 스케일링 효과가 크다.