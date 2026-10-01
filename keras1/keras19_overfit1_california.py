# keras19_overfit1_california.py
# 과적합(overfit) 확인 : fit 의 결과를 hist 에 받아서 loss 와 val_loss 를 그래프로 그린다
# hist.history 안에 epoch 별 loss / val_loss 가 리스트로 들어있다
# loss 는 계속 내려가는데 val_loss 가 어느 순간부터 올라가면 그 지점부터 과적합이다

# keras17_val1_california.py 베이스

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

#################### [참고] loss 그래프가 우하향 안 하는 이유 ####################
# california 는 컬럼마다 값의 범위가 완전히 다르다.
#   MedInc(소득) 0~15 / AveRooms 몇 개 / Population(인구) 수천 / AveOccup 는 1000 넘는 이상치까지
# 값이 큰 컬럼 하나 때문에 gradient 가 그쪽으로만 크게 튀어서,
# loss 가 내려가다가 갑자기 솟구치는(톱니 모양) 그래프가 된다. -> "대체로 우하향"이 안 보인다.
# -> 이 문제는 keras28_Scaler01_california.py 에서 MinMaxScaler 로 해결한다.
################################################################################

#2. 모델 구성
model = Sequential()
# 활성화 함수가 하나도 없으면 층을 아무리 쌓아도 결국 y = wx + b 짜리 "직선 모델" 1개와 같다
# (선형 x 선형 = 선형) 그래서 표현력이 부족해 loss 가 어느 선에서 더 못 내려가고 출렁인다
# -> 은닉층에는 relu 를 붙이고, 회귀의 출력층에는 activation 을 안 붙인다
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
          validation_split=0.2)

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
print("loss :", loss)

#4. 평가 예측
# 평가는 훈련에도 검증에도 안 쓴 x_test 로만 한다 -> 위의 evaluate 하나면 충분하다
# 전체 데이터(x)를 그대로 넣으면 훈련에 쓴 데이터가 섞여 있어서(시험문제에 답안지 포함) 평가 의미가 없다
# (이 파일에는 스케일러가 없다. 스케일링은 keras27 부터 다룬다)

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
plt.rcParams['axes.unicode_minus'] = False   # [추가] 한글 폰트 쓸 때 음수 부호가 □ 로 깨지는 것 방지

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss') # y값만 넣으면 시간 순으로 그림
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')

plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title('California(캘리포니아) Loss')

plt.xlabel('epochs')
plt.ylabel('loss')   # y축은 loss, x축은 epochs

plt.grid() # 격자 표시 추가
plt.show()
