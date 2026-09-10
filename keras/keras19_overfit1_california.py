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
# [수정] 활성화 함수가 하나도 없으면 층을 아무리 쌓아도 결국 y = wx + b 짜리 "직선 모델" 1개와 같다.
#        (선형 x 선형 = 선형) 그래서 표현력이 부족해 loss 가 어느 선에서 더 못 내려가고 출렁인다.
#        은닉층에는 relu 를 붙이고, 회귀의 출력층에는 activation 을 안 붙인다.
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
# [수정] 원본은 model.evaluate(x, y) 였는데 두 가지가 문제였다.
#        1) x 는 스케일링이 안 된 원본이라 학습 때와 기준이 달라 loss 가 엉뚱하게 나온다.
#        2) x 안에는 train 데이터가 이미 들어있어서(= 시험문제에 답안지 포함) 평가 의미가 없다.
#        전체 데이터로 확인하고 싶다면 최소한 같은 scaler 로 변환해서 넣어야 한다.
loss_all = model.evaluate(scaler.transform(x), y)
print("loss(전체 데이터, 참고용) :", loss_all)

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
plt.ylabel('loss')   # [수정] 원본은 xlabel 이 두 번이라 x축 라벨이 'loss' 로 덮여쓰이고 y축은 비어 있었다

plt.grid() # 격자 표시 추가
plt.show()
