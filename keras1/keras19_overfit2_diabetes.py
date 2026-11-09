# keras19_overfit2_diabetes.py
# keras17_val2_diabetes.py 베이스
# [목적] 과적합 확인 : loss 와 val_loss 를 같이 그려서 val_loss 가 올라가기 시작하는 지점을 찾는다
# 당뇨 데이터는 442개로 작아서 val_loss 가 크게 출렁인다

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

# x_val 을 만들었지만 아래 fit 은 validation_data 대신 validation_split=0.3 을 써서 x_val 은 사용되지 않는다
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.7,
    random_state=77,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=500, batch_size=16,
          validation_split=0.3,
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
# [10:] : 처음 10 epoch 은 loss 가 너무 커서 그래프가 납작해지므로 빼고 그린다
plt.plot(hist.history['loss'][10:], c='red', label='loss') # y값만 넣으면 시간 순으로 그림
plt.plot(hist.history['val_loss'][10:], c='blue', label='val_loss')

plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.title('Diabetes Loss')

plt.xlabel('epochs')
# ※ 아래 두 번째 plt.xlabel('loss') 는 y축 이름이므로 plt.ylabel('loss') 이어야 한다 (지금은 x축 이름이 'loss' 로 덮어써지고 y축 이름은 비어 있다)
plt.xlabel('loss')

plt.grid() # 격자 표시 추가
plt.show()

# loss: 2066.9373 - val_loss: 3627.6526
# loss: 3002.0483