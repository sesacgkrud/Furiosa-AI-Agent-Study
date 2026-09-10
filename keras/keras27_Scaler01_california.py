# keras19_overfit1_california.py 베이스

from sklearn.datasets import fetch_california_housing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

'''
MinMaxScaler Algorithm
-> (원값 - Min) / (Max - Min)
-> 0 ~ 1 사이로 수렴
'''

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) # sklearn 에서 fit -> 실행하다 로 생각
x = scaler.transform(x) # x에 있는 모든 데이터는 0~1 사이로 수렴
print(x)
print('Min :', np.min(x), 'Max :', np.max(x)) # Min : 0.0 Max : 1.0000000000000002 -> 부동 소숫점 연산에 의한 오차 발생 (문제X)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.75,
    random_state=100,
)

#################### [참고] 위 스케일링의 문제점 ####################
# 위에서는 split 하기 전에 전체 x 로 fit 을 했다.
# 이러면 x_test 의 Min/Max 까지 변환 기준에 반영되므로,
# 아직 보면 안 되는 평가 데이터의 정보가 미리 새어 들어간다. (데이터 누수)
# -> keras28_Scaler01_california.py 에서 split 을 먼저 하고
#    scaler.fit(x_train) 으로 훈련 데이터에만 fit 하도록 고친다.
####################################################################

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

from sklearn.metrics import r2_score, mean_squared_error

print("loss(mse) :", loss)
y_predict = model.predict(x_test).ravel()

# [삭제] 여기에 r2_score, mean_squared_error 를 다시 import 하는 줄이 있었는데
#        맨 위에서 이미 import 했으므로 중복이라 지웠다.

r2 = r2_score(y_test, y_predict)

print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# [주의] 아래 결과는 이 파일의 스케일링 코드가 바뀌기 전에 나온 값이라 지금 코드와 맞지 않는다.
#        현재 코드로 다시 실행해서 갱신할 것.
# loss(mse) : 0.333017498254776
# r2 : 0.7474706509573199
# mse : 0.33301733070585554
# RMSE : 0.5770765379963524