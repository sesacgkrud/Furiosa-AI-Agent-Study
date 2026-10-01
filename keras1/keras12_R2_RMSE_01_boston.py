# keras12_R2_RMSE_01_boston.py
# 회귀 평가지표 3종을 처음 정리한 파일 (한 줄씩 주석을 달아 둔 기준 파일)
# r2   : 1 에 가까울수록 좋다 (데이터가 달라도 서로 비교할 수 있다)
# mse  : 오차를 제곱해서 평균 낸 값 -> loss='mse' 로 훈련했다면 evaluate 의 loss 와 같은 값이 나온다
# RMSE : mse 에 루트를 씌운 값 -> 정답과 단위가 같아져서 "평균 몇 만큼 틀렸는지" 로 읽을 수 있다

from tensorflow.keras.models import Sequential # 순차적으로 층을 쌓는 신경망 모델을 불러옴
from tensorflow.keras.layers import Dense # 완전연결층(Dense)을 불러옴
from tensorflow.keras.datasets import boston_housing # 보스턴 주택 가격 데이터를 불러옴
import numpy as np

#1. 데이터
# 보스턴 주택 가격 데이터를 학습용과 테스트용으로 나눠서 불러옴
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# x_train : 학습용 입력 데이터
# y_train : 학습용 정답 데이터
# x_test  : 테스트용 입력 데이터
# y_test  : 테스트용 정답 데이터

print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
# x_train의 형태 → (404, 13)
# x_test의 형태  → (102, 13)
# 13개의 특징(feature)을 이용해서 집값을 예측함

print(y_train.shape, y_test.shape) # (404,) (102,)
# y_train의 형태 → (404,)
# y_test의 형태  → (102,)
# 각각의 데이터에 대한 실제 집값이 들어 있음

#2. 모델 구성
model = Sequential()
# Sequential 모델 생성
# 신경망 층을 순서대로 하나씩 쌓아가는 방식

model.add(Dense(30, input_dim=13))
# 첫 번째 Dense 층
# 입력 데이터의 특징(feature)이 13개이므로 input_dim=13
# 뉴런 30개를 사용

model.add(Dense(20))
# 두 번째 Dense 층
# 뉴런 20개를 사용

model.add(Dense(30))
# 세 번째 Dense 층
# 뉴런 30개를 사용

model.add(Dense(1))
# 출력층
# 집값 하나를 예측해야 하므로 뉴런 1개 사용

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# 모델 학습 방법 설정
#
# loss='mse'
# → 손실 함수로 평균제곱오차(MSE)를 사용
# → 실제값과 예측값의 차이를 제곱해서 평균을 구함
#
# optimizer="adam"
# → Adam 옵티마이저를 사용하여 가중치를 업데이트함

model.fit(x_train, y_train, epochs=1000, batch_size=16)
# 학습 데이터(x_train)와 정답(y_train)을 이용해서 모델을 학습
#
# epochs=1000
# → 전체 학습 데이터를 1000번 반복해서 학습
#
# batch_size=64
# → 한 번에 64개의 데이터를 사용하여 가중치를 업데이트

print("========== ========== ========== ========== ==========")

loss = model.evaluate(x_test, y_test)
# 학습에 사용하지 않은 테스트 데이터로 모델의 성능을 평가
#
# x_test → 테스트 입력 데이터
# y_test → 테스트 실제 정답
#
# 여기서는 compile에서 loss='mse'를 사용했기 때문에
# 반환되는 loss 값은 MSE(평균제곱오차)

print("loss :", loss)
# 테스트 데이터에 대한 MSE 출력
#
# 예:
# loss : 23.9038
#
# → 실제 집값과 예측 집값의 차이를 제곱해서 평균낸 값

# loss : 23.9038639068603

#4. 평가 예측
loss = model.evaluate(x_test, y_test, )
# 테스트 데이터를 이용해서 모델 성능을 다시 평가
# 위에서 이미 평가했기 때문에 사실상 같은 작업을 한 번 더 하는 것

print("loss(mse) :", loss)
# 테스트 데이터의 MSE 출력

y_predict = model.predict(x_test)
# 테스트 데이터(x_test)를 모델에 넣어서
# 집값을 예측함
#
# y_predict → 모델이 예측한 집값

from sklearn.metrics import r2_score, mean_squared_error # 평가할 때 주로 사용
# scikit-learn에서 R²(결정계수)를 계산하는 함수를 가져옴

r2 = r2_score(y_test, y_predict) # 원값과 예측값 비교 -> loss
# 실제 정답값(y_test)과
# 모델의 예측값(y_predict)을 비교해서
# R² 점수를 계산함

print("r2 :", r2)
# 계산된 R² 점수를 출력

# loss : 23.059614181518555
# loss(mse) : 23.059614181518555
# r2 : 0.7229871515207699

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict): #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict)) # MSE에 루트 씌움

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# loss : 27.36833381652832
# loss(mse) : 27.36833381652832
# r2 : 0.6712269182342471
# mse : 27.368336615896183
# RMSE : 5.231475567743405