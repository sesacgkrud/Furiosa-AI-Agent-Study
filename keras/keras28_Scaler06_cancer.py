# keras28_Scaler06_cancer.py
# 유방암 (이진 분류)
# 스케일러 4종 비교 (MinMax / Standard / MaxAbs / Robust) - 파일 아래쪽에 스케일러별 결과를 기록해 둔다
# 순서가 중요하다 : train_test_split 을 먼저 하고 -> scaler.fit(x_train) -> x_test 는 transform 만
#  fit 은 변환 기준(Min/Max, 평균, 중앙값 등)을 구하는 단계라 x_train 으로만 해야 한다
#  x_test 로 fit 하면 아직 보면 안 되는 평가 데이터의 정보가 기준에 섞인다 (데이터 누수)

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer # 유방암 관련 데이터셋 불러오기

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR) # describe 묘사하다 약자
print(datasets.feature_names) # 컬럼명 출력 -> 30개

# x = datasets.data
x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x)) # <class 'numpy.ndarray'>
print(y)

# 0과 1의 갯수가 몇 개인지? -> numpy
print(np.unique(y)) # 중복 제거한 유니크한 값 -> [0 1]
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357])) -> 0은 212개, 1은 357개

# 0과 1의 갯수가 몇 개인지? -> pandas
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42,
    stratify=y, # y데이터를 stratify = y를 기준으로 데이터의 수가 동등하게 분배
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

# scaler.fit(x_train) # sklearn 에서 fit -> 실행하다 로 생각
# x_train = scaler.transform(x_train) # x에 있는 모든 데이터는 0~1 사이로 수렴
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0000000000000002
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : -0.12351543942992871 Max : 2.4667638053782883

# print(np.unique(y_train, return_counts=True)) # (array([0, 1]), array([149, 249]))
# print(np.unique(y_test, return_counts=True)) # (array([0, 1]), array([ 63, 108])) -> test는 평가에만 들어가니 값 차이가 많이 나도 문제 없음

# stratify=y 추가 후
print(np.unique(y_train, return_counts=True)) # (array([0, 1]), array([148, 250]))
print(np.unique(y_test, return_counts=True)) # (array([0, 1]), array([ 64, 107]))

print(x_train.shape, x_test.shape) # (398, 30) (171, 30)
print(y_train.shape, y_test.shape) # (398,) (171,)

# exit()

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=30, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 무조건 sigmoid 를 넣음

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            #   metrics=['accuracy'],
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=32,
          verbose=1,
          callbacks=[es],
          validation_split=0.3,
          )
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
# print("loss :", loss) # loss : [0.20410895347595215, 0.9181286692619324] -> [loss, accuracy]
print("loss :", loss[0]) # loss : 0.17925329506397247
print("acc :", round(loss[1], 4)) # acc : 0.9357


# exit()

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
# print(y_predict[:10])
'''
[실행 결과]
[[1.]
 [1.]
 [1.]
 [0.]
 [1.]
 [1.]
 [1.]
 [0.]
 [1.]
 [0.]]
'''

# print(y_predict[:10])
'''
[실행 결과]
[[9.7620481e-01]
 [9.7930312e-01]
 [9.9648571e-01]
 [9.9375278e-02]
 [7.3675966e-01]
 [6.2057316e-01]
 [9.9857819e-01]
 [5.6451809e-04]
 [9.9710459e-01]
 [1.4124110e-09]]
-> 범위 : 0 ~ 1 -> 반올림
'''

# exit()

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_predict) # y_test -> 0, 1 / y_predict -> 0 ~ 1 (sigmoid 통과했기 때문)
# ValueError: Classification metrics can't handle a mix of binary and continuous targets

print("acc_score :", acc_score)
# acc_score : 0.9239766081871345

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# acc_score : 0.9707602339181286

# [결론] 0.9240 -> 0.9708 로 올랐다. cancer 는 mean area(143~2501) 처럼
#        범위가 큰 컬럼과 mean smoothness(0.05~0.16) 처럼 작은 컬럼이 섞여 있어서 효과가 기대되는 데이터다.
#        [주의] 다만 x_test 가 171개뿐이라 0.9240 -> 0.9708 은 맞힌 개수로 8개 차이다.
#        seed 를 고정하지 않았으니 이 숫자 하나만으로 단정하지 말고 몇 번 더 돌려보고 판단할 것.

# ========== ========== ========== ========== ========== <- StandardScaler 적용 후

# loss : 0.07083778083324432
# acc : 0.9825
# acc_score : 0.9824561403508771

# ========== ========== ========== ========== ========== <- MaxAbsScaler 적용 후

# loss : 0.07983746379613876
# acc : 0.9649
# acc_score : 0.9649122807017544

# ========== ========== ========== ========== ========== <- RobustScaler 적용 후

# loss : 0.09062832593917847
# acc : 0.9649
# acc_score : 0.9649122807017544