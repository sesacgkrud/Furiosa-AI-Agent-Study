# keras21_sigmoid_metrics_cancer.py
# 이진 분류 첫 실습 (유방암 : 악성 0 / 양성 1) - 회귀와 달라지는 3가지
#  1) 출력층 activation = sigmoid  -> 결과를 0 ~ 1 사이 확률로 눌러 준다
#  2) loss = 'binary_crossentropy' -> 확률이 정답에서 멀수록 크게 벌점을 준다 (mse 대신)
#  3) metrics=['acc']              -> 훈련 중에 정확도를 같이 찍어 본다 (loss 와 달리 훈련에는 영향 없음)
# stratify=y : train / test 를 나눌 때 0 과 1 의 비율을 똑같이 유지한다

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

print("acc_score :", acc_score) # acc_score : 0.9239766081871345