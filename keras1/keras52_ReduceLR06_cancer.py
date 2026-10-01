# keras52_ReduceLR06_cancer.py
# 유방암 (이진 분류) - ReduceLROnPlateau 로 훈련 도중 learning_rate 를 줄인다
# 스케일러(RobustScaler)와 learning_rate 후보는 keras52_optimizer 파일 그대로다
#
# [ 오늘 추가되는 곳은 콜백 하나다 ]
#  keras52_optimizer 는 learning_rate 를 처음부터 끝까지 고정했다
#  하지만 한 값으로 두 가지를 다 할 수는 없다
#   초반에는 크게 움직여 빨리 내려가야 하고, 최저점 근처에서는 작게 움직여야 지나치지 않는다
#
# [ ReduceLROnPlateau : 훈련 도중 learning_rate 를 줄여 준다 ]
#  plateau = 고원. val_loss 가 더 이상 줄지 않고 평평해지는 구간을 말한다
#  monitor 가 patience 동안 좋아지지 않으면 learning_rate 에 factor 를 곱한다
#   factor=0.5 -> 0.0006 -> 절반 -> 또 절반 ... 으로 떨어진다
#  EarlyStopping 과 역할이 다르다 : es 는 '멈춘다', rlr 은 '보폭을 줄여 더 해 본다'
#   그래서 rlr 의 patience 를 es 보다 짧게 줘야 "줄여 보고 -> 그래도 안 되면 멈춘다" 순서가 된다
#  이 파일 : 시작 learning_rate = 0.0006 / es patience = 20 / rlr patience = 20
#   -> 둘이 같아서 lr 이 줄어드는 바로 그 epoch 에 es 도 같이 걸린다
#
# keras28_Scaler06_cancer.py 베이스

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.datasets import load_breast_cancer # 유방암 관련 데이터셋 불러오기

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR) # describe 묘사하다 약자
print(datasets.feature_names) # 컬럼명 출력 -> 30개

x = datasets['data']
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x)) # <class 'numpy.ndarray'>
print(y)

print(np.unique(y)) # 중복 제거한 유니크한 값 -> [0 1]
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357])) -> 0은 212개, 1은 357개

print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42,
    stratify=y, # y데이터를 stratify = y를 기준으로 데이터의 수가 동등하게 분배
)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0000000000000002
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : -0.12351543942992871 Max : 2.4667638053782883

print(np.unique(y_train, return_counts=True)) # (array([0, 1]), array([148, 250]))
print(np.unique(y_test, return_counts=True)) # (array([0, 1]), array([ 64, 107]))

print(x_train.shape, x_test.shape) # (398, 30) (171, 30)
print(y_train.shape, y_test.shape) # (398,) (171,)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=30, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 무조건 sigmoid 를 넣음

#3. 컴파일, 훈련
# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009
learning_rate = 0.0006

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate),
            #   metrics=['accuracy'],
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
)

# val_loss 가 patience 동안 좋아지지 않으면 learning_rate 에 factor 를 곱해 줄인다
# verbose=1 -> 줄어드는 순간 'ReduceLROnPlateau reducing learning rate to ...' 가 찍힌다
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=32,
          verbose=1,
          callbacks=[es, rlr],
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

# ========== ========== ========== ========== ========== <- learning rate 적용 후

# loss : 0.08060426265001297
# acc : 0.9766
# acc_score : 0.9766081871345029

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후

# loss : 0.09319167584180832
# acc : 0.9708
# acc_score : 0.9707602339181286

# [결론] 0.01 고정 0.9766 -> 시작 0.0006 + ReduceLROnPlateau 0.9708. 1개 차이라 사실상 같다.
#        es 와 rlr 의 patience 가 둘 다 20 이라 lr 이 줄어드는 순간 es 도 같이 걸린다.
