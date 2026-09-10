import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype

#1. 데이터
datasets = fetch_covtype()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
# print(x.shape) # (581012, 54)
# print(y.shape) # (581012,)

########## OneHot Encording 1. (to_categorical) ##########
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y.shape) # (581012, 8)

## 선생님 실습 - 첫번째 컬럼이 0으로 채워져 있는 것 확인용
# y_oh = to_categorical(y)
# print(y_oh[:10])
# print(y[:10])
# print(y_oh.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42,
    shuffle=True,
    stratify=y,
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) # sklearn 에서 fit -> 실행하다 로 생각
x_train = scaler.transform(x_train) # x에 있는 모든 데이터는 0~1 사이로 수렴
x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : 0.0 Max : 1.0050359712230217

print(x_train.shape, x_test.shape) # (406708, 54) (174304, 54)
print(y_train.shape, y_test.shape) # (406708, 8) (174304, 8)

#2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim=54, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(8, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=1000,
          batch_size=32,
          verbose=1,
          validation_split=0.3,
          callbacks=[es],
)
end_time = time.time()

#4. 평가 예측
result = model.evaluate(x_test, y_test, )
print('loss :', result[0])
print('acc :', round(result[1],3))

y_predict = model.predict(x_test)

y_test_arg = np.argmax(y_test, axis=1)
y_predict_arg = np.argmax(y_predict, axis=1)

accuracy_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', accuracy_score)
print('소요 시간 :', round(end_time - start_time), '초')

# [목표] acc = 0.93 이상 합격
# [참고] 데이터 50만개, 시간 측정, batch_size 작게 주지 말 것

# 시간 오래 걸림, 이 예제의 경우 0부터 시작하는 to_categorical 적합하지 않음 (다른 방법으로 시도하기)
# loss : 0.34686315059661865
# acc : 0.861
# accuracy_score : 0.8611735817881403
# 소요 시간 : 1463 초

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# loss : 0.1859578788280487
# acc : 0.936
# accuracy_score : 0.9361001468698366
# 소요 시간 : 1110 초

# [결론] 0.861 -> 0.936. 이번 실습에서 스케일링 효과가 가장 확실하게 나온 데이터다.
#        covtype 은 Elevation(1859~3858), Horizontal_Distance(0~7000) 같은 큰 값 컬럼과
#        Wilderness_Area / Soil_Type 처럼 0 아니면 1 인 컬럼이 한 테이블에 섞여 있다.
#        스케일링 전에는 값이 큰 컬럼 쪽으로만 gradient 가 크게 튀어서 학습이 제대로 안 됐던 것.
#        데이터가 58만개라 test 도 17만개 -> 이 정도 표본이면 0.861 -> 0.936 은 우연이 아니다.
#        [참고] 시간이 1463초 -> 1110초로 줄어든 것도 같은 이유다. 더 빨리 수렴해서 EarlyStopping 이 일찍 걸렸다.
#        목표였던 acc 0.93 도 스케일링만으로 넘겼다.