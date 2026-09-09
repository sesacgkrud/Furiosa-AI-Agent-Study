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