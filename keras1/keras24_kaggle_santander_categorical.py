# keras24_kaggle_santander_categorical.py
# keras22 에서 sigmoid(이진 분류)로 풀었던 산탄데르를 softmax(다중 분류) 방식으로 다시 푼다
# 답이 0 / 1 두 개뿐이어도 원핫 + softmax(2칸) 으로 풀 수 있다 -> 두 방식의 캐글 점수를 비교하는 것이 목적
# 제출 값도 달라진다 : sigmoid 는 확률(0.22...), 여기는 argmax 를 거친 0 또는 1

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

path = 'c:/furiosa_study/_data/kaggle_santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,
    random_state=777,
    stratify=np.argmax(y, axis=1)
)

model = Sequential()
model.add(Dense(500, input_dim=200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=32,
    verbose=1,
    callbacks=[es],
    validation_split=0.2
)

end_time = time.time()

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', round(loss[1], 4))

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

y_test_label = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test_label, y_predict)
print('acc_score : ', acc_score)

y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis=1)

# 맨 위에서 읽어 둔 submission_csv 를 그대로 재사용한다 (같은 파일을 두 번 읽을 필요가 없다)
submission_csv['target'] = y_submit

# 제출 파일은 실습마다 고유한 이름으로 저장한다.
# keras22_sigmoid_santander.py(sigmoid 이진 분류)와 이 파일(softmax 다중 분류)은
# 저장하는 값의 성격이 다르다.
# sigmoid 는 확률값(0.22...)을, softmax + argmax 는 0/1 을 저장한다.
# -> 파일명이 겹치면 서로 덮어써서 두 방식의 점수를 비교할 수 없다.
# 규칙 : submit_날짜_시간_실습이름.csv
submission_csv.to_csv(
    path + 'submit/' + 'submit_0910_1104_categorical.csv'
)

# loss :  [0.24368469417095184, 0.9112666845321655]
# acc :  0.9113
# acc_score :  0.9112666666666667