# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# path = './_data/kaggle_santander/' # 상대경로
path = 'c:/furiosa_study/_data/kaggle_santander/' # 절대경로
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# print(train_csv.shape) # (200000, 202) --- index_col=0 ---> (200000, 201)
# print(test_csv.shape) # (200000, 201) --- index_col=0 ---> (200000, 200)
# print(submission_csv.shape) # (200000, 2) --- index_col=0 ---> (200000, 1)

# print(train_csv.info())
# print(train_csv.isna().sum())
# print(test_csv.isnull().sum())

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# print(x.shape) # (200000, 200)
# print(y.shape) # (200000,)

# print(np.unique(y, return_counts=True)) # (array([0, 1]), array([179902,  20098])) -> 0: 179902개, 1: 20098개

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=777,
    stratify=y,
)

# print(np.unique(y_train, return_counts=True)) # (array([0, 1]), array([107941,  12059]))
# print(np.unique(y_test, return_counts=True)) # (array([0, 1]), array([71961,  8039]))

print(x_train.shape, x_test.shape) # (120000, 200) (80000, 200)
print(y_train.shape, y_test.shape) # (120000,) (80000,)

#2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
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
          validation_split=0.2,
          )
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc :', round(loss[1], 4))

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)
print('acc_score :', acc_score)

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
y_submit = model.predict(test_csv)
submission['target'] = y_submit
submission.to_csv(path + 'submit/' + 'submit_0908_1642.csv')