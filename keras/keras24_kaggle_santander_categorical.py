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

# [수정] 맨 위에서 이미 submission_csv 로 같은 파일을 읽었는데 여기서 또 읽고 있었다. 중복이라 위의 것을 재사용한다.
submission_csv['target'] = y_submit

# [수정] 저장 파일명이 submit_0908_1642.csv 로 되어 있었다.
#        그 파일은 keras22_sigmoid_santander.py(sigmoid 이진 분류)가 만든 제출 파일인데,
#        이 파일(softmax 다중 분류)을 실행할 때마다 그 결과를 덮어써 버렸다.
#        sigmoid 는 확률값(0.22...)을, 여기 softmax + argmax 는 0/1 을 저장하므로 내용도 완전히 다르다.
#        -> 실습마다 제출 파일을 따로 남겨야 sigmoid 방식과 softmax 방식의 점수를 비교할 수 있다.
#        규칙: submit_날짜_시간_실습이름.csv 로 실습마다 고유한 이름을 준다.
submission_csv.to_csv(
    path + 'submit/' + 'submit_0910_1104_categorical.csv'
)

# loss :  [0.24368469417095184, 0.9112666845321655]
# acc :  0.9113
# acc_score :  0.9112666666666667