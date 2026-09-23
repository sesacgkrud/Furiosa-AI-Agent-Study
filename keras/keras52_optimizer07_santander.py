# keras52_optimizer07_santander.py
# 산탄데르 (이진 분류, softmax 2칸 + 제출 파일 생성) - learning_rate 직접 지정
# 스케일러는 keras28 에서 비교해 고른 RobustScaler 를 그대로 쓴다
#
# [ 오늘 바뀌는 곳은 compile 한 줄이다 ]
#  before : model.compile(..., optimizer='adam')                  -> learning_rate 가 기본값 0.001 로 고정
#  after  : model.compile(..., optimizer=Adam(learning_rate=...))  -> 보폭을 내가 정한다
#
# [ learning_rate = 가중치를 한 번에 얼마나 크게 고칠지 정하는 보폭 ]
#  크면 (0.05 / 0.01)  : 최저점을 건너뛰고 그 주변에서 튕겨 다닌다
#  작으면 (0.00001)    : 한 걸음이 작아서 epoch 가 끝날 때까지 최저점에 닿지 못한다
#  데이터마다 맞는 값이 달라서 후보를 주석으로 적어 두고 하나씩 열어 가며 직접 비교했다
#  이 파일에서 고른 값 : learning_rate = 0.0015
#
# keras28_Scaler07_santander.py 베이스

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

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0000000000000002
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : -0.12100998038044875 Max : 1.07988866097767

model = Sequential()
model.add(Dense(500, input_dim=200, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# optimizer 를 문자열('adam') 이 아니라 객체로 만들어 넘겨야 learning_rate 를 바꿀 수 있다
from tensorflow.keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001    # optimizer Adam default value
# learning_rate = 0.00001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009
learning_rate = 0.0015

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=learning_rate),
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

test_csv_scaled = scaler.transform(test_csv)

y_submit = model.predict(test_csv_scaled)
y_submit = np.argmax(y_submit, axis=1)

submission_csv['target'] = y_submit
submission_csv.to_csv(
    path + 'submit/' + 'submit_0910_1724_scaler.csv'
)

# loss :  [0.24368469417095184, 0.9112666845321655]
# acc :  0.9113
# acc_score :  0.9112666666666667

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# loss :  [0.23260438442230225, 0.9146166443824768]
# acc :  0.9146
# acc_score :  0.9146166666666666

# [결론] 0.9113 -> 0.9146 으로 거의 변화 없다.
#        santander 의 var_0 ~ var_199 는 이미 비슷한 크기의 값으로 익명화돼 있어서
#        스케일링으로 바로잡을 '범위 차이'가 애초에 없다. 효과가 없는 게 정상이다.
#        [주의] test_csv 스케일링을 고쳤으므로 제출 파일은 다시 만들어야 한다.
#        위 acc 는 x_test 기준이라 그대로 유효하지만, Kaggle 점수는 새로 제출해서 확인할 것.

# ========== ========== ========== ========== ========== <- StandardScaler 적용 후

# loss :  [0.24281403422355652, 0.9108666777610779]
# acc :  0.9109
# acc_score :  0.9108666666666667

# ========== ========== ========== ========== ========== <- MaxAbsScaler 적용 후

# loss :  [0.23883703351020813, 0.9131166934967041]
# acc :  0.9131
# acc_score :  0.9131166666666667

# ========== ========== ========== ========== ========== <- RobustScaler 적용 후

# loss :  [0.24184152483940125, 0.9115666747093201]
# acc :  0.9116
# acc_score :  0.9115666666666666

# ========== ========== ========== ========== ========== <- learning rate 적용 후

# loss :  [0.23909297585487366, 0.911633312702179]
# acc :  0.9116
# acc_score :  0.9116333333333333

# [결론] RobustScaler 0.9116 -> learning_rate 0.0015 로 0.9116. 변화가 없다.
#        santander 는 스케일링에도 반응이 없던 데이터다 (var_0 ~ var_199 가 이미 비슷한 크기).
#        learning_rate 를 만져도 한계가 같다는 것을 확인한 셈이다.
