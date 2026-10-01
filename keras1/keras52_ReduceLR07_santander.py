# keras52_ReduceLR07_santander.py
# 산탄데르 (이진 분류, softmax 2칸 + 제출 파일 생성) - ReduceLROnPlateau 적용
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
#   factor=0.5 -> 0.0015 -> 절반 -> 또 절반 ... 으로 떨어진다
#  EarlyStopping 과 역할이 다르다 : es 는 '멈춘다', rlr 은 '보폭을 줄여 더 해 본다'
#   그래서 rlr 의 patience 를 es 보다 짧게 줘야 "줄여 보고 -> 그래도 안 되면 멈춘다" 순서가 된다
#  이 파일 : 시작 learning_rate = 0.0015 / es patience = 20 / rlr patience = 20
#   -> 둘이 같아서 lr 이 줄어드는 바로 그 epoch 에 es 도 같이 걸린다
#
# keras28_Scaler07_santander.py 베이스

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=32,
    verbose=1,
    callbacks=[es, rlr],
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

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후

# loss :  [0.24184267222881317, 0.9104666709899902]
# acc :  0.9105
# acc_score :  0.9104666666666666

# [결론] 0.0015 고정 0.9116 -> ReduceLROnPlateau 0.9105. 역시 거의 같다.
#        바꿀 수 있는 것을 다 바꿔도 0.91 에서 멈춘다 -> 모델 / 데이터 쪽 문제라는 신호다.
