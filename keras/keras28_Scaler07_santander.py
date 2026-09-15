# keras28_Scaler07_santander.py
# 산탄데르 (이진 분류, 제출 파일 생성)
# 스케일러 4종 비교 (MinMax / Standard / MaxAbs / Robust) - 파일 아래쪽에 스케일러별 결과를 기록해 둔다
# 순서가 중요하다 : train_test_split 을 먼저 하고 -> scaler.fit(x_train) -> x_test 는 transform 만
#  fit 은 변환 기준(Min/Max, 평균, 중앙값 등)을 구하는 단계라 x_train 으로만 해야 한다
#  x_test 로 fit 하면 아직 보면 안 되는 평가 데이터의 정보가 기준에 섞인다 (데이터 누수)

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
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : -0.12100998038044875 Max : 1.07988866097767

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

#################### 제출용 test_csv 도 같은 scaler 로 변환한다 ####################
# 모델은 0~1 로 변환된 x_train 으로 학습하므로 제출용 예측에 원본 test_csv 를 그대로 넣으면 안 된다.
# x_test 로 잰 acc 는 (x_test 는 변환했으므로) 맞는 값이어도,
# 저장되는 제출 csv 만 학습 때와 단위가 다른 입력으로 뽑은 예측이 된다.
# 여기서도 scaler.fit 은 절대 다시 하지 않는다.
# scaler 안에 저장된 x_train 의 Min/Max 를 공식에 대입해 값만 바꾼다. (Min/Max 를 새로 구하지 않는다)
# test_csv 로 fit 하면 훈련 때와 다른 Min/Max 로 변환되어 학습한 모델과 단위가 어긋난다.
test_csv_scaled = scaler.transform(test_csv)
######################################################################

y_submit = model.predict(test_csv_scaled)
y_submit = np.argmax(y_submit, axis=1)

# 맨 위에서 읽어 둔 submission_csv 를 그대로 재사용한다 (같은 파일을 두 번 읽을 필요가 없다)
submission_csv['target'] = y_submit

# 제출 파일은 실습마다 다른 이름으로 저장한다.
# 스케일링 적용 전(keras24)의 제출 파일과 캐글 점수를 비교하는 것이 이 실습의 목적이므로
# 이전 파일은 남겨두고 새 이름(_scaler)으로 저장한다.
# 규칙 : submit_날짜_시간_실습이름.csv
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