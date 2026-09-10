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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) # sklearn 에서 fit -> 실행하다 로 생각
x_train = scaler.transform(x_train) # x에 있는 모든 데이터는 0~1 사이로 수렴
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

#################### [수정] test_csv 스케일링 누락 ####################
# 모델은 0~1 로 변환된 x_train 으로 학습했는데, 제출용 예측에만 원본 test_csv 를 넣고 있었다.
# x_test 로 잰 acc 0.9146 은 (x_test 는 변환했으므로) 맞는 값이지만,
# 실제로 저장한 제출 csv 는 학습 때와 단위가 다른 입력으로 뽑은 엉터리 예측이었다.
# 여기서도 scaler.fit 은 절대 다시 하지 않는다.
# scaler 안에 저장된 x_train 의 Min/Max 를 공식에 대입해 값만 바꾼다. (Min/Max 를 새로 구하지 않는다)
# test_csv 로 fit 하면 훈련 때와 다른 Min/Max 로 변환되어 학습한 모델과 단위가 어긋난다.
test_csv_scaled = scaler.transform(test_csv)
######################################################################

y_submit = model.predict(test_csv_scaled)
y_submit = np.argmax(y_submit, axis=1)

# [수정] 맨 위에서 이미 submission_csv 로 같은 파일을 읽었는데 여기서 또 읽고 있었다. 중복이라 위의 것을 재사용한다.
submission_csv['target'] = y_submit

# [수정] 저장 파일명이 submit_0908_1642.csv 로 되어 있어서
#        스케일링 적용 전(keras24)에 만들어 둔 제출 파일을 덮어써 버렸다.
#        before / after 의 Kaggle 점수를 비교하는 게 이 실습의 목적이므로
#        이전 파일은 남겨두고 새 이름으로 저장해야 한다.
#        (덮어쓴 원본은 git 에 커밋돼 있으니 되살릴 수 있다:
#         git checkout -- _data/kaggle_santander/submit/submit_0908_1642.csv)
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