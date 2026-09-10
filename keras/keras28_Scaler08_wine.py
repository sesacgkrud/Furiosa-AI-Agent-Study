import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

#1. 데이터
datasets = load_wine()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']
# print(x.shape) # (178, 13)
# print(y.shape) # (178,)

########## OneHot Encording 1. (to_categorical) ##########
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y.shape) # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=50,
    shuffle=True,
    stratify=y,
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) # sklearn 에서 fit -> 실행하다 로 생각
x_train = scaler.transform(x_train) # x에 있는 모든 데이터는 0~1 사이로 수렴
x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # -0.2222222222222222 Max : 1.2690763052208835

# print(x_train.shape, x_test.shape) # (124, 13) (54, 13)
# print(y_train.shape, y_test.shape) # (124, 3) (54, 3)

#2. 모델 구성
model = Sequential()
model.add(Dense(200, input_dim=13, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=2000,
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

# [목표] acc = 0.95 이상 합격
# loss : 0.12328256666660309
# acc : 0.963
# accuracy_score : 0.9629629629629629
# 소요 시간 : 16 초

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# loss : 0.08898330479860306
# acc : 0.981
# accuracy_score : 0.9814814814814815
# 소요 시간 : 116 초

# [결론] 0.963 -> 0.981. 좋아진 것처럼 보이지만 그대로 믿으면 안 된다.
#        x_test 가 54개뿐이라 0.963 = 52/54, 0.981 = 53/54 -> 딱 1개 차이다.
#        seed 를 고정하지 않았으니 이 정도는 다시 돌리면 뒤집힐 수 있는 숫자다.
#        [참고] 소요 시간이 16초 -> 116초로 늘어난 것도 스케일링 자체가 느려서가 아니다.
#        스케일링 후 val_loss 가 더 오래 개선되니까 EarlyStopping(patience=50)이 늦게 걸린 것이다.
#        즉 '더 오래 학습할 수 있었다'는 뜻이라 오히려 좋은 신호에 가깝다.