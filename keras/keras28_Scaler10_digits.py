import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

#1. 데이터
datasets = load_digits()
# print(datasets.DESCR)
# print(datasets.feature_names) # (1797, 64)

x = datasets.data
y = datasets['target']
# print(x.shape) # (1797, 64)
# print(y.shape) # (1797,)
# print(y) # [0 1 2 ... 8 9 8]
# print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

########## OneHot Encording 1. (to_categorical) ##########
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y.shape) # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.6,
    random_state=99,
    shuffle=True,
    stratify=y,
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) # sklearn 에서 fit -> 실행하다 로 생각
x_train = scaler.transform(x_train) # x에 있는 모든 데이터는 0~1 사이로 수렴
x_test = scaler.transform(x_test) # x에 있는 모든 데이터는 0~1 사이로 수렴

print('Min :', np.min(x_train), 'Max :', np.max(x_train)) # Min : 0.0 Max : 1.0
print('Min :', np.min(x_test), 'Max :', np.max(x_test)) # Min : 0.0 Max : 2.6666666666666665
# [참고] x_test 의 Max 가 2.67 로 1 을 넘는데 이건 버그가 아니라 정상이다.
#        digits 는 8x8 이미지라 가장자리 픽셀은 대부분 0 이다.
#        train 에서 그 컬럼의 max 가 6 이었다면 (원값 - 0) / (6 - 0) 이 기준이 되는데,
#        test 에 16 짜리 값이 나오면 16/6 = 2.67 이 된다.
#        오히려 fit 을 x_train 에만 제대로 했다는 증거다. (전체 x 에 fit 했다면 절대 1 을 안 넘는다)

print(x_train.shape, x_test.shape) # (1257, 64) (540, 64)
print(y_train.shape, y_test.shape) # (1257, 10) (540, 10)

#2. 모델 구성
model = Sequential()
model.add(Dense(50, input_dim=64, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=100,
    restore_best_weights=True,
)
start_time = time.time()
model.fit(x_train, y_train,
          epochs=3000,
          batch_size=4,
          verbose=1,
          validation_split=0.3,
          callbacks=[es],
          )
end_time = time.time()

#4. 평가 예측
result = model.evaluate(x_test, y_test, )

print('loss :', result[0])
print('acc :', round(result[1], 3))

y_predict = model.predict(x_test)

y_test_arg = np.argmax(y_test, axis=1)
y_predict_arg = np.argmax(y_predict, axis=1)

accuracy_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', accuracy_score)
print('소요 시간 :', round(end_time - start_time), '초')

# [목표] acc = 1.00 합격
# loss : 0.12950600683689117
# acc : 0.965
# accuracy_score : 0.9652294853963839
# 소요 시간 : 34 초

# ========== ========== ========== ========== ========== <- MinMaxScaler 적용 후

# loss : 0.12810786068439484
# acc : 0.974
# accuracy_score : 0.9735744089012517
# 소요 시간 : 34 초

# [결론] 0.9652 -> 0.9736. 예상대로 거의 변화가 없다.
#        digits 는 모든 컬럼이 '픽셀 밝기 0~16' 으로 이미 단위가 같아서
#        스케일링으로 바로잡을 컬럼 간 범위 차이가 애초에 없다.
#        x_test 540개 기준으로 521개 -> 526개, 5개 차이라 실행할 때마다 생기는 편차 범위다.
#        소요 시간도 34초로 동일하다.
#        효과가 없는 것을 확인한 대조군 역할의 데이터다.