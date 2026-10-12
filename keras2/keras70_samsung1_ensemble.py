# keras70_samsung1_ensemble.py
# 아모레 · 하이닉스 · 삼성전자 → 다음 거래일 삼성전자 시가 예측 (앙상블, LSTM + Conv1D)

# [목적] 제출용 원 코드 : 주가 CSV 3개로 앙상블 모델을 훈련하고 가중치를 _save/keras70/ 에 저장한다
#   - 데이터 : _data/samsung/ 의 삼성전자 · 아모레 · 하이닉스 CSV (최신 → 과거 순서, 원본 값은 그대로 사용)
#   - 시작 일자 : 가장 늦게 시작하는 삼성전자(2022/03/14) 기준 1120 거래일로 맞춘다 (세 파일 날짜 모두 일치)
#   - 컬럼 : 종목마다 시가 · 고가 · 저가 · 종가 · 거래량 · 금액(백만) 6개, x 만 StandardScaler (y 는 원 단위 그대로)
#   - split_x 로 20 거래일씩 잘라 x, 그 다음 날 삼성전자 시가를 y → (1100, 20, 6) / 마지막 20일은 예측용
#   - 모델 : 아모레 LSTM / 하이닉스 Conv1D 2층 / 삼성전자 Conv1D → LSTM 을 Concatenate 로 합쳐 시가 1개 출력
#   - keras70_samsung2_load_weights.py 가 이 가중치를 불러와 predict 만 한다

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
path = './_data/samsung/'
save_path = './_save/keras70/'

samsung = pd.read_csv(path + '삼성전자261012.csv', encoding='cp949', thousands=',')
amore = pd.read_csv(path + '아모레261012.csv', encoding='cp949', thousands=',')
hynix = pd.read_csv(path + '하이닉스261012.csv', encoding='cp949', thousands=',')

# 시작 일자 맞추기 : 삼성전자 기준 1120일 (2022/03/14 ~ 2026/10/12)
amore = amore[:1120]
hynix = hynix[:1120]

# 최신 → 과거 순서라 과거 → 최신으로 뒤집는다
samsung = samsung[::-1]
amore = amore[::-1]
hynix = hynix[::-1]

cols = ['시가', '고가', '저가', '종가', '거래량', '금액(백만)']
x1_data = amore[cols].values        # (1120, 6)
x2_data = hynix[cols].values        # (1120, 6)
x3_data = samsung[cols].values      # (1120, 6)
y_data = samsung['시가'].values     # (1120,)

scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
x1_data = scaler1.fit_transform(x1_data)
x2_data = scaler2.fit_transform(x2_data)
x3_data = scaler3.fit_transform(x3_data)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

size = 20

# 마지막 날은 정답(다음 날 시가) 이 없으니 빼고 자른다
x1 = split_x(x1_data[:-1], size)    # (1100, 20, 6)
x2 = split_x(x2_data[:-1], size)
x3 = split_x(x3_data[:-1], size)
y = y_data[size:]                   # (1100,)

# 예측용 : 마지막 20일
x1_pred = x1_data[-size:].reshape(1, size, 6)
x2_pred = x2_data[-size:].reshape(1, size, 6)
x3_pred = x3_data[-size:].reshape(1, size, 6)

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1, x2, x3, y,
    train_size=0.8,
    random_state=333,
)
print(x1_train.shape, x1_test.shape, y_train.shape, y_test.shape)

#2-1. 모델1 아모레 (LSTM)
input1 = Input(shape=(size, 6))
dense1 = LSTM(64, name='am1')(input1)
dense1 = Dense(32, activation='relu', name='am2')(dense1)
output1 = Dense(16, activation='relu', name='am3')(dense1)

#2-2. 모델2 하이닉스 (Conv1D)
input2 = Input(shape=(size, 6))
dense2 = Conv1D(64, 3, padding='same', activation='relu', name='hy1')(input2)
dense2 = Conv1D(32, 3, padding='same', activation='relu', name='hy2')(dense2)
dense2 = Flatten(name='hy3')(dense2)
output2 = Dense(16, activation='relu', name='hy4')(dense2)

#2-3. 모델3 삼성전자 (Conv1D + LSTM)
input3 = Input(shape=(size, 6))
dense3 = Conv1D(64, 3, padding='same', activation='relu', name='ss1')(input3)
dense3 = LSTM(64, name='ss2')(dense3)
dense3 = Dense(32, activation='relu', name='ss3')(dense3)
output3 = Dense(16, activation='relu', name='ss4')(dense3)

#2-4. 모델 병합
merge1 = Concatenate(name='mg1')([output1, output2, output3])
merge2 = Dense(64, activation='relu', name='mg2')(merge1)
merge3 = Dense(32, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2, input3], outputs=last_output)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
)

model.fit([x1_train, x2_train, x3_train], y_train,
          epochs=3000,
          batch_size=32,
          validation_split=0.2,
          callbacks=[es],
          )

model.save_weights(save_path + 'keras70_samsung.weights.h5')

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('loss :', loss)

y_test_pred = model.predict([x1_test, x2_test, x3_test])
print('r2 :', r2_score(y_test, y_test_pred))

y_pred = model.predict([x1_pred, x2_pred, x3_pred])
print('2026/10/13 삼성전자 시가 예측 :', y_pred)

# loss : 37825364.0
# r2 : 0.9907946586608887
# 2026/10/13 삼성전자 시가 예측 : [[259291.11]]