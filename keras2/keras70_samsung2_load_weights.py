# keras70_samsung2_load_weights.py
# keras70_samsung1_ensemble.py 가중치 불러와서 predict 만 (모델 구조 · 데이터 처리는 동일해야 한다)

# [목적] 제출용 예측 코드 : 훈련 없이 저장된 가중치로 다음 거래일 삼성전자 시가를 예측한다
#   - load_weights 는 가중치 숫자만 불러오므로, 훈련 코드와 똑같은 모델 구조를 먼저 만들어 그 틀에 채운다
#   - 스케일러는 저장하지 않고 같은 CSV 를 같은 순서로 다시 fit → 훈련 때와 같은 입력이 만들어진다
#   - compile · fit 이 없어 가중치가 바뀌지 않으므로 여러 번 돌려도 같은 예측값이 나온다

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten, Concatenate

#1. 데이터
path = './_data/samsung/'
save_path = './_save/keras70/'

samsung = pd.read_csv(path + '삼성전자261012.csv', encoding='cp949', thousands=',')
amore = pd.read_csv(path + '아모레261012.csv', encoding='cp949', thousands=',')
hynix = pd.read_csv(path + '하이닉스261012.csv', encoding='cp949', thousands=',')

amore = amore[:1120]
hynix = hynix[:1120]

samsung = samsung[::-1]
amore = amore[::-1]
hynix = hynix[::-1]

cols = ['시가', '고가', '저가', '종가', '거래량', '금액(백만)']
x1_data = amore[cols].values
x2_data = hynix[cols].values
x3_data = samsung[cols].values

scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
x1_data = scaler1.fit_transform(x1_data)
x2_data = scaler2.fit_transform(x2_data)
x3_data = scaler3.fit_transform(x3_data)

size = 20

# 예측용 : 마지막 20일
x1_pred = x1_data[-size:].reshape(1, size, 6)
x2_pred = x2_data[-size:].reshape(1, size, 6)
x3_pred = x3_data[-size:].reshape(1, size, 6)

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
model.load_weights(save_path + 'keras70_samsung.weights.h5')

#4. 예측
y_pred = model.predict([x1_pred, x2_pred, x3_pred])
print('2026/10/13 삼성전자 시가 예측 :', y_pred)

# 2026/10/13 삼성전자 시가 예측 : [[259291.11]]
