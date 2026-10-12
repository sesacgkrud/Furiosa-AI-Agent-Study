# keras70_samsung4_load_weights_improved.py (개선용 : 가중치 불러와서 predict 만)
# keras70_samsung3_ensemble_improved.py 가 저장한 _save/keras70/keras70_samsung_improved.weights.h5 를 불러온다
#
# [같은 예측값이 나오는 조건]
#   1) 모델 구조가 훈련 코드와 완전히 같아야 한다 → build_model() 을 그대로 복사
#      load_weights 는 "가중치 숫자" 만 불러오므로, 층이 하나라도 다르면 불러오는 순간 에러가 난다
#   2) 입력 데이터가 훈련 때와 같은 방식으로 만들어져야 한다
#      스케일러를 파일로 저장하지 않았으므로, 같은 CSV 를 같은 순서로 읽어 다시 fit 한다
#      → 데이터가 같으면 평균 · 표준편차도 같으니 스케일링 결과가 똑같다
#   3) fit 을 하지 않는다 → 가중치가 바뀌지 않으므로 몇 번을 돌려도 같은 값이 나온다
#   ※ GPU 와 CPU 는 LSTM 계산 방식이 달라 소수점 아래 ~ 수십 원 정도 차이가 날 수 있다

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Flatten, Concatenate

#1. 데이터 (훈련 코드와 동일한 순서 · 방식)
path = './_data/samsung/'
save_path = './_save/keras70/'

samsung = pd.read_csv(path + '삼성전자261012.csv', encoding='cp949', thousands=',')
amore = pd.read_csv(path + '아모레261012.csv', encoding='cp949', thousands=',')
hynix = pd.read_csv(path + '하이닉스261012.csv', encoding='cp949', thousands=',')

# 최신 → 과거 순서를 과거 → 최신 순서로 정렬하고 행 번호를 0 부터 다시 매긴다
samsung = samsung.sort_values('일자').reset_index(drop=True)
amore = amore.sort_values('일자').reset_index(drop=True)
hynix = hynix.sort_values('일자').reset_index(drop=True)

# 시작 일자를 삼성전자(2022/03/14) 기준으로 맞추고, 세 파일 날짜가 행마다 같은지 확인
start_date = samsung['일자'].min()
amore = amore[amore['일자'] >= start_date].reset_index(drop=True)
hynix = hynix[hynix['일자'] >= start_date].reset_index(drop=True)
assert (samsung['일자'] == amore['일자']).all() and (samsung['일자'] == hynix['일자']).all()

cols = ['시가', '고가', '저가', '종가', '거래량', '금액(백만)']
x1_data = amore[cols].values.astype(np.float32)       # (1120, 6)
x2_data = hynix[cols].values.astype(np.float32)
x3_data = samsung[cols].values.astype(np.float32)

# 훈련 코드와 같은 데이터 전체로 fit → 같은 기준(평균, 표준편차) 으로 변환된다
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
x1_data = scaler1.fit_transform(x1_data)
x2_data = scaler2.fit_transform(x2_data)
x3_data = scaler3.fit_transform(x3_data)

size = 20

# 예측 입력은 마지막 20 거래일 한 묶음뿐이다
#   훈련 코드의 split_x 결과 마지막 묶음 x1[-1:] 과 같은 값이라 split_x 전체를 다시 만들 필요가 없다
#   모델은 3차원 (묶음 수, 20, 6) 을 받으므로 (20, 6) → (1, 20, 6) 으로 reshape
x1_pred = x1_data[-size:].reshape(1, size, len(cols))
x2_pred = x2_data[-size:].reshape(1, size, len(cols))
x3_pred = x3_data[-size:].reshape(1, size, len(cols))

#2. 모델 구성 (훈련 코드의 build_model() 그대로)
def build_model():
    #2-1. 모델1 아모레 (LSTM)
    input1 = Input(shape=(size, len(cols)), name='amore')
    dense1 = LSTM(64, name='am1')(input1)
    dense1 = Dense(32, activation='relu', name='am2')(dense1)
    output1 = Dense(16, activation='relu', name='am3')(dense1)

    #2-2. 모델2 하이닉스 (Conv1D)
    input2 = Input(shape=(size, len(cols)), name='hynix')
    dense2 = Conv1D(64, kernel_size=3, padding='same', activation='relu', name='hy1')(input2)
    dense2 = Conv1D(32, kernel_size=3, padding='same', activation='relu', name='hy2')(dense2)
    dense2 = Flatten(name='hy3')(dense2)
    output2 = Dense(16, activation='relu', name='hy4')(dense2)

    #2-3. 모델3 삼성전자 과거 (Conv1D → LSTM)
    input3 = Input(shape=(size, len(cols)), name='samsung')
    dense3 = Conv1D(64, kernel_size=3, padding='same', activation='relu', name='ss1')(input3)
    dense3 = LSTM(64, name='ss2')(dense3)
    dense3 = Dense(32, activation='relu', name='ss3')(dense3)
    output3 = Dense(16, activation='relu', name='ss4')(dense3)

    #2-4. 모델 병합
    merge1 = Concatenate(name='mg1')([output1, output2, output3])
    merge2 = Dense(64, activation='relu', name='mg2')(merge1)
    merge3 = Dense(32, activation='relu', name='mg3')(merge2)
    last_output = Dense(1, name='last')(merge3)

    return Model(inputs=[input1, input2, input3], outputs=last_output)

model = build_model()
model.load_weights(save_path + 'keras70_samsung_improved.weights.h5')

#4. 예측
# predict 만 하므로 compile 은 필요 없다 (evaluate 로 loss 를 계산할 때만 compile 이 필요)
y_pred = model.predict([x1_pred, x2_pred, x3_pred], verbose=0)     # (1, 1)
print('마지막 입력 일자 :', samsung['일자'].iloc[-1])
print('다음 거래일 삼성전자 시가 예측 :', round(float(y_pred[0, 0]), 2), '원')
