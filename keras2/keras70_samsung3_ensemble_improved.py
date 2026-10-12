# keras70_samsung3_ensemble_improved.py (개선용 : 훈련 + 가중치 저장)
# keras70_samsung1_ensemble.py (제출용) 과 같은 문제를 조금 더 "안전한" 방법으로 푼 버전
#
# [문제]
#   아모레 · 하이닉스 (+ 삼성전자 과거) 데이터로 "다음 거래일 삼성전자 시가" 1개를 예측한다
#
# [제출용과 다른 점]
#   1) sort_values('일자')      : [::-1] 처럼 "파일이 최신순일 것" 이라고 가정하지 않고 날짜 기준으로 정렬한다
#   2) 날짜 기준으로 시작일 맞추기 : [:1120] 처럼 행 개수를 직접 세지 않고, 삼성전자 시작일 이후 날짜만 남긴다
#   3) assert 로 세 파일 날짜 확인 : 같은 행 = 같은 날짜인지 코드가 직접 검사한다 (다르면 바로 에러)
#   4) build_model() 함수         : 훈련 코드 · 예측 코드가 "똑같은 구조" 를 만들도록 한 곳에 묶는다
#   5) 시드 고정                  : 다시 훈련해도 결과가 최대한 비슷하게 나오게 한다
#   6) Adam(learning_rate=0.005)  : y 를 스케일링하지 않아 값이 크다 → 학습률을 키워 빨리 따라가게 한다
#
# [지켜야 할 것]
#   - CSV 원본 값은 바꾸지 않는다 (채우기 · 삭제 · 수정 없음, 읽기만 한다)
#   - y(삼성전자 시가) 는 스케일링 · 로그 변환 등 아무것도 하지 않는다 → 예측값이 바로 "원" 단위로 나온다

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 시드 고정 : 가중치 초기값 · train_test_split · 배치 섞기 순서가 매번 같아진다
#   (GPU 연산 특성상 완전히 같지는 않을 수 있다 → 그래서 최종 제출은 "가중치 파일" 로 재현한다)
SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#1. 데이터
path = './_data/samsung/'
save_path = './_save/keras70/'
os.makedirs(save_path, exist_ok=True)       # 저장 폴더가 없으면 만든다 (있으면 그냥 넘어간다)

# encoding='cp949' : 한글 엑셀(HTS) 에서 내려받은 CSV 라 utf-8 로 읽으면 글자가 깨진다
# thousands=','    : "260,000" 같은 문자열을 숫자 260000 으로 읽는다
#   → 메모리에서 읽는 방식만 바꾸는 것이고, CSV 파일 자체는 건드리지 않는다
samsung = pd.read_csv(path + '삼성전자261012.csv', encoding='cp949', thousands=',')
amore = pd.read_csv(path + '아모레261012.csv', encoding='cp949', thousands=',')
hynix = pd.read_csv(path + '하이닉스261012.csv', encoding='cp949', thousands=',')

# 날짜 정렬 : 파일은 최신 → 과거 순서다. split_x 는 "위 → 아래 = 과거 → 미래" 라고 보고 자르므로 반드시 뒤집어야 한다
#   뒤집지 않으면 "다음 날 시가" 가 아니라 "전날 시가" 를 맞추는 모델이 된다
#   sort_values('일자') 는 날짜 값을 보고 정렬하므로 파일 순서가 섞여 있어도 안전하다
#   ('2026/10/12' 처럼 년/월/일 자릿수가 고정된 문자열이라 문자열 정렬 = 날짜 정렬)
# reset_index(drop=True) : 정렬하면 행 번호(index) 가 1119, 1118 ... 처럼 거꾸로 남는다
#   → 0, 1, 2 ... 로 다시 매긴다. 아래 assert 에서 세 파일을 "행 번호끼리" 비교하기 때문에 필요하다
samsung = samsung.sort_values('일자').reset_index(drop=True)
amore = amore.sort_values('일자').reset_index(drop=True)
hynix = hynix.sort_values('일자').reset_index(drop=True)
print(samsung['일자'].min(), amore['일자'].min(), hynix['일자'].min())     # 2022/03/14 2018/05/16 2021/06/17

# 시작 일자 맞추기 : 세 파일 중 가장 늦게 시작하는 삼성전자(2022/03/14) 기준으로 아모레 · 하이닉스 앞부분을 잘라낸다
#   (행을 잘라내는 것뿐, 남은 값은 그대로다)
start_date = samsung['일자'].min()
amore = amore[amore['일자'] >= start_date].reset_index(drop=True)
hynix = hynix[hynix['일자'] >= start_date].reset_index(drop=True)

# 세 파일의 같은 행이 같은 날짜인지 확인 : 한 파일에만 있는 날짜(휴장일 차이 등) 가 있으면 여기서 멈춘다
#   → 이 데이터는 2022/03/14 이후 1120 거래일이 세 파일 모두 같아서 통과한다
assert (samsung['일자'] == amore['일자']).all() and (samsung['일자'] == hynix['일자']).all()
print(samsung.shape, amore.shape, hynix.shape)                             # (1120, 17) (1120, 17) (1120, 17)

# 컬럼 선택 : 종목마다 6개 (조건 : 5개 이상). 나머지(대비, 등락률, 신용비, 개인, 기관, 외인 ...) 는 버린다
cols = ['시가', '고가', '저가', '종가', '거래량', '금액(백만)']
x1_data = amore[cols].values.astype(np.float32)       # (1120, 6)  아모레
x2_data = hynix[cols].values.astype(np.float32)       # (1120, 6)  하이닉스
x3_data = samsung[cols].values.astype(np.float32)     # (1120, 6)  삼성전자 과거
y_data = samsung['시가'].values.astype(np.float32)    # (1120,)    정답 = 삼성전자 시가 원본 그대로

# 스케일링 : x 만 한다
#   종목마다 가격대(아모레 13만, 하이닉스 160만), 컬럼마다 단위(가격 vs 거래량 수백만) 가 달라서 맞춰준다
#   종목별로 스케일러를 따로 둔다 → 각 종목의 컬럼 기준(평균, 표준편차) 이 섞이지 않는다
#   split_x 로 자르기 "전" 2차원 상태에서 해야 reshape 없이 컬럼별로 바로 스케일링된다
#   y 는 스케일링하지 않는다 → 예측값이 바로 "원" 단위, inverse_transform 도 필요 없다
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
x1_data = scaler1.fit_transform(x1_data)
x2_data = scaler2.fit_transform(x2_data)
x3_data = scaler3.fit_transform(x3_data)

# split_x : 한 칸씩 밀면서 size 일씩 자른다 (2차원도 행 기준으로 잘리고 열 6개는 유지된다)
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa, dtype=np.float32)

size = 20       # 과거 20 거래일(약 한 달) 을 보고 다음 날 시가를 맞춘다

# 1120 일을 20 일씩 자르면 1120 - 20 + 1 = 1101 묶음
x1 = split_x(x1_data, size)     # (1101, 20, 6)
x2 = split_x(x2_data, size)
x3 = split_x(x3_data, size)

# 마지막 묶음(2026/09/14 ~ 2026/10/12) 은 정답(= 내일 시가) 이 아직 없다
#   → 훈련에서 빼고 예측 입력으로 쓴다
#   [-1:] 처럼 슬라이싱으로 꺼내면 (1, 20, 6) 3차원이 유지돼서 reshape 가 필요 없다 ([-1] 이면 (20, 6) 이 된다)
x1_pred, x2_pred, x3_pred = x1[-1:], x2[-1:], x3[-1:]    # (1, 20, 6)
x1, x2, x3 = x1[:-1], x2[:-1], x3[:-1]                   # (1100, 20, 6)

# y : i 번째 묶음(i ~ i+19 일) 의 정답 = i+20 일의 삼성전자 시가
#   y_data[20:] 은 21번째 날부터 → 첫 묶음(0 ~ 19일) 의 정답이 20일 시가와 짝이 된다
#   개수 1120 - 20 = 1100 → x 묶음 수와 같다
y = y_data[size:]                                        # (1100,)
print(x1.shape, x2.shape, x3.shape, y.shape, x1_pred.shape)

# train / test 분리 : 입력 3개 + 정답 1개를 한 번에 넣으면 같은 행끼리 같은 위치로 섞인다
#   shuffle 은 "묶음 단위" 라서 묶음 안의 20일 순서는 그대로 유지된다
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1, x2, x3, y,
    train_size=0.8,
    random_state=SEED,
)
print(x1_train.shape, x1_test.shape, y_train.shape, y_test.shape)   # (880, 20, 6) (220, 20, 6) (880,) (220,)

#2. 모델 구성
# build_model() : 모델 만드는 코드를 함수로 묶는다
#   load_weights 는 가중치만 불러오므로, 예측 코드에서도 "층 종류 · 노드 수 · 순서" 가 완전히 같은 모델이 필요하다
#   함수로 묶어 두면 이 부분만 그대로 복사하면 되고, 한 곳만 고치다가 구조가 어긋나는 실수를 막는다
def build_model():
    #2-1. 모델1 아모레 (LSTM) : 시간 순서를 따라가며 흐름을 기억
    input1 = Input(shape=(size, len(cols)), name='amore')                  # (None, 20, 6)
    dense1 = LSTM(64, name='am1')(input1)                                  # (None, 64)
    dense1 = Dense(32, activation='relu', name='am2')(dense1)
    output1 = Dense(16, activation='relu', name='am3')(dense1)

    #2-2. 모델2 하이닉스 (Conv1D) : 커널 3 이 시간축(20일) 을 따라 움직이며 3일 단위 패턴을 본다
    #   padding='same' 이라 timesteps 20 이 줄지 않는다
    input2 = Input(shape=(size, len(cols)), name='hynix')                  # (None, 20, 6)
    dense2 = Conv1D(64, kernel_size=3, padding='same', activation='relu', name='hy1')(input2)    # (None, 20, 64)
    dense2 = Conv1D(32, kernel_size=3, padding='same', activation='relu', name='hy2')(dense2)    # (None, 20, 32)
    dense2 = Flatten(name='hy3')(dense2)                                   # (None, 640)
    output2 = Dense(16, activation='relu', name='hy4')(dense2)

    #2-3. 모델3 삼성전자 과거 (Conv1D → LSTM) : Conv1D 로 짧은 패턴을 뽑고, LSTM 으로 그 흐름을 본다
    #   Conv1D 출력이 3차원 (None, 20, 64) 이라 그대로 LSTM 에 넣을 수 있다
    input3 = Input(shape=(size, len(cols)), name='samsung')                # (None, 20, 6)
    dense3 = Conv1D(64, kernel_size=3, padding='same', activation='relu', name='ss1')(input3)    # (None, 20, 64)
    dense3 = LSTM(64, name='ss2')(dense3)                                  # (None, 64)
    dense3 = Dense(32, activation='relu', name='ss3')(dense3)
    output3 = Dense(16, activation='relu', name='ss4')(dense3)

    #2-4. 모델 병합 : 세 모델의 출력 16 + 16 + 16 = 48 개를 이어 붙인다
    merge1 = Concatenate(name='mg1')([output1, output2, output3])          # (None, 48)
    merge2 = Dense(64, activation='relu', name='mg2')(merge1)
    merge3 = Dense(32, activation='relu', name='mg3')(merge2)
    last_output = Dense(1, name='last')(merge3)       # 다음 거래일 삼성전자 시가 1개 (회귀라 activation 없음 = linear)

    return Model(inputs=[input1, input2, input3], outputs=last_output)

model = build_model()
model.summary()

#3. 컴파일, 훈련
# y 를 스케일링하지 않아 값이 10만 ~ 27만 원이다
#   출력층은 0 근처에서 시작하므로 기본 학습률(0.001) 로는 그 크기까지 올라가는 데 epoch 이 오래 걸린다
#   → 학습률을 0.005 로 키운다
# mae : mse 는 원² 단위라 감이 안 온다 → 평균 몇 원 틀렸는지 같이 본다
model.compile(loss='mse', optimizer=Adam(learning_rate=0.005), metrics=['mae'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,      # 끝난 시점이 아니라 val_loss 가 가장 좋았던 가중치로 되돌린다
)

model.fit([x1_train, x2_train, x3_train], y_train,
          epochs=2000,
          batch_size=32,
          validation_split=0.2,
          callbacks=[es],
          verbose=2,                # epoch 마다 한 줄만 출력 (진행 막대 생략)
          )

# 가중치만 저장 : restore_best_weights 로 되돌린 최적 가중치가 저장된다
#   제출용 가중치(keras70_samsung.weights.h5) 를 덮어쓰지 않도록 파일 이름을 다르게 한다
model.save_weights(save_path + 'keras70_samsung_improved.weights.h5')

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test, verbose=0)
print('loss(mse), mae :', loss)

y_test_pred = model.predict([x1_test, x2_test, x3_test], verbose=0).reshape(-1)
print('r2 :', r2_score(y_test, y_test_pred))

# 예측 : 마지막 20 거래일(~2026/10/12) → 다음 거래일(2026/10/13) 시가
y_pred = model.predict([x1_pred, x2_pred, x3_pred], verbose=0)     # (1, 1)
print('마지막 입력 일자 :', samsung['일자'].iloc[-1])
print('다음 거래일 삼성전자 시가 예측 :', round(float(y_pred[0, 0]), 2), '원')
