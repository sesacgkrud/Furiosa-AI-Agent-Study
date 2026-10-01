# keras32_MCP_load_04_dacon_ddareung.py
# https://dacon.io/competitions/open/235576/overview/description
# 구조 : keras30_ModelCheckPoint2_load.py 기준
# 짝 파일 : keras31_MCP_save_04_dacon_ddareung.py -> 반드시 그 파일을 먼저 실행해서 keras31_mcp4.keras 를 만들어야 한다.
#   불러온 모델로 평가한 값이 keras31 실행 결과와 소수점까지 같아야 한다.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import load_model
import datetime

path = "./_data/ddareung/"      # 데이터 폴더
path_save = './_save/keras31/'  # 모델을 불러올 폴더

#1. 데이터
# #1 은 keras31 과 완전히 같아야 한다. (결측치 처리, split 2번, random_state, scaler)
# 모델은 '그때 변환된 x' 에 맞춰 학습됐으므로 하나만 달라도 결과가 달라진다.
train_csv = pd.read_csv(path + "train.csv", index_col=0)

train_csv = train_csv.dropna()  # 빠지면 행 개수가 달라져서 split 이 keras31 과 다른 행을 x_test 로 뽑는다 (+ NaN 이 남는다)

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

# 1단계 : 전체 -> train(80%) / test(20%)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=222,
)

# 2단계 : x_val 은 안 쓰지만 이 split 은 반드시 있어야 한다.
# keras31 의 scaler 는 '2단계 뒤의 x_train(64%)' 으로 fit 했기 때문에, 빠지면 scaler 기준이 달라진다.
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    train_size=0.8,
    random_state=77,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # 훈련은 안 하지만 scaler 기준을 만들려면 x_train 으로 fit 해야 한다
x_test = scaler.transform(x_test)
# x_val 변환은 필요 없다 (훈련/검증을 하지 않고, x_test 결과에도 영향이 없음)

# keras31 출력과 같은 값이 찍혀야 한다. 다르면 #1 전처리가 어긋난 것
print('Min :', np.min(x_train), 'Max :', np.max(x_train))
print('Min :', np.min(x_test), 'Max :', np.max(x_test))

#2. 모델 불러오기 (#2 모델 구성 + #3 컴파일, 훈련 을 이 한 줄이 대신한다)
# .keras 파일에는 구조 + 가중치 + compile 정보가 같이 저장돼 있어서 Sequential / compile / fit 이 필요 없다.
model = load_model(path_save + 'keras31_mcp4.keras')

print("========== ========== ========== ========== ==========")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss(mse) :", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

#5. 모델 저장 (불러온 모델을 결과값이 들어간 이름으로 같은 폴더에 다시 저장)
# 불러온 모델 = keras31 에서 val_loss 가 가장 좋았던 epoch 의 모델이고, fit 을 안 했으니 가중치도 keras31_mcp4.keras 와 같다.
# 파일명 만들기는 keras30_ModelCheckPoint3 방식 : 폴더 + 접두어 + 날짜 + 결과값
date = datetime.datetime.now().strftime('%m%d_%H%M')   # 실행한 날짜_시간 (예: 0914_1850)
filepath = ''.join([path_save, 'k32_04_', date, '_', f'{loss:.4f}', '.keras'])   # x_test loss 를 소수 4자리로
# 예) ./_save/keras31/k32_04_0914_1850_0.1234.keras -> 이름만 보고 어떤 결과를 낸 모델인지 알 수 있다
model.save(filepath)
print('모델 저장 :', filepath)

# ===== 실행 결과 (2026-09-14, keras31 실행 직후 keras32 실행) =====
# loss(mse) : 2351.59130859375
# r2 : 0.6807679617592108
# mse : 2351.591344888818
# RMSE : 48.49320926571903
# -> keras31 (save) 실행 결과와 소수점까지 같다. (Min / Max 출력도 같음 = #1 전처리 동일)
# keras31 은 가중치 초기값을 고정하지 않아서 다시 실행하면 값이 달라지고 .keras 파일도 덮어써진다.
# 그러니 비교할 때는 항상 keras31 을 실행한 "직후" 에 keras32 를 실행할 것.
