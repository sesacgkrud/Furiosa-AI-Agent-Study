# keras32_MCP_load_07_santander.py
# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
# 구조 : keras30_ModelCheckPoint2_load.py 기준
# 짝 파일 : keras31_MCP_save_07_santander.py -> 반드시 그 파일을 먼저 실행해서 keras31_mcp7.keras 를 만들어야 한다.
#   불러온 모델로 평가한 값이 keras31 실행 결과와 소수점까지 같아야 한다.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import load_model
import datetime
from tensorflow.keras.utils import to_categorical

path = './_data/kaggle_santander/'  # 데이터 폴더
path_save = './_save/keras31/'      # 모델을 불러올 폴더

#1. 데이터
# #1 은 keras31 과 완전히 같아야 한다. (원핫, split 비율, random_state, stratify, scaler)
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=777,
    stratify=np.argmax(y, axis=1),
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # 훈련은 안 하지만 scaler 기준을 만들려면 x_train 으로 fit 해야 한다
x_test = scaler.transform(x_test)

# keras31 출력과 같은 값이 찍혀야 한다. 다르면 #1 전처리가 어긋난 것
print('Min :', np.min(x_train), 'Max :', np.max(x_train))
print('Min :', np.min(x_test), 'Max :', np.max(x_test))

#2. 모델 불러오기 (#2 모델 구성 + #3 컴파일, 훈련 을 이 한 줄이 대신한다)
# .keras 파일에는 구조 + 가중치 + compile 정보가 같이 저장돼 있어서 Sequential / compile / fit 이 필요 없다.
model = load_model(path_save + 'keras31_mcp7.keras')

print("========== ========== ========== ========== ==========")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :', round(loss[1], 4))

y_predict = model.predict(x_test)

# 확률 기준 r2 / mse / rmse (참고용, save / load 값 비교용)
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

y_predict = np.argmax(y_predict, axis=1)
y_test_arg = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test_arg, y_predict)
print('acc_score :', acc_score)

#5. 모델 저장 (불러온 모델을 결과값이 들어간 이름으로 같은 폴더에 다시 저장)
# 불러온 모델 = keras31 에서 val_loss 가 가장 좋았던 epoch 의 모델이고, fit 을 안 했으니 가중치도 keras31_mcp7.keras 와 같다.
# 파일명 만들기는 keras30_ModelCheckPoint3 방식 : 폴더 + 접두어 + 날짜 + 결과값
date = datetime.datetime.now().strftime('%m%d_%H%M')   # 실행한 날짜_시간 (예: 0914_1850)
filepath = ''.join([path_save, 'k32_07_', date, '_', f'{loss[0]:.4f}', '.keras'])   # x_test loss 를 소수 4자리로
# 예) ./_save/keras31/k32_07_0914_1850_0.1234.keras -> 이름만 보고 어떤 결과를 낸 모델인지 알 수 있다
model.save(filepath)
print('모델 저장 :', filepath)

# ===== 실행 결과 (2026-09-14, keras31 실행 직후 keras32 실행) =====
# loss : 0.24074189364910126
# acc : 0.9112
# r2 : 0.23876517495920896
# mse : 0.06880530055315723
# RMSE : 0.2623076448622061
# acc_score : 0.91125
# -> keras31 (save) 실행 결과와 소수점까지 같다. (Min / Max 출력도 같음 = #1 전처리 동일)
# keras31 은 가중치 초기값을 고정하지 않아서 다시 실행하면 값이 달라지고 .keras 파일도 덮어써진다.
# 그러니 비교할 때는 항상 keras31 을 실행한 "직후" 에 keras32 를 실행할 것.
