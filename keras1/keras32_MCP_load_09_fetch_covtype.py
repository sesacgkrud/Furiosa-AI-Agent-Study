# keras32_MCP_load_09_fetch_covtype.py
# 구조 : keras30_ModelCheckPoint2_load.py 기준
# 짝 파일 : keras31_MCP_save_09_fetch_covtype.py -> 반드시 그 파일을 먼저 실행해서 keras31_mcp9.keras 를 만들어야 한다.
#   불러온 모델로 평가한 값이 keras31 실행 결과와 소수점까지 같아야 한다.
# [참고] 훈련을 안 하므로 keras31 과 달리 몇 초면 끝난다

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import load_model
import datetime
from tensorflow.keras.utils import to_categorical

path = './_save/keras31/'

#1. 데이터
# #1 은 keras31 과 완전히 같아야 한다. (원핫, split 비율, random_state, stratify, scaler)
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)

y = to_categorical(y)   # (581012, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42,
    shuffle=True,
    stratify=y,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # 훈련은 안 하지만 scaler 기준을 만들려면 x_train 으로 fit 해야 한다
x_test = scaler.transform(x_test)

# keras31 출력과 같은 값이 찍혀야 한다. 다르면 #1 전처리가 어긋난 것
print('Min :', np.min(x_train), 'Max :', np.max(x_train))
print('Min :', np.min(x_test), 'Max :', np.max(x_test))

#2. 모델 불러오기 (#2 모델 구성 + #3 컴파일, 훈련 을 이 한 줄이 대신한다)
# .keras 파일에는 구조 + 가중치 + compile 정보가 같이 저장돼 있어서 Sequential / compile / fit 이 필요 없다.
model = load_model(path + 'keras31_mcp9.keras')

print("========== ========== ========== ========== ==========")

#4. 평가 예측
result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('acc :', round(result[1], 3))

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

y_test_arg = np.argmax(y_test, axis=1)
y_predict_arg = np.argmax(y_predict, axis=1)
acc_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', acc_score)

#5. 모델 저장 (불러온 모델을 결과값이 들어간 이름으로 같은 폴더에 다시 저장)
# 불러온 모델 = keras31 에서 val_loss 가 가장 좋았던 epoch 의 모델이고, fit 을 안 했으니 가중치도 keras31_mcp9.keras 와 같다.
# 파일명 만들기는 keras30_ModelCheckPoint3 방식 : 폴더 + 접두어 + 날짜 + 결과값
date = datetime.datetime.now().strftime('%m%d_%H%M')   # 실행한 날짜_시간 (예: 0914_1850)
filepath = ''.join([path, 'k32_09_', date, '_', f'{result[0]:.4f}', '.keras'])   # x_test loss 를 소수 4자리로
# 예) ./_save/keras31/k32_09_0914_1850_0.1234.keras -> 이름만 보고 어떤 결과를 낸 모델인지 알 수 있다
model.save(filepath)
print('모델 저장 :', filepath)

# ===== 실행 결과 (2026-09-14, keras31 실행 직후 keras32 실행) =====
# loss : 0.1659817397594452
# acc : 0.943
# r2 : 0.7335482841185299
# mse : 0.010635867259575785
# RMSE : 0.1031303411202338
# accuracy_score : 0.9426978153111805
# -> keras31 (save) 실행 결과와 소수점까지 같다. (Min / Max 출력도 같음 = #1 전처리 동일)
# keras31 은 가중치 초기값을 고정하지 않아서 다시 실행하면 값이 달라지고 .keras 파일도 덮어써진다.
# 그러니 비교할 때는 항상 keras31 을 실행한 "직후" 에 keras32 를 실행할 것.
