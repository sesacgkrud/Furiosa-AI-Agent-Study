# keras13_ddareung02_submit.py
# keras13_ddareung01 에 제출(submission) 파일 만들기까지 추가한다
# 제출용 test_csv 는 정답(count)이 없어서 evaluate 를 못 쓴다 -> predict 결과를 submission 의 count 칸에 넣어 저장한다
# 결측치 처리 2번째 방법 : fillna(평균값) -> test_csv 는 행을 지우면 제출 개수(715)가 안 맞으므로 지우지 않고 채운다

# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = "./_data/ddareung/" # 상대경로, 데이터가 있는 폴더 경로를 path 변수에 저장

train_csv = pd.read_csv(path + "train.csv", index_col=0) # 데이터를 pandas 데이터 형태로 리턴, index_col=0 -> 첫번째 컬럼 = index 컬럼
print(train_csv) # [1459 rows x 11 columns] -> index(id열) 제거 후 -> [1459 rows x 10 columns]
# count(y값) 있음
# 훈련시키기 위해 x와 y 분리

# 제출용 (test_csv, submission)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [715 rows x 9 columns]
# count(y값) 없음

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission) # [715 rows x 1 columns]

print(train_csv.shape) # (1459, 10) -> (None, 10)
print(test_csv.shape) # (715, 9) evaluate 사용 불가, test 데이터로 submission을 맞추야 하기 위한 파일
print(submission.shape) # (715, 1)

print(train_csv.columns)
print(train_csv.info()) # 많은 결측치 존재

print(test_csv.info())

# exit() # 여기까지만 실행 후 중단, 이후 코드 실행X

########## 결측치 처리 1. 삭제 ##########

train_csv = train_csv.dropna()
print(train_csv) # [1328 rows x 10 columns]

# ★★★ train_csv를 x와 y로 분리 ★★★
x = train_csv.drop(['count'], axis=1)
print(x)

y = train_csv['count']
print(y)
print(y.shape) # (1328,) -> 벡터 형태로 출력

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=33,
)

############# submit 물밑 작업 #############
print(test_csv.info())


######### 결측치 처리 2. 평균값 넣기 #########
test_csv = test_csv.fillna(test_csv.mean()) # pandas dataset
print(test_csv.info()) # (715, 9)
print(test_csv.shape) # (715, 9)

# exit()
########## ########## ########## ##########

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=9, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2000, batch_size=16)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
loss = model.evaluate(x_test, y_test, )

print("loss(mse) :", loss)
y_predict = model.predict(x_test).ravel() # 지표가 잘 맞는지 확인하기 위함


from sklearn.metrics import r2_score, mean_squared_error # 평가할 때 주로 사용
# scikit-learn에서 R²(결정계수)를 계산하는 함수를 가져옴

r2 = r2_score(y_test, y_predict) # 원값과 예측값 비교 -> loss

print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict): #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict)) # MSE에 루트 씌움

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# r2 : 0.7157199435830112
# mse : 2126.0653703580174
# RMSE : 46.10927640245526



##### submission.csv 만들기 // count 컬럼에 값 넣기 #####
print(submission)
y_submit = model.predict(test_csv)

submission['count'] = y_submit # 최종 weight에 test_csv 적용?
print(submission) # 최종 정답지
print(submission.shape) # (715, 1)

submission.to_csv(path + "submit/" + "submit_0904_1142.csv")

## 2번째
# r2 : 0.7206033808088118
# mse : 2032.4276956930526
# RMSE : 45.0824544107023

## 3번째
# r2 : 0.6956768637368235
# mse : 2021.4625755786747
# RMSE : 44.96067810407973

## 5번째
# r2 : 0.6853760077493756
# mse : 1853.6822498491663
# RMSE : 43.05441034144082