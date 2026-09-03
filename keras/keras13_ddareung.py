# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = "./_data/ddareung/" # 데이터가 있는 폴더 경로를 path 변수에 저장
train_csv = pd.read_csv(path + "train.csv", index_col=0) # 데이터를 pandas 데이터 형태로 리턴, index_col=0 -> 첫번째 컬럼 = index 컬럼
# print(train_csv) # [1459 rows x 11 columns] -> index(id열) 제거 후 -> [1459 rows x 10 columns]
# count(y값) 있음
# 훈련시키기 위해 x와 y 분리

# 제출용 (test_csv, submission)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [715 rows x 9 columns]
# count(y값) 없음

submission = pd.read_csv(path + "submission.csv", index_col=0)
# print(submission) # [715 rows x 1 columns]

# print(train_csv.shape) # (1459, 10)
# print(test_csv.shape) # (715, 9)
# print(submission.shape) # (715, 1)

# print(train_csv.columns)
'''
Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
      dtype='str')
'''

# print(train_csv.info())
'''
<class 'pandas.DataFrame'>
Index: 1459 entries, 3 to 2179
Data columns (total 10 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   hour                    1459 non-null   int64  
 1   hour_bef_temperature    1457 non-null   float64
 2   hour_bef_precipitation  1457 non-null   float64
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64
 9   count                   1459 non-null   float64
dtypes: float64(9), int64(1)
memory usage: 125.4 KB
None
'''

# print(test_csv.info())
'''
<class 'pandas.DataFrame'>
Index: 715 entries, 0 to 2177
Data columns (total 9 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   hour                    715 non-null    int64  
 1   hour_bef_temperature    714 non-null    float64
 2   hour_bef_precipitation  714 non-null    float64
 3   hour_bef_windspeed      714 non-null    float64
 4   hour_bef_humidity       714 non-null    float64
 5   hour_bef_visibility     714 non-null    float64
 6   hour_bef_ozone          680 non-null    float64
 7   hour_bef_pm10           678 non-null    float64
 8   hour_bef_pm2.5          679 non-null    float64
dtypes: float64(8), int64(1)
memory usage: 55.9 KB
None
'''

# exit() # 여기까지만 실행 후 중단, 이후 코드 실행X

########## 결측치 처리 1. 삭제 ##########

train_csv = train_csv.dropna()
# print(train_csv) # [1328 rows x 10 columns]

# ★★★ train_csv를 x와 y로 분리 ★★★
x = train_csv.drop(['count'], axis=1)
print(x)
'''
id                                                                            ...                                                                    
3       20                  16.3                     1.0                 1.5  ...                576.0           0.027           76.0            33.0
6       13                  20.1                     0.0                 1.4  ...                916.0           0.042           73.0            40.0
7        6                  13.9                     0.0                 0.7  ...               1382.0           0.033           32.0            19.0
8       23                   8.1                     0.0                 2.7  ...                946.0           0.040           75.0            64.0
9       18                  29.5                     0.0                 4.8  ...               2000.0           0.057           27.0            11.0
...    ...                   ...                     ...                 ...  ...                  ...             ...            ...             ...
2174     4                  16.8                     0.0                 1.6  ...               2000.0           0.031           37.0            27.0
2175     3                  10.8                     0.0                 3.8  ...               2000.0           0.039           34.0            19.0
2176     5                  18.3                     0.0                 1.9  ...               2000.0           0.009           30.0            21.0
2178    21                  20.7                     0.0                 3.7  ...               1395.0           0.082           71.0            36.0
2179    17                  21.1                     0.0                 3.1  ...               1973.0           0.046           38.0            17.0

[1328 rows x 9 columns]
'''

y = train_csv['count']
# print(y)
'''
id
3        49.0
6       159.0
7        26.0
8        57.0
9       431.0
        ...  
2174     21.0
2175     20.0
2176     22.0
2178    216.0
2179    170.0
Name: count, Length: 1328, dtype: float64
'''

# print(y.shape) # (1328,) -> 벡터 형태로 출력

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=222,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=16)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
loss = model.evaluate(x_test, y_test, )

print("loss(mse) :", loss)
y_predict = model.predict(x_test).ravel()

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