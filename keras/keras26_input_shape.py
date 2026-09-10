# keras23_softmax1_OneHot_iris.py 베이스

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris() # load_iris()를 실행하면 Iris 데이터셋이 들어있는 객체가 만들어진다.
print(datasets)
print(datasets.data) # -> 입력 데이터(x) dictionary
print(datasets.target) #  -> 정답 데이터(y) list

print(datasets.DESCR) # # Iris 데이터셋에 대한 설명을 출력
print(datasets.feature_names) # 입력 데이터의 컬럼 이름을 출력
# ['sepal length (cm)',
#  'sepal width (cm)',
#  'petal length (cm)',
#  'petal width (cm)']

x = datasets.data # 문제지에 해당하는 입력 데이터 -> Iris 데이터의 꽃받침/꽃잎 길이와 너비 4개를 사용
y = datasets['target'] # 정답 데이터 -> Iris는 3개의 종류를 분류
# 0 -> setosa
# 1 -> versicolor
# 2 -> virginica

# print(x.shape) # (150, 4) 총 150개의 데이터가 있고 한 데이터당 feature가 4개 있음
# print(y.shape) # (150,) 데이터 150개의 정답이 각각 숫자 하나로 들어있음
# print(y) # [0 0 0 0 0 ... 1 1 1 ... 2 2 2] -> 현재 정답은 숫자 0, 1, 2로 되어 있음

# print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([50, 50, 50])) -> 0, 1, 2라는 클래스가 존재하고 각각 50개씩 있음 (pandas의 value_counts()와 비슷한 역할)


'''
OneHot Encoding은 모델을 구성하거나 훈련하는 작업이 아니라,
학습에 사용할 정답 데이터 y의 형태를 바꾸는 전처리 과정이기 때문에
#1. 데이터 에서 처리하는 게 맞음
'''
########## OneHot Encoding 1. (tensorflow) ##########
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print('\n===== to_categorical One-Hot Encoding =====')
# print(y)
# print(y.shape) # (150, 3)


########## OneHot Encoding 2. (pandas) ##########
# y_pandas = pd.DataFrame({'target': y}) # y를 DataFrame으로 변환
# y = pd.get_dummies(y_pandas['target']).values # get_dummies()를 이용한 One-Hot Encoding

# print('\n===== pandas One-Hot Encoding =====')
# print(y)
# print('y.shape :', y.shape)  # (150, 3)

########## OneHot Encoding 2. (pandas) - 선생님 코드 ##########
# y = pd.get_dummies(y, dtype=int)
# print(y)


########## OneHot Encoding 3. (sklearn) ##########
from sklearn.preprocessing import OneHotEncoder

# sklearn OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

# y = ohe.fit_transform(y) # 1차원 형태 (벡터 형태)

# 2차원 형태로 넣어야 함
# y = y.reshape(150,1) # (150,) -> (150,1)
# print(y, y.shape) # (150,1)
y = y.reshape(-1, 1) # 가장 마지막 값까지

# One-Hot Encoding
y = ohe.fit_transform(y)

print('\n===== sklearn One-Hot Encoding =====')
print(y)
print('y.shape :', y.shape)  # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=77,
    shuffle=True,
    stratify=y,
)

print(x_train.shape, x_test.shape) # (120, 4) (30, 4)
print(y_train.shape, y_test.shape) # (120, 3) (30, 3)

#2. 모델 구성
model = Sequential()
# model.add(Dense(30, input_dim=4, activation='relu')) # 뉴런 30개, 입력 데이터 feature 4개, 이 층의 활성화 함수로 ReLU 사용
model.add(Dense(30, input_shape=(4,), activation='relu')) # 뉴런 30개, 입력 데이터 feature 4개, 이 층의 활성화 함수로 ReLU 사용
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) # Iris의 클래스가 3개이기 때문에 3, ex) [2.3, 0.8, -1.2] -> 이런 형식으로 출력 (각 클래스일 확률) -- softmax --> [0.80, 0.18, 0.02]

'''
원데이터   -> input_shape
(n,4)     -> (4,)
(n,100,3) -> (100,3)
(n,100,100,3) -> (100,100,3)
이상 없음
'''

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # categorical_crossentropy -> 다중 분류에서 One-Hot Encoding된 정답과 softmax의 예측 결과를 비교해서 오차를 계산하는 손실 함수
es = EarlyStopping( # 훈련을 언제 멈출지 결정
    monitor='val_loss', # 검증 데이터의 loss를 감시
    mode='auto',
    patience=20, # val_loss가 좋아지지 않는 상태가 20번 정도 계속되면 훈련 중단
    restore_best_weights=True, # 훈련 중 가장 좋았던 시점의 가중치를 다시 사용
)
start_time = time.time() # 훈련 시작 시간 기록
model.fit(x_train, y_train, # 신경망이 학습하는 부분 (x_train: 공부할 문제, y_train: 문제의 정답)
          epochs=1000,
          batch_size=16, # 훈련 데이터를 한 번에 16개씩 처리
          verbose=1,
          validation_split=0.3, # x_train, y_train 중 30%를 검증용으로 사용 (test 데이터는 여기에 포함X)
          callbacks=[es], # EarlyStopping을 실제 훈련에 적용
          )
end_time = time.time() # 훈련 끝난 시간 기록

#4. 평가 예측
result = model.evaluate(x_test, y_test, )
# print('loss :', loss) # compile에 있는 metrics=['acc'] 코드 때문에 값 2개 출력 / 0번째 진짜 loss, 1번째 accuracy

print('loss :', result[0])
print('acc :', round(result[1],3))

y_predict = model.predict(x_test) # test 데이터의 정답을 모델에게 알려주지 않고, 입력값만 넣어서 예측

# print(y_predict) # y_predict는 아직 0, 1, 2가 아님(현재 확률값)

y_test_arg = np.argmax(y_test, axis=1) # argmax : 확률값 -> 클래스 번호로 변경 (y_test는 One-Hot 형태로 가장 큰 값 위치를 찾음)
y_predict_arg = np.argmax(y_predict, axis=1) # 예측도 위와 동일

accuracy_score = accuracy_score(y_test_arg, y_predict_arg) # 실제 정답과 모델 예측 비교 (accuracy_score: 모델 예측 / 실제 정답)
print('accuracy_score :', accuracy_score)
print('소요 시간 :', round(end_time - start_time), '초')

# loss : 0.07295681536197662
# acc : 0.978
# accuracy_score : 0.9777777777777777
# 소요 시간 : 8 초