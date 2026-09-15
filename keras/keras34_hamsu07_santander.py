# keras34_hamsu07_santander.py
# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
# 구조 : keras34_hamsu00.py (함수형 모델) 기준 / 데이터, 모델 : keras33_dropout07_santander.py 를 함수형으로 변환
# 함수형 모델 : Input 으로 입력층을 만들고, '층(이전 층)' 형태로 하나씩 연결한 뒤 Model(inputs, outputs) 로 범위를 정한다
#               층 구성이 같으면 Sequential 과 파라미터 수도 같은 똑같은 모델이다 (만드는 방법만 다르다)
#               단, 같은 것은 '구조(Total params)' 다. 가중치 초기값 / Dropout 이 끄는 노드 / 배치 섞기가 매번 랜덤이라
#               시드를 고정하지 않으면 같은 코드를 두 번 돌려도 loss, acc 는 조금씩 다르게 나온다
#               (random_state 는 train_test_split 의 데이터 분할만 고정한다)
# Dropout : 훈련할 때마다 층 출력의 일부 노드를 랜덤으로 꺼서 특정 노드에만 의존하지 않게 한다 -> 과적합 방지
#           evaluate / predict 때는 자동으로 꺼지고 모든 노드를 다 사용한다

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Model                   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input   # Input -> 함수형 모델의 입력층
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

path = './_data/kaggle_santander/'  # 데이터 폴더
path_save = './_save/keras34/'      # 모델 저장 폴더

#1. 데이터
# 제출 파일을 만들지 않으므로 test.csv / sample_submission.csv 는 읽지 않는다
# (제출 저장 코드를 두면 실행할 때마다 keras28 에서 만든 submit_0910_1724_scaler.csv 를 덮어쓴다)
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)  # var_0 ~ var_199 (200개)
y = train_csv['target']                 # 0 / 1

num_classes = len(np.unique(y))                 # 2
y = to_categorical(y, num_classes=num_classes)  # 원핫 : 0 -> [1, 0], 1 -> [0, 1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=777,
    stratify=np.argmax(y, axis=1),  # 원핫을 다시 0 / 1 로 바꿔서 비율 기준으로 사용
)

scaler = RobustScaler()                 # keras28 에서 4종 비교 후 RobustScaler 로 통일
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

print('Min :', np.min(x_train), 'Max :', np.max(x_train))
print('Min :', np.min(x_test), 'Max :', np.max(x_test))

#2. 모델 구성 (함수형) - keras33 과 층 / Dropout 위치 / 출력층 activation 이 똑같다
input1 = Input(shape=(200,))                    # 입력층 : 컬럼 200개 (Sequential 의 input_dim=200)
dense1 = Dense(500, activation='relu')(input1)  # (input1) -> input1 뒤에 연결
drop1 = Dropout(0.3)(dense1)                    # dense1 출력의 30% 를 훈련 때마다 끈다

dense2 = Dense(250, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)

dense3 = Dense(125, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense3)

dense4 = Dense(60, activation='relu')(drop3)
drop4 = Dropout(0.3)(dense4)

dense5 = Dense(30, activation='relu')(drop4)
drop5 = Dropout(0.3)(dense5)

# 다중 분류(원핫) 출력층은 softmax 가 반드시 있어야 칸별 확률(합 = 1)이 나온다
# softmax 가 없으면 출력이 확률이 아닌 실수 값 그대로라 categorical_crossentropy 계산이 맞지 않는다
output1 = Dense(num_classes, activation='softmax')(drop5)

model = Model(inputs=input1, outputs=output1)   # 시작(input1) ~ 끝(output1) 범위를 정해서 모델 완성
model.summary()                                 # keras33 Sequential 과 Total params 가 같아야 제대로 변환한 것

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,                # val_loss 가 20 epoch 동안 안 좋아지면 멈춘다
    restore_best_weights=True,  # 멈춘 뒤 val_loss 가 가장 낮았던 가중치로 되돌린다
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',         # es 와 같은 기준으로 '최고 epoch' 를 고른다
    mode='auto',                # val_loss 는 낮을수록 좋으므로 auto(=min)
    save_best_only=True,        # 최고 기록이 갱신될 때만 덮어쓴다 -> 마지막에 남는 파일 = 최고 epoch 모델
    filepath=path_save + 'keras34_mcp7.keras',  # 파일 번호 규칙 : 07 -> mcp7
    verbose=1,
)

hist = model.fit(x_train, y_train,
                 epochs=500,
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[es, mcp],   # callbacks 에 넣어야 실제로 동작한다
                 verbose=1,
                 )

print("========== ========== ========== ========== ==========")

#4. 평가 예측 (훈련에도 검증에도 안 쓴 x_test 로만)
loss = model.evaluate(x_test, y_test)   # [loss, acc]
print('loss :', loss[0])
print('acc :', round(loss[1], 4))

y_predict = model.predict(x_test)       # softmax 확률 (행마다 2칸)

# 분류 성능은 acc 로 판단한다. r2 / mse / rmse 는 '확률 vs 원핫 정답' 의 오차로 참고용이다.
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

y_predict = np.argmax(y_predict, axis=1)    # 확률이 가장 큰 칸 번호 -> 예측 클래스
y_test_arg = np.argmax(y_test, axis=1)      # 원핫 정답도 클래스 번호로
acc_score = accuracy_score(y_test_arg, y_predict)
print('acc_score :', acc_score)

# ===== 이전 기록 =====
# Sequential + Dropout (keras33_dropout07, 2026-09-14)  loss : 0.24355342984199524 / acc_score : 0.9075333333333333

# ===== 실행 결과 (2026-09-14, 함수형 + Dropout + MCP + r2/mse/rmse, keras33 과 같은 구성) =====
# loss : 0.24084995687007904
# acc : 0.911
# r2 : 0.2369156587196699
# mse : 0.06897247172892762
# RMSE : 0.26262610633546624
# acc_score : 0.9110333333333334
# (Epoch 23: early stopping / Restoring model weights from the end of the best epoch: 3.)
