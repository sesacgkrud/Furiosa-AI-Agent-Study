import numpy as np
import time

from sklearn.datasets import fetch_covtype  # 산림 피복 데이터 (다중 분류 : 1 ~ 7)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

path = './_save/keras34/'   # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)

# 라벨이 1 ~ 7 이라 to_categorical 은 0 번 칸까지 만들어서 8칸이 된다 (0 번 칸은 항상 0)
y = to_categorical(y)   # (581012, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42,
    shuffle=True,
    stratify=y,     # 클래스 비율을 train 과 test 에 똑같이 나눈다
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

print(x_train.shape, x_test.shape) # (406708, 54) (174304, 54)

# Conv2D 는 (행, 열, 채널) 4차원 입력이 필요하다 -> 컬럼 54개를 6 x 9 x 1 이미지처럼 바꾼다
x_train = x_train.reshape(-1, 6, 9, 1)  # (406708, 6, 9, 1)
x_test = x_test.reshape(-1, 6, 9, 1)    # (174304, 6, 9, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', activation='relu', input_shape=(6, 9, 1)))  # 출력 : (6, 9, 64)  param 320 = (2x2x1+1)x64
model.add(Conv2D(64, (2,2), activation='relu'))                                         # 출력 : (5, 8, 64)  param 16448
model.add(Conv2D(32, (2,2), activation='relu'))                                         # 출력 : (4, 7, 32)  param 8224
model.add(Flatten())                                                                    # 출력 : (896,)  4x7x32
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,                # val_loss 가 30 epoch 동안 안 좋아지면 멈춘다
    restore_best_weights=True,  # 멈춘 뒤 val_loss 가 가장 낮았던 가중치로 되돌린다
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',                # val_loss 는 낮을수록 좋으므로 auto(=min)
    save_best_only=True,        # 최고 기록이 갱신될 때만 덮어쓴다 -> 마지막에 남는 파일 = 최고 epoch 모델
    filepath=path + 'keras34_mcp9.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=10,
                 batch_size=32,
                 validation_split=0.3,
                 callbacks=[es, mcp],
                 verbose=1,
                 )

end_time = time.time()

print("소요 시간 :", round(end_time - start_time, 2), "초")

print("========== ========== ========== ========== ==========")

#4. 평가 예측 (훈련에도 검증에도 안 쓴 x_test 로만)
result = model.evaluate(x_test, y_test) # [loss, acc]
print('loss :', result[0])
print('acc :', round(result[1], 3))

y_predict = model.predict(x_test)       # softmax 확률 (행마다 8칸)

# 분류 성능은 acc 로 판단한다. r2 / mse / rmse 는 '확률 vs 원핫 정답' 의 오차로 참고용이다.
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

y_test_arg = np.argmax(y_test, axis=1)          # 원핫 정답 -> 클래스 번호
y_predict_arg = np.argmax(y_predict, axis=1)    # 확률이 가장 큰 칸 -> 예측 클래스 번호

acc_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', acc_score)

# ===== CPU 기록 =====
# 소요 시간 : 136.02 초
# loss : 0.2725818455219269
# acc : 0.889
# r2 : 0.6215999397276787
# mse : 0.020001761734731602
# RMSE : 0.14142758477302653
# accuracy_score : 0.8892337525243253

# ===== GPU 기록 =====
# 소요 시간 : 234.58 초
# loss : 0.2714778184890747
# acc : 0.892
# r2 : 0.6270689964294434
# mse : 0.01972629874944687
# RMSE : 0.1404503426462423
# accuracy_score : 0.8924063704791628

# ===== GPU 기록 ===== <- Conv2D 적용
# 소요 시간 : 248.45 초
# loss : 0.23483967781066895
# acc : 0.906
# r2 : 0.6672987341880798
# mse : 0.017069324851036072
# RMSE : 0.13064962629504942
# accuracy_score : 0.9060664127042408