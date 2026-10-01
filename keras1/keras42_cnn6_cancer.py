import numpy as np
import time

from sklearn.datasets import load_breast_cancer   # 유방암 데이터 (이진 분류 : 0 악성 / 1 양성)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras34/'   # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=42,
    shuffle=True,
    stratify=y,     # 0 / 1 비율을 train 과 test 에 똑같이 나눈다
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

# Conv2D 는 (행, 열, 채널) 4차원 입력이 필요하다 -> 컬럼 30개를 5 x 6 x 1 로 바꾼다
x_train = x_train.reshape(-1, 5, 6, 1)  # (398, 5, 6, 1)
x_test = x_test.reshape(-1, 5, 6, 1)    # (171, 5, 6, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', activation='relu', input_shape=(5, 6, 1)))  # 출력 : (5, 6, 64)  param 320 = (2x2x1+1)x64
model.add(Conv2D(64, (2,2), activation='relu'))                                         # 출력 : (4, 5, 64)  param 16448 = (2x2x64+1)x64
model.add(Conv2D(32, (2,2), activation='relu'))                                         # 출력 : (3, 4, 32)  param 8224 = (2x2x64+1)x32
model.add(Flatten())                                                                    # 출력 : (384,)  3x4x32
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))                                                                 # 훈련 때마다 30% 를 끈다 (평가 / 예측 때는 전부 사용)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))                                               # 이진 분류 -> sigmoid 로 0 ~ 1 확률

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,                # val_loss 가 20 epoch 동안 안 좋아지면 멈춘다
    restore_best_weights=True,  # 멈춘 뒤 val_loss 가 가장 낮았던 가중치로 되돌린다
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',                # val_loss 는 낮을수록 좋으므로 auto(=min)
    save_best_only=True,        # 최고 기록이 갱신될 때만 덮어쓴다 -> 마지막에 남는 파일 = 최고 epoch 모델
    filepath=path + 'keras34_mcp6.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=100,
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

y_predict = model.predict(x_test)       # sigmoid 확률 (0 ~ 1)

# 분류 성능은 acc 로 판단한다. r2 / mse / rmse 는 '확률 vs 정답(0, 1)' 의 오차로 참고용이다.
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

y_predict_arg = np.round(y_predict)     # 0.5 기준 반올림 -> 0 또는 1 (안 하면 accuracy_score 가 ValueError)

acc_score = accuracy_score(y_test, y_predict_arg)
print('accuracy_score :', acc_score)

# ===== CPU 기록 =====
# 소요 시간 : 8.34 초
# loss : 0.13056117296218872
# acc : 0.9649
# r2 : 0.8715246319770813
# mse : 0.030087867751717567
# RMSE : 0.17345854764674346
# acc_score : 0.9649122807017544

# ===== GPU 기록 =====
# 소요 시간 : 6.49 초
# loss : 0.1108744665980339
# acc : 0.9591
# r2 : 0.8735296726226807
# mse : 0.029618313536047935
# RMSE : 0.17209971974424576
# acc_score : 0.9590643274853801

# ===== GPU 기록 ===== <- Conv2D 적용
# Epoch 51: early stopping
# 소요 시간 : 5.99 초
# loss : 0.14266230165958405
# acc : 0.965
# r2 : 0.8670377135276794
# mse : 0.031138665974140167
# RMSE : 0.1764615141444167
# accuracy_score : 0.9649122807017544