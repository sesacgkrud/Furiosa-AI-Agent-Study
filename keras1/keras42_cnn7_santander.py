import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

path = './_data/kaggle_santander/'  # 데이터 폴더
path_save = './_save/keras34/'      # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

#1. 데이터
train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # 제출을 안 하므로 test.csv / sample_submission.csv 는 읽지 않는다

x = train_csv.drop(['target'], axis=1)  # var_0 ~ var_199 (200개)
y = train_csv['target']                 # 0 / 1
print(x.shape, y.shape) # (200000, 200) (200000,)

num_classes = len(np.unique(y))                 # 2
y = to_categorical(y, num_classes=num_classes)  # 원핫 : 0 -> [1, 0], 1 -> [0, 1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=777,
    shuffle=True,
    stratify=np.argmax(y, axis=1),  # 원핫을 다시 0 / 1 로 바꿔서 비율 기준으로 사용
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

# Conv2D 는 (행, 열, 채널) 4차원 입력이 필요하다 -> 컬럼 200개를 10 x 20 x 1 로 바꾼다
x_train = x_train.reshape(-1, 10, 20, 1)  # (140000, 10, 20, 1)
x_test = x_test.reshape(-1, 10, 20, 1)    # (60000, 10, 20, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', activation='relu', input_shape=(10, 20, 1)))  # 출력 : (10, 20, 64)  param 320 = (2x2x1+1)x64
model.add(Conv2D(64, (2,2), activation='relu'))                                           # 출력 : (9, 19, 64)  param 16448 = (2x2x64+1)x64
model.add(Conv2D(32, (2,2), activation='relu'))                                           # 출력 : (8, 18, 32)  param 8224 = (2x2x64+1)x32
model.add(Flatten())                                                                      # 출력 : (2592,)  8x18x32
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))                                                                   # 훈련 때마다 30% 를 끈다 (평가 / 예측 때는 전부 사용)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))                                       # 다중 분류(원핫) -> softmax 로 칸별 확률 (합 = 1)

model.summary()

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
    monitor='val_loss',
    mode='auto',                # val_loss 는 낮을수록 좋으므로 auto(=min)
    save_best_only=True,        # 최고 기록이 갱신될 때만 덮어쓴다 -> 마지막에 남는 파일 = 최고 epoch 모델
    filepath=path_save + 'keras34_mcp7.keras',
    verbose=1,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs=10,
                 batch_size=32,
                 validation_split=0.2,
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

y_test_arg = np.argmax(y_test, axis=1)          # 원핫 정답 -> 클래스 번호
y_predict_arg = np.argmax(y_predict, axis=1)    # 확률이 가장 큰 칸 -> 예측 클래스 번호

acc_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', acc_score)

# ===== CPU 기록 =====
# 소요 시간 : 65.38 초
# loss : 0.26788538694381714
# acc : 0.9085
# r2 : 0.1891651251119561
# mse : 0.07328847213823145
# RMSE : 0.27071843701202075
# acc_score : 0.9085166666666666

# ===== GPU 기록 =====
# 소요 시간 : 77.38 초
# loss : 0.2606703042984009
# acc : 0.9087
# r2 : 0.19318419694900513
# mse : 0.07292849570512772
# RMSE : 0.2700527646685509
# acc_score : 0.9087333333333333

# ===== GPU 기록 ===== <- Conv2D 적용
# 소요 시간 : 84.21 초
# loss : 0.24611467123031616
# acc : 0.911
# r2 : 0.2405877709388733
# mse : 0.06864365935325623
# RMSE : 0.2619993499099878
# accuracy_score : 0.91095