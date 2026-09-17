import numpy as np
import time

from sklearn.datasets import load_wine     # 와인 데이터 (다중 분류 : 0 / 1 / 2)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

path = './_save/keras34/'   # 모델 저장 폴더 (furiosa_study 폴더에서 실행하는 기준)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)

y = to_categorical(y)   # 원핫 : (178,) -> (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=50,
    shuffle=True,
    stratify=y,     # 클래스 비율을 train 과 test 에 똑같이 나눈다
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # x_train 으로 기준(중앙값, IQR)을 구하고 변환까지 한 번에
x_test = scaler.transform(x_test)       # test 는 transform 만 (fit 하면 데이터 누수)

# Conv2D 는 (행, 열, 채널) 4차원 입력이 필요하다 -> 컬럼 13개를 13 x 1 x 1 로 바꾼다
# 13 은 소수라 직사각형으로 못 나눈다 -> 세로로 세우고 커널을 (2,1) 로 쓴다
x_train = x_train.reshape(-1, 13, 1, 1)  # (124, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)    # (54, 13, 1, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,1), padding='same', activation='relu', input_shape=(13, 1, 1)))  # 출력 : (13, 1, 64)  param 192 = (2x1x1+1)x64
model.add(Conv2D(64, (2,1), activation='relu'))                                          # 출력 : (12, 1, 64)  param 8256 = (2x1x64+1)x64
model.add(Conv2D(32, (2,1), activation='relu'))                                          # 출력 : (11, 1, 32)  param 4128 = (2x1x64+1)x32
model.add(Flatten())                                                                     # 출력 : (352,)  11x1x32
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))                                                                  # 훈련 때마다 40% 를 끈다 (평가 / 예측 때는 전부 사용)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))                                                # 다중 분류(원핫) -> softmax 로 칸별 확률 (합 = 1)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,                # val_loss 가 50 epoch 동안 안 좋아지면 멈춘다
    restore_best_weights=True,  # 멈춘 뒤 val_loss 가 가장 낮았던 가중치로 되돌린다
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',                # val_loss 는 낮을수록 좋으므로 auto(=min)
    save_best_only=True,        # 최고 기록이 갱신될 때만 덮어쓴다 -> 마지막에 남는 파일 = 최고 epoch 모델
    filepath=path + 'keras34_mcp8.keras',
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

y_predict = model.predict(x_test)       # softmax 확률 (행마다 3칸)

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
# 소요 시간 : 8.02 초
# loss : 0.1516449749469757
# acc : 0.963
# r2 : 0.9148545485493251
# mse : 0.01852531937049567
# RMSE : 0.13610774911993684
# accuracy_score : 0.9629629629629629

# ===== GPU 기록 =====
# 소요 시간 : 4.65 초
# loss : 0.0985560491681099
# acc : 0.981
# r2 : 0.9433357119560242
# mse : 0.01232890784740448
# RMSE : 0.111035615220543
# accuracy_score : 0.9814814814814815

# ===== GPU 기록 ===== <- Conv2D 적용
# 소요 시간 : 6.49 초
# loss : 0.16413834691047668
# acc : 0.963
# r2 : 0.9251127243041992
# mse : 0.016293292865157127
# RMSE : 0.1276451834780973
# accuracy_score : 0.9629629629629629