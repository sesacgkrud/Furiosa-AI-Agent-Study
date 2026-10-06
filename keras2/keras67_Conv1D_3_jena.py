# keras66_jena_CNN.py 베이스 → Conv2D 를 3차원 입력 Conv1D 로 바꿔서 비교
# y = 'T (degC)' 기온 (keras58_kaggle_jena1 에서 풍향 → 기온으로 바꿔 저장한 npy)

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'    # GPU 메모리 조각 모으기 (tensorflow import 전에 써야 한다)

import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
# keras58_kaggle_jena1 에서 저장한 npy 불러오기
np_path = './_data/kaggle_jena_npy/'
x_data = np.load(np_path + 'keras58_x_data.npy')        # (420263, 13)
y_data = np.load(np_path + 'keras58_y_data.npy')        # (420263,)
x_predict = np.load(np_path + 'keras58_x_predict.npy')  # (144, 13)
y_cor = np.load(np_path + 'keras58_y_cor.npy')          # (144,)  정답 (채점용)

# 스케일링 : split_x 로 자르기 "전" 2차원 상태에서 한다
#   컬럼마다 단위가 다르다 → p(mbar) 는 1000 근처, rho 는 1300 근처, wv 는 0 ~ 10 근처
#   2차원 (행, 13) 일 때 하면 reshape 없이 컬럼별로 바로 스케일링된다
#   (split_x 후 3차원에서 하려면 reshape(-1, 13) → 스케일링 → 다시 reshape 해야 하고 메모리도 훨씬 많이 쓴다)
#   fit 은 훈련 구간(x_data) 으로만 하고, x_predict 는 transform 만 한다 → 예측 구간 정보가 훈련에 새지 않는다
#   y(기온 'T (degC)') 는 스케일링하지 않는다 → 예측값이 바로 ℃ 단위로 나온다
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
x_predict = scaler.transform(x_predict)

# split_x : 한 칸씩 밀면서 size 개씩 자른다 (2차원도 행 기준으로 잘리고 열 개수는 유지된다)
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa, dtype=np.float32)

size = 144      # 10분 × 144 = 하루

x = split_x(x_data, size)   # (420120, 144, 13) → 420263 - 144 + 1 = 420120 묶음
y = split_x(y_data, size)   # (420120, 144)     → y 도 144개씩 묶어서 "다음 하루 144개" 를 한 번에 맞춘다
print(x.shape, y.shape)
# 메모리 : 420120 × 144 × 13 × 4byte(float32) ≈ 3.1GB
#   float64 였으면 약 6.3GB → jena1 에서 float32 로 저장한 이유

# Conv1D 는 3차원 (N, timesteps, feature) 를 받는다 → split_x 결과 (420120, 144, 13) 를 reshape 없이 그대로 쓴다
#   4차원 (144, 13, 1) 로 넣으면 에러 없이 돌아가긴 하지만, Conv1D 가 시간축(144) 이 아니라 컬럼축(13) 을 따라 움직인다
#   → 144개 시점마다 13개 컬럼을 따로 보게 되어 시간 흐름을 못 배우고, 계산량만 늘어난다

# 예측용 x 도 훈련 x 와 같은 3차원 (묶음 1개, 144시점, 13컬럼) 으로 맞춘다
x_predict = x_predict.reshape(1, size, 13)  # (1, 144, 13)

# train / test 분리
#   shuffle(기본값 True) 은 "묶음 단위" 로 섞는다
#   묶음 안의 144개 시점 순서는 그대로 유지되므로, 시계열 순서가 깨지지 않는다
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=333,
)
del x, y    # 원본은 더 이상 안 쓰므로 메모리에서 지운다
print(x_train.shape, x_test.shape)  # (336096, 144, 13) (84024, 144, 13)
print(y_train.shape, y_test.shape)  # (336096, 144) (84024, 144)

#2. 모델 구성
# Conv1D : 커널이 시간축(144) 을 따라 움직이고, 한 번에 kernel_size 개 시점 × 13개 컬럼 전체를 본다
#   padding='same' 이라 timesteps 144 가 줄지 않는다
#   param = (kernel_size × 입력 feature + 1) × filters
#   filters 10 → 64 / 32 로 늘려서 시점마다 뽑는 특징 수를 키웠다
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(size, 13)))  # (None, 144, 64)  param (3×13+1)×64 = 2560
model.add(Conv1D(32, 3, padding='same', activation='relu'))                 # (None, 144, 32)  param (3×64+1)×32 = 6176
model.add(Flatten())                                                        # (None, 144×32) = (None, 4608)
model.add(Dense(64, activation='relu'))                                     # (None, 64)
model.add(Dense(32, activation='relu'))                                     # (None, 32)
model.add(Dense(size))                                                      # 출력 144개 = 다음 하루 144개 시점의 기온
model.summary()

#3. 컴파일, 훈련
# 기온(℃) 은 연속값 회귀 → 원핫(to_categorical) 없이 y 를 그대로 쓰고 loss 는 mse
#   출력층도 activation 없이 (linear) 144개 값을 바로 낸다
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
)

# 훈련이 오래 걸리므로 val_loss 가 가장 좋았던 모델을 파일로 남겨둔다
save_path = './_save/keras67/'
os.makedirs(save_path, exist_ok=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=save_path + 'keras67_jena_conv1d.hdf5',
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=100,
          batch_size=1024,          # 데이터가 33만 개라 배치를 크게 잡아야 한 epoch 이 빨리 끝난다
          validation_split=0.2,
          callbacks=[es, mcp],
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1024)
print('loss :', loss)

y_predict = model.predict(x_predict)    # (1, 144)
y_predict = y_predict.reshape(-1)       # (144,) → 정답 y_cor 와 모양을 맞춘다

# RMSE : 예측값과 정답의 차이를 "원래 단위(도)" 로 보여준다
#   mse 는 제곱이라 단위가 ℃² → 루트를 씌워 몇 ℃ 정도 틀렸는지로 해석한다
rmse = np.sqrt(mean_squared_error(y_cor, y_predict))
print('2016.12.31 00:10 ~ 2017.01.01 00:00 기온 RMSE :', rmse)
print('걸린 시간 :', round(end_time - start_time, 2), '초')

########################### 결과 ########################### <- keras58 LSTM(64) (3차원)
# loss : 3.2112793922424316
# 2016.12.31 00:10 ~ 2017.01.01 00:00 기온 RMSE : 4.803594697560632
# 걸린 시간 : 484.4 초

########################### 결과 ########################### <- 4차원 변환 (Conv2D)
# loss : 3.5331783294677734
# 2016.12.31 00:10 ~ 2017.01.01 00:00 기온 RMSE : 2.4415735294254532
# 걸린 시간 : 921.35 초

########################### 결과 ########################### <- Conv1D 변환 (3차원)
# loss : 3.8668642044067383
# 2016.12.31 00:10 ~ 2017.01.01 00:00 기온 RMSE : 2.4624463496681295
# 걸린 시간 : 320.37 초

# 결과 비교 (y = 'T (degC)' 기온, 같은 데이터 / 같은 split)
#   LSTM(64)                (keras58) : test loss 3.2113 / 마지막 하루 기온 RMSE 4.8036 / 484.4 초
#   Bidirectional(LSTM(64)) (keras59) : test loss 4.1010 / 마지막 하루 기온 RMSE 2.1046 / 919.26 초
#   Conv2D 4차원            (keras66) : test loss 3.5332 / 마지막 하루 기온 RMSE 2.4416 / 921.35 초
#   Conv1D 3차원            (keras67) : test loss 3.8669 / 마지막 하루 기온 RMSE 2.4624 / 320.37 초
#
# [결론] Conv1D 는 Conv2D 와 RMSE 가 거의 같으면서 시간은 약 1/3 이다
#        Conv2D 는 (144, 13, 1) 위를 3x3 으로 훑어 컬럼 3개씩만 보지만, Conv1D 는 한 번에 13개 컬럼 전체를 보고 시간축만 따라 움직인다
#        LSTM 처럼 시점을 하나씩 순서대로 계산하지 않아서 빠르다
