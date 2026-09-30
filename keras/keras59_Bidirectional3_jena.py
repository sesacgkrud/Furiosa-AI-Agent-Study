# keras58_kaggle_jena2.py 베이스 → LSTM 을 Bidirectional(LSTM) 으로 바꿔서 비교

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'    # GPU 메모리 조각 모으기 (tensorflow import 전에 써야 한다)

import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
# keras58_kaggle_jena1 에서 저장한 npy 불러오기 (y = 'T (degC)' 기온)
np_path = './_data/kaggle_jena_npy/'
x_data = np.load(np_path + 'keras58_x_data.npy')        # (420263, 13)
y_data = np.load(np_path + 'keras58_y_data.npy')        # (420263,)
x_predict = np.load(np_path + 'keras58_x_predict.npy')  # (144, 13)
y_cor = np.load(np_path + 'keras58_y_cor.npy')          # (144,)  정답 (채점용)

# 스케일링 : split_x 로 자르기 "전" 2차원 상태에서 한다 (keras58_kaggle_jena2 와 같음)
#   fit 은 훈련 구간(x_data) 으로만 하고, x_predict 는 transform 만 한다
#   y(기온) 는 스케일링하지 않는다 → 예측값이 바로 섭씨 온도로 나온다
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
x_predict = scaler.transform(x_predict)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa, dtype=np.float32)

size = 144      # 10분 × 144 = 하루

x = split_x(x_data, size)   # (420120, 144, 13)
y = split_x(y_data, size)   # (420120, 144)
print(x.shape, y.shape)

x_predict = x_predict.reshape(1, size, 13)  # (1, 144, 13)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=333,
)
del x, y    # 원본은 더 이상 안 쓰므로 메모리에서 지운다
print(x_train.shape, x_test.shape)  # (336096, 144, 13) (84024, 144, 13)
print(y_train.shape, y_test.shape)  # (336096, 144) (84024, 144)

#2. 모델 구성
# Bidirectional : RNN 층을 감싸서 "정방향 + 역방향" 두 번 읽게 만든다
#   정방향 : 1번째 시점 → 144번째 시점 순서로 읽는다 (기존 LSTM)
#   역방향 : 144번째 시점 → 1번째 시점 순서로 거꾸로 읽는다
#   두 방향의 마지막 출력을 이어 붙인다(concat) → 출력 크기가 units × 2
#     LSTM(64)                : (None, 144, 13) → (None, 64)
#     Bidirectional(LSTM(64)) : (None, 144, 13) → (None, 128)
#
# 사용법
#   Bidirectional 은 혼자 쓰는 층이 아니라 RNN 층(SimpleRNN / LSTM / GRU) 을 "감싸는" 층이다
#   input_shape 는 안쪽 LSTM 이 아니라 바깥쪽 Bidirectional 에 준다
#   return_sequences=True 는 안쪽 LSTM 에 준다 → 여러 층 쌓을 때 (None, 144, 128) 3차원이 나온다
#
# 효과
#   정방향만 읽으면 앞쪽 시점 정보가 144칸을 지나오면서 흐려지기 쉽다
#   역방향으로 한 번 더 읽으면 앞쪽 시점도 "가까운 곳" 에서 한 번 더 보게 된다
#   대신 LSTM 을 두 개 쓰는 것과 같아서 파라미터와 훈련 시간이 약 2배가 된다
model = Sequential()
model.add(Bidirectional(LSTM(64), input_shape=(size, 13)))    # (None, 144, 13) → (None, 128)
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(size))                                        # 출력 144개 = 다음 하루 144개 시점의 기온

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  bidirectional (Bidirectiona  (None, 128)              39936
#  l)
#  dense (Dense)               (None, 128)               16512
#  dense_1 (Dense)             (None, 64)                8256
#  dense_2 (Dense)             (None, 144)               9360
# =================================================================
# Total params: 74,064
#
# Bidirectional 파라미터 = LSTM 파라미터 × 2 (정방향 LSTM, 역방향 LSTM 이 가중치를 따로 가진다)
#   LSTM(64) 하나 : 4 × 64 × (13 + 64 + 1) = 19968
#   × 2           : 39936
# 뒤의 Dense(128) 도 입력이 128 이라 128 × 128 + 128 = 16512 (LSTM 만 쓸 때는 64 × 128 + 128 = 8320)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
)

save_path = './_save/keras59/'
os.makedirs(save_path, exist_ok=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=save_path + 'keras59_Bidirectional3_jena.hdf5',
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=100,
          batch_size=1024,
          validation_split=0.2,
          callbacks=[es, mcp],
          )
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1024)
print('loss :', loss)

y_predict = model.predict(x_predict)    # (1, 144)
y_predict = y_predict.reshape(-1)       # (144,) → 정답 y_cor 와 모양을 맞춘다

# RMSE : 몇 도(℃) 정도 틀렸는지
rmse = np.sqrt(mean_squared_error(y_cor, y_predict))
print('2016.12.31 00:10 ~ 2017.01.01 00:00 기온 RMSE :', rmse)
print('걸린 시간 :', round(end_time - start_time, 2), '초')

########################### 결과 ###########################
# loss : 3.2112793922424316
# 2016.12.31 00:10 ~ 2017.01.01 00:00 풍향 RMSE : 4.803594697560632
# 걸린 시간 : 484.4 초

#################### Bidirectional 적용 ####################
# loss : 4.1010355949401855
# 2016.12.31 00:10 ~ 2017.01.01 00:00 기온 RMSE : 2.1045934666029846
# 걸린 시간 : 919.26 초

# 결과 비교 (y = 'T (degC)' 기온, 같은 데이터 / 같은 split)
#   LSTM(64)                : test loss 3.2113 / 마지막 하루 기온 RMSE 4.8036 / 484.4 초
#   Bidirectional(LSTM(64)) : test loss 4.1010 / 마지막 하루 기온 RMSE 2.1046 / 919.26 초
#
# [결론]
#   1. 마지막 하루(144개) 기온 RMSE 는 Bidirectional 이 더 좋았다 (4.80 → 2.10 도)
#   2. test 전체 loss 는 LSTM 이 더 낮았다 (3.21 vs 4.10)
#      → 144개 한 구간 점수는 그날 날씨에 따라 흔들리고, test loss 는 8만 개 묶음 전체의 평균이다
#      → 한 가지 점수만 보지 말고 두 지표를 같이 본다
#   3. 걸린 시간은 약 2배 (484 → 919 초) → Bidirectional 은 LSTM 을 두 개 쓰는 것과 같다
#   4. Bidirectional 은 문장처럼 앞뒤 문맥이 모두 중요한 데이터에서 효과가 크다
#      과거로 미래를 맞추는 시계열에서는 항상 좋아진다고 할 수 없다 → 시간 대비 성능을 같이 보고 고른다
