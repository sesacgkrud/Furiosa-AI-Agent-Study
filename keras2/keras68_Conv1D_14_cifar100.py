import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization   # [수정] BatchNormalization 추가
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras68/'

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

x_train = x_train.reshape(-1, 32, 32*3)   # Conv1D 는 3차원 입력 -> 이미지 한 행(32픽셀 × RGB 3 = 96)을 한 시점으로 본다
x_test = x_test.reshape(-1, 32, 32*3)
print(x_train.shape, x_test.shape)

print(y_train.shape, y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', input_shape=(32, 96)))   # input_shape (32, 32, 3) -> (32, 96), 클래스 100개라 filters 64 -> 128
model.add(BatchNormalization())   # 층마다 출력 분포를 맞춰 학습을 안정시키는 BatchNormalization
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=25,
    restore_best_weights=True,
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=path + 'keras68_conv1d_14_cifar100.keras',
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp]
          )

end_time = time.time()

#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# ===== GPU 기록 (1차) ===== <- MaxPooling 없음 (Conv2D 7층, Total params 6,736,100)
# loss : 3.258212089538574
# acc : 0.26930001378059387
# accuracy_score : 0.2693
# 소요 시간 : 394.21 초

# ===== GPU 기록 (2차) ===== <- MaxPooling 적용
# loss : 2.2390894889831543
# acc : 0.43709999322891235
# accuracy_score : 0.4371
# 소요 시간 : 227.8 초

# ===== GPU 기록 ===== <- GlobalAveragePooling2D 적용
# loss : 2.1279296875
# acc : 0.46209999918937683
# accuracy_score : 0.4621
# 소요 시간 : 227.22 초

# ===== GPU 기록 ===== <- DNN
# loss : 3.4793953895568848
# acc : 0.29899999499320984
# accuracy_score : 0.299
# 소요 시간 : 201.95 초

# ===== GPU 기록 ===== <- 함수형 적용
# Epoch 107: early stopping
# loss : 3.402829647064209
# acc : 0.2971999943256378
# accuracy_score : 0.2972
# 소요 시간 : 185.42 초

# ===== GPU 기록 ===== <- 증폭 적용
# Epoch 260: early stopping
# loss : 3.504183530807495
# acc : 0.3325999975204468
# accuracy_score : 0.3326
# 소요 시간 : 889.23 초

# ===== GPU 기록 ===== <- learning rate 적용
# loss : 3.5019049644470215
# acc : 0.31439998745918274
# accuracy_score : 0.3144
# 소요 시간 : 759.51 초

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후

# Epoch 234: early stopping
# loss : 3.6142590045928955
# acc : 0.3269999921321869
# accuracy_score : 0.327
# 소요 시간 : 738.67 초

# [결론] 0.01 고정 0.3144 -> 같은 0.01 에서 시작 + ReduceLROnPlateau 0.3270 (Epoch 234).
#        시작값이 같은데 결과가 올랐다 -> 줄여 나간 것 자체가 효과를 낸 경우다.

# ===== GPU 기록 ===== <- Conv1D 적용
# Epoch 77: early stopping
# loss : 2.5791525840759277
# acc : 0.37400001287460327
# accuracy_score : 0.374
# 소요 시간 : 117.5 초