import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/keras68/'

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5

x_train = x_train.reshape(-1, 28, 28)   # Conv1D 는 3차원 입력 (N, timesteps, feature) -> 이미지 한 행(28픽셀)을 한 시점으로 본다
x_test = x_test.reshape(-1, 28, 28)
print(x_train.shape, x_test.shape)

print(y_train.shape, y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28)))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,   # patience 100 -> 10 : epochs 100 과 같으면 EarlyStopping 이 의미가 없다
    restore_best_weights=True,
    verbose=1,
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=path + 'keras68_conv1d_12_fashion.keras',
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp],
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

import matplotlib.pyplot as plt
plt.imshow(x_train[50000], 'gray')
plt.show()

# ===== CPU 기록 =====
# loss : 0.26822808384895325
# acc : 0.9261999726295471
# accuracy_score : 0.9262
# 소요 시간 : 951.0 초

# ===== CPU 기록 =====  <- MaxPooling 적용
# loss : 0.2809465229511261
# acc : 0.9279000163078308
# accuracy_score : 0.9279
# 소요 시간 : 763.36 초

# ===== GPU 기록 =====
# loss : 0.26894062757492065
# acc : 0.9258000254631042
# accuracy_score : 0.9258
# 소요 시간 : 380.58 초

# ===== GPU 기록 =====  <- MaxPooling 적용
# loss : 0.2832645773887634
# acc : 0.9229999780654907
# accuracy_score : 0.923
# 소요 시간 : 116.61 초

# ===== GPU 기록 ===== <- GlobalAveragePooling2D 적용
# loss : 0.2701389491558075
# acc : 0.9247999787330627
# accuracy_score : 0.9248
# 소요 시간 : 118.45 초

# ===== GPU 기록 ===== <- DNN
# Epoch 75: early stopping
# loss : 0.43086478114128113
# acc : 0.866100013256073
# accuracy_score : 0.8661
# 소요 시간 : 77.2 초

# ===== GPU 기록 ===== <- 함수형 적용
# loss : 0.48146337270736694
# acc : 0.8787999749183655
# accuracy_score : 0.8788
# 소요 시간 : 102.52 초

# ===== GPU 기록 ===== <- 증폭 적용
# Epoch 84: early stopping
# loss : 0.385041207075119
# acc : 0.8790000081062317
# accuracy_score : 0.879
# 소요 시간 : 154.43 초

# ===== GPU 기록 ===== <- learning rate 적용
# Epoch 26: early stopping
# loss : 0.9346636533737183
# acc : 0.6219000220298767
# accuracy_score : 0.6219
# 소요 시간 : 46.84 초

# ========== ========== ========== ========== ========== <- ReduceLROnPlateau 적용 후

# Epoch 29: early stopping
# loss : 0.6090297698974609
# acc : 0.7825000286102295
# accuracy_score : 0.7825
# 소요 시간 : 121.27 초

# [결론] 0.01 고정 0.6219 -> 시작 0.005 + ReduceLROnPlateau 0.7825 (Epoch 29).
#        0.16 이나 올랐지만 증폭 기록 0.879 에는 못 미친다.

# ===== GPU 기록 ===== <- Conv1D 적용
# Epoch 39: early stopping
# loss : 0.27920597791671753
# acc : 0.8996999859809875
# accuracy_score : 0.8997
# 소요 시간 : 118.98 초