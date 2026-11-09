# keras41_dnn1_mnist.py
# [목적] mnist 이미지를 (60000, 784) 로 펴서 Conv2D 없이 Dense 만(DNN) 으로 분류하고 CNN 결과와 비교 (목표 : CNN 을 이겨라)
#
# 아래 설명 블록은 베이스 파일에서 복사된 내용 (이 파일에서 새로 바뀐 점은 바로 위 [목적] 참고)
# mnist1 과 달라진 점
#  1) Conv2D 를 7층으로 늘리고 필터 수를 32 -> 64 -> 128 로 키웠다 ((5,5) 커널을 섞어 크기를 빠르게 줄임)
#  2) Dense 를 256 / 128 로 키우고 Dropout 을 0.25 ~ 0.4 로 올렸다
#  3) EarlyStopping 을 monitor='val_acc', mode='max' 로 걸었다
#  4) validation_split 0.2 -> 0.1 (훈련에 쓰는 데이터를 늘림)
#
# 이미지 데이터 스케일링 : 픽셀값 0 ~ 255 를 그대로 넣지 않고 작은 범위로 줄인다
#  방법 1) x / 255.            -> 0 ~ 1   (MinMax / MaxAbs 와 같은 결과)
#  방법 2) (x - 127.5) / 127.5 -> -1 ~ 1
# 이미지는 모든 컬럼(픽셀)의 범위가 0 ~ 255 로 같아서 scaler 없이 나눗셈만으로 스케일링이 된다

# keras40_GAP01_mnist.py 베이스

import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

########## 스케일링 1 ##########
# x_train = x_train/255. # . 의미 -> float 형태로 출력, 이미지 데이터라서 255로 나눈 것 -> 이 상태에서 Min-Max나 MaxAbs와 동일
# x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0

########## 스케일링 2 ##########
# x_train = (x_train - 127.5)/127.5
# x_test = (x_test - 127.5)/127.5
# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

########## 4차원으로 reshape ##########
# Conv2D 의 입력은 (장수, 가로, 세로, 채널) 4차원 -> mnist 는 흑백이라 채널 1 을 직접 붙인다
x_train = x_train.reshape(-1, 28 * 28 * 1) # 4차원 -> 2차원으로 변경 (DNN으로 만들기 위해서)
x_test = x_test.reshape(-1, 28 * 28 * 1)
print(x_train.shape, x_test.shape)         # (60000, 784) (10000, 784)

########## y 원핫 인코딩 ##########
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)   # 희소행렬이 아니라 numpy 배열로 받는다
# y_train = ohe.fit_transform(y_train)     # (60000,) 1차원을 그대로 넣으면 아래 에러

########## 에러 발생 ##########
# ValueError: Expected 2D array, got 1D array instead:
# array=[5 0 4 ... 5 6 8].
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
# -> OneHotEncoder 는 2차원 입력만 받는다. (60000,) 를 (60000, 1) 로 바꿔주면 해결된다

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)

#2. 모델 구성
# Conv2D 없이 Dense 만 쌓는다 → 입력은 28 x 28 = 784 개 픽셀을 한 줄로 편 2차원 (60000, 784)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()
# Total params: 599,178  (이 파일의 층 구성으로 계산한 값)
#  -> 첫 층 784 x 512 + 512 = 401,920 이 가장 크다

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

# EarlyStopping 의 기준을 val_loss 가 아니라 val_acc 로 바꿔봤다
#  - 목표가 정확도(acc 0.995)이기 때문. val_loss 최저점 epoch 와 val_acc 최고점 epoch 는 서로 다르다
#  - loss 는 낮을수록 좋고 acc 는 높을수록 좋으므로 mode 를 'max' 로 같이 바꿔야 한다
#    (val_acc 인데 mode 를 그대로 두면 정확도가 떨어지는 쪽을 좋다고 판단한다)
es = EarlyStopping(
    monitor='val_acc',      # val_loss 기준이면 loss 가 가장 낮은 epoch 를 고르는데, 그 epoch 가 acc 최고점은 아니다
    mode='max',             # acc 는 높을수록 좋으므로 max
    patience=25,            # acc 는 들쭉날쭉해서 loss 기준보다 patience 를 넉넉히 준다
    restore_best_weights=True,
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
          )

end_time = time.time()

#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

# predict 결과는 (10000, 10) 짜리 확률 -> argmax 로 가장 큰 자리(예측 숫자)를 뽑는다
# y_test 도 원핫이라 똑같이 되돌려서 accuracy_score 에 넣는다
y_predict = model.predict(x_test)
y_predict=  np.argmax(y_predict, axis=1).reshape(-1,1) # reshape 없어도 됨
y_test = np.argmax(y_test, axis=1).reshape(-1,1)       # reshape 없어도 됨

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# ===== CPU 기록 =====
# Epoch 54: early stopping
# acc: 0.9938 - loss: 0.0334 
# loss : 0.03343440219759941
# acc : 0.9937999844551086
# accuracy_score : 0.9938
# 소요 시간 : 1861.89 초

# ===== GPU 기록 =====
# Epoch 70: early stopping
# loss: 0.0292 - acc: 0.9938
# loss : 0.029211144894361496
# acc : 0.9937999844551086
# accuracy_score : 0.9938
# 소요 시간 : 322.36 초

# ===== GPU 기록 ===== <- MaxPooling 적용
# loss : 0.025499815121293068
# acc : 0.9952999949455261
# accuracy_score : 0.9953
# 소요 시간 : 418.48 초

# ===== GPU 기록 ===== <- GlobalAveragePooling2D 적용
# loss : 0.017076997086405754
# acc : 0.9951000213623047
# accuracy_score : 0.9951
# 소요 시간 : 285.72 초

# ===== GPU 기록 ===== <- DNN
# Epoch 59: early stopping
# loss : 0.1566680520772934
# acc : 0.9785000085830688
# accuracy_score : 0.9785
# 소요 시간 : 223.62 초

# 목표 : CNN을 이겨라