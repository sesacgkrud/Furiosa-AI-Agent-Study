# keras36_cnn4_fashion.py
# fashion_mnist 의류 이미지 10종 분류 - padding='same' 과 strides 를 처음 적용한 파일 (목표 acc 0.92)
# keras36_cnn3_mnist2.py 베이스
#
# 새로 쓴 것
#  padding='same'  : 커널이 지나가도 가로 세로가 줄지 않는다 (가장자리에 0 을 둘러준다)
#                    default 는 padding='valid' (패딩 없음 -> 커널 크기 - 1 만큼 줄어든다)
#  strides=(2,2)   : 커널이 두 칸씩 건너뛴다 -> 가로 세로가 절반이 된다. default 는 1
# -> 두 개를 같이 쓰면 "크기는 strides 를 준 층에서만 줄인다" 가 되어 층 구성이 단순해진다
#
# fashion_mnist 는 mnist 와 모양이 완전히 같다 (28x28 흑백 6만 장, 정답 0~9)
# 다만 옷 / 신발 / 가방 사진이라 손글씨 숫자보다 어려워서 mnist 의 0.99 대는 나오지 않는다

import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

########## 스케일링 : -1 ~ 1 ##########
# x_train = (x_train - 127.5)/127.5
# x_test = (x_test - 127.5)/127.5
# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

########## 4차원으로 reshape ##########
# Conv2D 입력은 (장수, 가로, 세로, 채널) -> 흑백이라 채널 1 을 붙인다
x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)
print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

########## y 원핫 인코딩 ##########
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1,1)    # OneHotEncoder 는 2차원만 받으므로 (60000,) -> (60000, 1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)

#2. 모델 구성
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
                       # 의류 10종 -> 출력 10개 + softmax
model.summary()
# Total params: 1,907,018
#  -> 층 수는 mnist2(3,578,186)와 비슷한데 파라미터는 절반이다
#     strides 로 크기를 일찍 줄여서 Flatten 이 12800 -> 6272 로 작아졌기 때문

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_acc',      # val_loss 기준이면 loss 가 가장 낮은 epoch 를 고르는데, 그 epoch 가 acc 최고점은 아니다
    mode='max',             # acc 는 높을수록 좋으므로 max
    patience=25,            # acc 는 들쭉날쭉해서 loss 기준보다 patience 를 넉넉히 준다
    restore_best_weights=True,
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=128,
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

y_predict = model.predict(x_test)
y_predict=  np.argmax(y_predict, axis=1).reshape(-1,1) # reshape 없어도 됨
y_test = np.argmax(y_test, axis=1).reshape(-1,1)       # reshape 없어도 됨

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

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

# 목표 : CNN을 이겨라