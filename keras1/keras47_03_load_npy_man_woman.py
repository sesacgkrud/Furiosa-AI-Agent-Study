# keras47_03_load_npy_man_woman.py
# man_woman 2종 분류 - keras46_03 이 저장한 npy 를 불러와 훈련하고, 모델을 파일로 저장한다
#
# [ 이 파일이 하는 일 ]
#  1) 같은 데이터를 두 방법으로 준비해 시간을 비교한다
#     방법 1 : ImageDataGenerator 로 폴더를 읽는다 (27167장이라 오래 걸린다)
#     방법 2 : keras46_03 이 저장해 둔 npy 를 불러온다
#  2) 훈련에는 방법 2(npy) 로 불러온 데이터를 쓴다
#  3) 훈련이 끝난 모델을 저장한다 -> keras49_02 가 이 파일을 불러와 사진 한 장을 예측한다
#
# [ 모델을 저장하는 두 가지 ]
#  ModelCheckpoint : 훈련 도중 val_acc 가 갱신될 때마다 저장한다 (파일명에 날짜 / epoch / val_acc)
#  model.save      : 훈련이 끝난 시점의 모델을 고정된 이름으로 저장한다
#  -> mcp 파일명은 실행할 때마다 달라져서 다른 파일에서 불러 쓰기 불편하다
#     그래서 고정 이름으로 한 번 더 저장해 두고, keras49_02 는 그 이름을 쓴다
#  -> es 에 restore_best_weights=True 가 있으므로 model.save 시점의 가중치도 '가장 좋았던 상태' 다
#
# [ 분류 방식 ] keras46_03 과 동일
#  class_mode='binary' -> y (N,) / Dense(1, sigmoid) / binary_crossentropy / 예측 np.round

import numpy as np
import time

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
########## 방법 1 : ImageDataGenerator 로 폴더에서 읽기 (시간 측정용) ##########
start_time1 = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 픽셀값 0~255 를 0~1 로 줄인다
)

path_train = './_data/image/man_woman/'   # 하위 폴더 이름(man / woman)이 곧 클래스가 된다

xy_train = train_datagen.flow_from_directory(
    path_train,                 # 읽어올 상위 폴더
    target_size=(100,100),      # 사진마다 크기가 달라도 이 크기로 맞춰서 읽는다
    batch_size=27167,           # 전체 장수 -> 묶음이 1개
    class_mode='binary',        # 0 / 1 한 칸 -> y 가 (N,)
    color_mode='rgb',           # 컬러 -> x 가 (N, 100, 100, 3)
    shuffle=True,
)
# 실행 결과 : Found 27167 images belonging to 2 classes.

print(xy_train.class_indices)   # {'man': 0, 'woman': 1}  <- sigmoid 예측값이 가리키는 방향

# [0] 을 꺼내는 순간 이미지를 실제로 읽어 배열로 만든다 -> 시간은 여기까지 재야 한다
x_idg = xy_train[0][0]
y_idg = xy_train[0][1]

end_time1 = time.time()

print(x_idg.shape, y_idg.shape)         # (27167, 100, 100, 3) (27167,)

del xy_train, x_idg, y_idg      # x 만 약 3.3GB 라 시간을 잰 뒤 바로 지운다 (훈련에는 npy 쪽을 쓴다)

########## 방법 2 : npy 불러오기 ##########
start_time2 = time.time()

np_path = './_data/man_woman_npy/'      # keras46_03 이 저장한 폴더 (train / test 분리까지 끝난 상태)
x_train = np.load(np_path + 'keras46_03_x_train.npy')
y_train = np.load(np_path + 'keras46_03_y_train.npy')
x_test = np.load(np_path + 'keras46_03_x_test.npy')
y_test = np.load(np_path + 'keras46_03_y_test.npy')

end_time2 = time.time()

########## 두 방법 시간 비교 ##########
print('ImageDataGenerator 시간 :', round(end_time1 - start_time1, 2), '초')
print('npy 로드 시간 :', round(end_time2 - start_time2, 2), '초')

print(x_train.shape, y_train.shape)     # (21733, 100, 100, 3) (21733,)
print(x_test.shape, y_test.shape)       # (5434, 100, 100, 3) (5434,)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (5,5), padding='same', activation='relu', input_shape=(100, 100, 3)))
                                # padding='same' -> 크기 유지 : (100, 100, 64)
model.add(Conv2D(64, (5,5), activation='relu'))
                                # padding 없음 -> (커널 - 1) 만큼 감소 : (96, 96, 64)
model.add(MaxPooling2D())       # 절반으로 : (48, 48, 64), 파라미터 0
model.add(Dropout(0.2))

model.add(Conv2D(32, (5,5), padding='same', activation='relu'))
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())            # 4차원 -> 2차원

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# 이진 분류 출력층은 sigmoid 가 있어야 0 ~ 1 확률이 나온다
# man=0 / woman=1 로 훈련하므로 예측값은 '여자일 확률' 이 된다
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
# sigmoid 1칸 조합이라 loss 는 binary_crossentropy (softmax 였다면 categorical_crossentropy)
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

# 목표가 정확도라서 val_loss 가 아니라 val_acc 를 기준으로 잡는다
#  - loss 는 낮을수록 좋고 acc 는 높을수록 좋으므로 mode 도 'max' 로 같이 바꿔야 한다
es = EarlyStopping(
    monitor='val_acc',          # val_loss 최저점 epoch 와 val_acc 최고점 epoch 는 서로 다르다
    mode='max',                 # acc 는 높을수록 좋으므로 max
    patience=10,                # 데이터가 많아 한 epoch 가 길다 -> 너무 오래 기다리지 않도록 짧게
    restore_best_weights=True,  # 멈춘 뒤 val_acc 가 가장 높았던 가중치로 되돌린다
    verbose=1,
)

#################### MCP SAVE 파일명 만들기 ####################
# 실행할 때마다 이전 파일이 덮어써지지 않도록 날짜 / 시간을 파일명에 넣는다
import datetime
date = datetime.datetime.now()  # 현재 시간 반환

date = date.strftime('%m%d_%H%M')

path = './_save/keras47/'
filename = '{epoch:04d}_{val_acc:.4f}.keras'    # mcp 가 보는 기준(val_acc)을 파일명에도 그대로 쓴다
filepath = ''.join([path, 'k47_03_', date, '_', filename])

## 만들어지는 파일명 예)
# './_save/keras47/' + 'k47_03_' + '0921_1147' + '0053_0.8321.keras'

#################### #################### ####################

mcp = ModelCheckpoint(
    monitor='val_acc',          # es 와 기준이 다르면 저장되는 epoch 와 되돌리는 epoch 가 어긋난다
    mode='max',                 # acc 는 높을수록 좋으므로 max
    save_best_only=True,        # 기록이 갱신될 때만 저장 -> 마지막에 남는 파일이 가장 좋은 모델
    filepath=filepath,
    verbose=1,
)

start_time = time.time()

# 여기의 batch_size 는 '한 번에 GPU 에 올릴 장수' 다 (위 flow_from_directory 의 batch_size 와 뜻이 다르다)
model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1,
          validation_split=0.2,     # x_train 의 20% 를 검증용으로 뗀다
          callbacks=[es, mcp],      # callbacks 에 넣어야 실제로 동작한다
          )

end_time = time.time()

########## 모델 저장 ##########
# 구조 + 가중치 + 컴파일 설정까지 한 파일에 담긴다 -> keras49_02 는 load_model 한 줄로 불러 쓴다
model.save(path + 'keras47_03_man_woman.keras')
print('모델 저장 완료 :', path + 'keras47_03_man_woman.keras')

#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

# predict 결과는 (5434, 1) 짜리 sigmoid 확률 (0 ~ 1)
# 0.5 기준으로 반올림해야 0 / 1 정답과 비교할 수 있다 (softmax 였다면 np.argmax)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')
