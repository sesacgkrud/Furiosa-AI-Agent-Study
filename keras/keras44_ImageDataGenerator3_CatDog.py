# keras44_ImageDataGenerator1.py 베이스

import numpy as np
import time

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

#1. 데이터
start_time = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/cat_dog/training_set/'
path_test = './_data/image/cat_dog/test_set'

xy_train = train_datagen.flow_from_directory( # 해당 경로에서 가져오기 위함
    path_train,               # 경로
    target_size=(100,100),    # 설정한 크기대로 이미지를 수정
    batch_size=8005,
    class_mode='binary',      # 이진 분류 (ad, normal)
    color_mode='rgb',         # 흑백 -> 출력 형태 : (batch, x, y, 1)
    shuffle=True,
)
# 실행 결과 : Found 160 images belonging to 2 classes

xy_test = test_datagen.flow_from_directory(
    path_test,                # 경로
    target_size=(100,100),    # 설정한 크기대로 이미지를 수정
    batch_size=2023,
    class_mode='binary',      # 이진 분류 (ad, normal)
    color_mode='rgb',         # 흑백 -> 출력 형태 : (batch, x, y, 1)
    shuffle=False,            # train에서 했기 때문에 필요 없음
)
# 실행 결과 : Found 120 images belonging to 2 classes.

print(xy_train[0][0].shape)   # (160, 100, 100, 3)
print(xy_train[0][1].shape)   # (160,)  class_mode='binary' -> y 는 0(ad) / 1(normal) 한 칸

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) # (160, 100, 100, 3) (160,)
print(x_test.shape, y_test.shape)   # (120, 100, 100, 3) (120,)

# Dense 는 (샘플, 컬럼) 2차원 입력 -> 150 x 150 x 1 = 22500 개 컬럼으로 편다
# x_train = x_train.reshape(-1, 150*150)
# x_test = x_test.reshape(-1, 150*150)
print(x_train.shape, x_test.shape)  # (160, 100, 100, 3) (120, 100, 100, 3)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (5,5), padding='same', activation='relu',input_shape=(100, 100, 3)))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(32, (5,5), padding='same', activation='relu'))
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))

# 이진 분류 출력층은 sigmoid 가 반드시 있어야 0 ~ 1 확률이 나온다
# sigmoid 가 없으면 출력이 음수나 1 초과로 나와서 반올림해도 0, 1 이 되지 않는다
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
# 이진 분류라 loss 는 binary_crossentropy (softmax + categorical_crossentropy 자리)
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

# 목표가 정확도(acc 1.0)라서 val_loss 가 아니라 val_acc 를 기준으로 잡는다
#  - loss 는 낮을수록 좋고 acc 는 높을수록 좋으므로 mode 를 'max' 로 같이 바꿔야 한다
es = EarlyStopping(
    monitor='val_acc',      # val_loss 기준이면 loss 가 가장 낮은 epoch 를 고르는데, 그 epoch 가 acc 최고점은 아니다
    mode='max',             # acc 는 높을수록 좋으므로 max
    patience=130,           # epochs(100) 보다 크면 끝까지 안 멈춘다 -> restore_best_weights 도 동작하지 않는다
    restore_best_weights=True,
    verbose=1,
)

#################### MCP SAVE 파일명 만들기 ####################
import datetime
date = datetime.datetime.now() # 현재 시간 반환

date = date.strftime('%m%d_%H%M')

path = './_save/keras44/'
filename = '{epoch:04d}_{val_acc:.4f}.keras'   # mcp 가 보는 기준(val_acc)을 파일명에도 그대로 쓴다
filepath = ''.join([path, 'k_', date, '_', filename])

## 내가 생각하는 파일명 예)
# './_save/keras44/' + 'k_' + '0918_1147' + '0053_0.8321.keras'

#################### #################### ####################

mcp = ModelCheckpoint(
    monitor='val_acc',          # es 와 같은 기준으로 '최고 epoch' 를 고른다 (한쪽만 val_loss 면 서로 다른 epoch 를 고른다)
    mode='max',                 # acc 는 높을수록 좋으므로 max
    save_best_only=True,        # 최고 기록이 갱신될 때만 덮어쓴다 -> 마지막에 남는 파일 = 최고 epoch 모델
    filepath=filepath,
    verbose=1,
)

start_time = time.time()

# batch_size 는 '한 번에 GPU 에 올릴 장수' -> 전체 장수(8005)와 다른 값이다
# 10000 으로 주면 7204 장을 한 번에 Conv2D 에 넣게 되어 OOM (7204x64x100x100x4 = 약 18GB)
model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1,
          validation_split=0.1,
          callbacks=[es, mcp],
          )

end_time = time.time()

#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

# predict 결과는 (120, 1) 짜리 sigmoid 확률 (0 ~ 1)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)   # 0.5 기준 반올림 -> 0 또는 1 (안 하면 accuracy_score 가 ValueError)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# 목표 : acc 0.77
# loss : 0.6931535005569458
# acc : 0.5002471804618835
# accuracy_score : 0.5002471576866041
# 소요 시간 : 807.22 초