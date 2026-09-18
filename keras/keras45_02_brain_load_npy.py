# keras45_02_brain_load_npy.py
# keras45_01_brain_save_npy.py 에서 저장한 npy 를 불러와 바로 훈련한다
# ImageDataGenerator 로 폴더를 읽는 대신 np.load 로 불러온다
# 두 방법의 시간을 한 번에 비교하려고 ImageDataGenerator 쪽도 같이 돌려서 시간을 잰다
# (훈련에 쓰는 데이터는 npy 로 불러온 쪽이다)

import numpy as np
import time

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. 데이터
########## 방법 1 : ImageDataGenerator 로 폴더에서 읽기 (시간 측정용) ##########
start_time1 = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,           # 필수

    ######### 원 데이터 왜곡 없도록 주석 처리 #########
    # horizontal_flip=True,     # 수평 뒤집기 (필수x)
    # vertical_flip=True,       # 수직 뒤집기 (필수x)
    # width_shift_range=0.1,    # 평형 이동 (필수x)
    # height_shift_range=0.1,   # (필수x)
    # rotation_range=5,         # 각도 조절 (정해진 각도만큼 이미지 회전) (필수x)
    # zoom_range=1.2,           # (필수x)
    # shear_range=0.7,          # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동 (필수x)
    # fill_mode='nearest',      # (필수x)
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test'

xy_train = train_datagen.flow_from_directory( # 해당 경로에서 가져오기 위함
    path_train,               # 경로
    target_size=(150,150),    # 설정한 크기대로 이미지를 수정
    batch_size=160,
    class_mode='binary',      # 이진 분류 (ad, normal)
    color_mode='grayscale',   # 흑백 -> 출력 형태 : (batch, x, y, 1)
    shuffle=True,
)
# 실행 결과 : Found 160 images belonging to 2 classes

xy_test = test_datagen.flow_from_directory(
    path_test,                # 경로
    target_size=(150,150),    # 설정한 크기대로 이미지를 수정
    batch_size=120,
    class_mode='binary',      # 이진 분류 (ad, normal)
    color_mode='grayscale',   # 흑백 -> 출력 형태 : (batch, x, y, 1)
    shuffle=False,            # train에서 했기 때문에 필요 없음
)
# 실행 결과 : Found 120 images belonging to 2 classes.

# [0] 을 꺼내는 순간 실제로 이미지를 읽어 numpy 배열로 만든다 -> 여기까지가 ImageDataGenerator 의 시간
x_train_idg = xy_train[0][0]
y_train_idg = xy_train[0][1]
x_test_idg = xy_test[0][0]
y_test_idg = xy_test[0][1]

end_time1 = time.time()

print(x_train_idg.shape, y_train_idg.shape) # (160, 150, 150, 1) (160,)
print(x_test_idg.shape, y_test_idg.shape)   # (120, 150, 150, 1) (120,)

del xy_train, xy_test, x_train_idg, y_train_idg, x_test_idg, y_test_idg   # 시간만 재고 메모리에서 지운다

########## 방법 2 : npy 불러오기 ##########
start_time2 = time.time()

np_path = './_data/brain_npy/'   # keras45_01 이 저장한 brain 전용 폴더
x_train = np.load(np_path + 'keras45_01_x_train.npy')
y_train = np.load(np_path + 'keras45_01_y_train.npy')
x_test = np.load(np_path + 'keras45_01_x_test.npy')
y_test = np.load(np_path + 'keras45_01_y_test.npy')

end_time2 = time.time()

########## 두 방법 시간 비교 ##########
print('ImageDataGenerator 시간 :', round(end_time1 - start_time1, 2), '초')
print('npy 로드 시간 :', round(end_time2 - start_time2, 2), '초')

print(x_train.shape, y_train.shape) # (160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)   # (120, 150, 150, 1) (120,)

########## 시간 비교만 하려고 아래 훈련 코드는 통째로 주석 처리 ##########
'''
#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu',input_shape=(150, 150, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
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
    patience=130,           # acc 는 들쭉날쭉해서 loss 기준보다 patience 를 넉넉히 준다
    restore_best_weights=True,
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=200, batch_size=13,
          verbose=1,
          validation_split=0.3,
          callbacks=[es],
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
'''

# ImageDataGenerator 시간 : 0.24 초
# npy 로드 시간 : 0.05 초