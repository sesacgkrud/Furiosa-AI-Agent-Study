# keras46_01_save_npy_horse.py
# horse-human 이미지 2종 분류 (말 / 사람)
#
# [ 이 파일이 하는 일 ]
#  1) ImageDataGenerator 로 폴더의 이미지를 읽어 x, y 를 만든다   (keras44 에서 배운 방식)
#  2) train / test 로 나눈 뒤 npy 로 저장한다                      (keras45 에서 배운 방식)
#  3) 같은 데이터로 모델을 훈련하고 평가한다
#  -> 저장한 npy 는 keras47_01 에서 불러 쓴다
#
# [ 2진 분류를 sigmoid 가 아니라 softmax 로 푼다 ]
#  sigmoid : class_mode='binary'      -> y (N,)   / Dense(1, sigmoid) / binary_crossentropy      / 예측 np.round
#  softmax : class_mode='categorical' -> y (N, 2) / Dense(2, softmax) / categorical_crossentropy / 예측 np.argmax
#  둘 다 같은 문제를 풀지만 softmax 는 클래스가 늘어나도 출력 칸 수만 바꾸면 된다
#  (keras46_02 의 가위바위보 3종이 이 구조를 그대로 쓴다. sigmoid 1칸은 이렇게 확장할 수 없다)
#
# [ 데이터 구조 ]
#  horse-human 은 brain / cat_dog 과 달리 train / test 폴더가 나뉘어 있지 않다
#  _data/image/horse-human/horses (500장) , humans (527장)
#  -> 상위 폴더 하나를 읽어 전부 받은 뒤 train_test_split 으로 직접 나눈다

import numpy as np
import time

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
########## ImageDataGenerator 로 폴더에서 읽기 ##########
start_time1 = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 픽셀값 0~255 를 0~1 로 줄인다 (이미지에서는 이 한 줄이 스케일링 역할)
)                               # 뒤집기 / 회전 옵션은 원본을 왜곡하므로 데이터를 늘리고 싶을 때만 쓴다

path_train = './_data/image/horse-human/'   # 이 폴더의 하위 폴더 이름(horses / humans)이 곧 클래스가 된다

xy_train = train_datagen.flow_from_directory(
    path_train,                 # 읽어올 상위 폴더
    target_size=(100,100),      # 사진마다 크기가 달라도 이 크기로 맞춰서 읽는다
    batch_size=1027,            # 전체 장수 -> 묶음이 1개가 되어 xy_train[0] 에 전 데이터가 담긴다
    class_mode='categorical',   # 원핫 -> y 가 (N, 2)   ('binary' 면 (N,) 한 칸)
    color_mode='rgb',           # 컬러 -> x 가 (N, 100, 100, 3)   ('grayscale' 이면 채널 1)
    shuffle=True,               # 말 / 사람이 한쪽으로 몰리지 않도록 순서를 섞는다
)
# 실행 결과 : Found 1027 images belonging to 2 classes.

# 클래스 번호는 폴더 이름 알파벳순으로 케라스가 정한다 -> 외우지 말고 찍어서 확인한다
print(xy_train.class_indices)   # {'horses': 0, 'humans': 1}

# flow_from_directory 까지는 파일 목록만 훑은 상태다
# [0] 을 꺼내는 순간 실제로 이미지를 열어 numpy 배열로 바꾼다 -> 시간이 걸리는 지점
# 앞의 [0] = 몇 번째 묶음인가 / 뒤의 [0] [1] = 그 묶음의 x 냐 y 냐
x = xy_train[0][0]              # (1027, 100, 100, 3)
y = xy_train[0][1]              # (1027, 2)
# x, y = xy_train[0]            # 한 줄로 받으면 x 와 y 가 같은 묶음임이 확실해진다

end_time1 = time.time()
print('데이터 변환 시간 :', round(end_time1 - start_time1, 2), '초')

print(x.shape, y.shape)         # (1027, 100, 100, 3) (1027, 2)

########## train / test 분리 ##########
# test 폴더가 따로 없으므로 직접 나눈다 (brain / cat_dog 은 폴더가 나뉘어 있어 이 과정이 없었다)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=42,
    shuffle=True,
    stratify=np.argmax(y, axis=1),  # 원핫은 그대로 못 넣으므로 클래스 번호로 바꿔 말 / 사람 비율을 맞춘다
)

print(x_train.shape, y_train.shape)     # (821, 100, 100, 3) (821, 2)
print(x_test.shape, y_test.shape)       # (206, 100, 100, 3) (206, 2)

########## npy 로 저장 ##########
# 폴더에서 이미지를 읽는 건 느리지만, numpy 배열을 통째로 저장해 두면 다음부터는 순식간에 불러온다
# 나눈 뒤에 저장하므로 keras47_01 에서는 불러와서 바로 훈련만 하면 된다
np_path = './_data/horse_npy/'
np.save(np_path + 'keras46_01_x_train.npy', arr=x_train)
np.save(np_path + 'keras46_01_y_train.npy', arr=y_train)
np.save(np_path + 'keras46_01_x_test.npy', arr=x_test)
np.save(np_path + 'keras46_01_y_test.npy', arr=y_test)

#2. 모델 구성
# 이미지라서 Conv2D 를 쓴다 (Day14 에서 같은 이미지를 Dense 만으로 풀었을 때보다 CNN 이 확실히 좋았다)
model = Sequential()
model.add(Conv2D(64, (5,5), padding='same', activation='relu', input_shape=(100, 100, 3)))
                                # padding='same' -> 크기 유지 : (100, 100, 64)
model.add(Conv2D(64, (5,5), activation='relu'))
                                # padding 없음 -> (커널 - 1) 만큼 감소 : (96, 96, 64)
model.add(MaxPooling2D())       # 2x2 중 최대값만 남겨 절반 : (48, 48, 64), 파라미터 0
model.add(Dropout(0.2))         # 훈련 때만 20% 를 끈다 (평가 / 예측 때는 전부 사용)

model.add(Conv2D(32, (5,5), padding='same', activation='relu'))
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())            # 4차원 -> 2차원. 여기서부터 Dense 로 넘어간다

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# 원핫(2칸)으로 훈련하므로 출력층은 softmax 2개
# softmax 는 칸마다 확률을 만들고 칸들의 합이 1 이 된다
model.add(Dense(2, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
# 원핫 + softmax 조합이라 loss 는 categorical_crossentropy
# (sigmoid 1칸이었다면 binary_crossentropy - keras44_imageDataGenerator2 brain 파일 참고)
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

# 목표가 정확도라서 val_loss 가 아니라 val_acc 를 기준으로 잡는다
#  - loss 는 낮을수록 좋고 acc 는 높을수록 좋으므로 mode 도 'max' 로 같이 바꿔야 한다
es = EarlyStopping(
    monitor='val_acc',          # val_loss 최저점 epoch 와 val_acc 최고점 epoch 는 서로 다르다
    mode='max',                 # acc 는 높을수록 좋으므로 max
    patience=20,                # epochs 보다 작게 줘야 실제로 멈추고 restore_best_weights 도 동작한다
    restore_best_weights=True,  # 멈춘 뒤 val_acc 가 가장 높았던 가중치로 되돌린다
    verbose=1,
)

#################### MCP SAVE 파일명 만들기 ####################
# 실행할 때마다 이전 파일이 덮어써지지 않도록 날짜 / 시간을 파일명에 넣는다
import datetime
date = datetime.datetime.now()  # 현재 시간 반환

date = date.strftime('%m%d_%H%M')

path = './_save/keras46/'
filename = '{epoch:04d}_{val_acc:.4f}.keras'    # mcp 가 보는 기준(val_acc)을 파일명에도 그대로 쓴다
filepath = ''.join([path, 'k46_01_', date, '_', filename])

## 만들어지는 파일명 예)
# './_save/keras46/' + 'k46_01_' + '0921_1147' + '0053_0.9821.keras'

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
# 너무 크게 주면 한 번에 올리는 양이 GPU 메모리를 넘겨 OOM 이 난다
model.fit(x_train, y_train, epochs=200, batch_size=32,
          verbose=1,
          validation_split=0.2,     # x_train 의 20% 를 검증용으로 뗀다
          callbacks=[es, mcp],      # callbacks 에 넣어야 실제로 동작한다
          )

end_time = time.time()

#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

# predict 결과는 (206, 2) 짜리 softmax 확률 -> argmax 로 확률이 큰 칸의 번호를 뽑는다
# y_test 도 원핫이라 똑같이 되돌려야 accuracy_score 에 넣을 수 있다 (sigmoid 였다면 np.round)
y_predict = model.predict(x_test)
y_predict_arg = np.argmax(y_predict, axis=1)
y_test_arg = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# 2진 분류 + 소프트맥스
# 목표 : acc : 1.0
