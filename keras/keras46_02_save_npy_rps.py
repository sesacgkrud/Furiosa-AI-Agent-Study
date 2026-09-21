# keras46_02_save_npy_rps.py
# rps(가위바위보) 이미지 3종 분류
#
# [ 이 파일이 하는 일 ]
#  1) ImageDataGenerator 로 폴더의 이미지를 읽어 x, y 를 만든다
#  2) train / test 로 나눈 뒤 npy 로 저장한다
#  3) 같은 데이터로 모델을 훈련하고 평가한다
#  -> 저장한 npy 는 keras47_02 에서 불러 쓴다
#
# [ keras46_01(2종) 과 달라지는 점 ]
#  1) 클래스가 3개 -> 원핫이 (N, 3), 출력층이 Dense(3, softmax)
#     2종을 softmax 로 풀어 뒀기 때문에 출력 칸 수만 바꾸면 그대로 이어진다
#     (sigmoid 1칸 방식이었다면 3종으로는 늘릴 수 없다 - 이것이 softmax 를 쓰는 이유)
#  2) 장수가 고르지 않다 : rock 840 / paper 840 / scissors 369 (총 2049장)
#     -> train_test_split 의 stratify 로 세 클래스 비율을 train / test 에 똑같이 맞춘다
#  3) 원본이 300x300 PNG(RGBA) 지만 target_size 와 color_mode='rgb' 로 100x100x3 에 맞춰 읽는다

import numpy as np
import time

from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # 잘린 이미지가 섞여 있으면 읽다가 OSError 가 난다 -> 채워서 읽도록 허용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
########## ImageDataGenerator 로 폴더에서 읽기 ##########
start_time1 = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 픽셀값 0~255 를 0~1 로 줄인다
)

path_train = './_data/image/rps/'   # 이 폴더의 하위 폴더 이름(paper / rock / scissors)이 곧 클래스가 된다

xy_train = train_datagen.flow_from_directory(
    path_train,                 # 읽어올 상위 폴더
    target_size=(100,100),      # 원본 300x300 을 100x100 으로 줄여서 읽는다
    batch_size=2049,            # 전체 장수 -> 묶음이 1개가 되어 xy_train[0] 에 전 데이터가 담긴다
    class_mode='categorical',   # 원핫 -> y 가 (N, 3)
    color_mode='rgb',           # 컬러 -> x 가 (N, 100, 100, 3)   PNG 의 투명 채널은 버린다
    shuffle=True,               # 세 종류가 고르게 섞이도록 순서를 섞는다
)
# 실행 결과 : Found 2049 images belonging to 3 classes.

# 클래스 번호는 폴더 이름 알파벳순으로 케라스가 정한다 -> 외우지 말고 찍어서 확인한다
print(xy_train.class_indices)   # {'paper': 0, 'rock': 1, 'scissors': 2}

# [0] 을 꺼내는 순간 실제로 이미지를 열어 numpy 배열로 바꾼다
# 앞의 [0] = 몇 번째 묶음인가 / 뒤의 [0] [1] = 그 묶음의 x 냐 y 냐
x = xy_train[0][0]              # (2049, 100, 100, 3)
y = xy_train[0][1]              # (2049, 3)

end_time1 = time.time()
print('데이터 변환 시간 :', round(end_time1 - start_time1, 2), '초')

print(x.shape, y.shape)         # (2049, 100, 100, 3) (2049, 3)

########## train / test 분리 ##########
# rps 도 test 폴더가 따로 없어서 직접 나눈다
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=42,
    shuffle=True,
    stratify=np.argmax(y, axis=1),  # scissors 가 적으므로 비율을 맞춰 나눠야 test 가 한쪽으로 쏠리지 않는다
)

print(x_train.shape, y_train.shape)     # (1639, 100, 100, 3) (1639, 3)
print(x_test.shape, y_test.shape)       # (410, 100, 100, 3) (410, 3)

########## npy 로 저장 ##########
# 한 번 읽어 둔 numpy 배열을 저장해 두면 다음부터는 이미지를 다시 디코딩하지 않아도 된다
np_path = './_data/rps_npy/'
np.save(np_path + 'keras46_02_x_train.npy', arr=x_train)
np.save(np_path + 'keras46_02_y_train.npy', arr=y_train)
np.save(np_path + 'keras46_02_x_test.npy', arr=x_test)
np.save(np_path + 'keras46_02_y_test.npy', arr=y_test)

#2. 모델 구성
# keras46_01 과 같은 구조다. 마지막 출력 칸 수만 2 -> 3 으로 바뀐다
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

# 다중 분류(원핫) 출력층은 softmax 가 반드시 있어야 칸별 확률(합 = 1)이 나온다
model.add(Dense(3, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
# 원핫 + softmax 조합이라 loss 는 categorical_crossentropy
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
filepath = ''.join([path, 'k46_02_', date, '_', filename])

## 만들어지는 파일명 예)
# './_save/keras46/' + 'k46_02_' + '0921_1147' + '0053_0.9821.keras'

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

#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

# predict 결과는 (410, 3) 짜리 softmax 확률 -> argmax 로 확률이 가장 큰 칸의 번호를 뽑는다
# y_test 도 원핫이라 똑같이 되돌려야 accuracy_score 에 넣을 수 있다
y_predict = model.predict(x_test)
y_predict_arg = np.argmax(y_predict, axis=1)
y_test_arg = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# 3종 분류 + 소프트맥스
# 목표 : acc : 1.0
