# keras46_03_save_npy_man_woman.py
# man_woman 이미지 2종 분류 (남 / 여)
#
# [ 이 파일이 하는 일 ]
#  ImageDataGenerator 로 폴더의 이미지를 읽어 train / test 로 나눈 뒤 npy 로 저장한다
#  훈련과 예측은 하지 않는다 -> 훈련은 keras47_03, 사진 한 장 예측은 keras49_02
#  (keras46_01 / 46_02 는 저장과 훈련을 한 파일에서 했지만, 여기서는 역할을 나눴다.
#   데이터가 크고 훈련을 여러 번 반복하게 되므로 '읽기' 를 한 번만 하려는 것)
#
# [ 이 데이터의 특징 ]
#  1) 장수가 많다 : man 17678 / woman 9489 (총 27167장)
#     100 x 100 x 3 float32 -> 한 장에 120KB, 전체 약 3.3GB 가 메모리에 올라온다
#     train_test_split 이 복사본을 만드는 순간 잠깐 2배까지 쓰므로 여유 메모리를 확인하고 돌린다
#  2) 장수가 고르지 않다 (남 65% / 여 35%) -> stratify 로 비율을 맞춰 나눈다
#  3) cat_dog 처럼 sigmoid 로 푼다 : class_mode='binary' -> y (N,) / 출력 Dense(1, sigmoid)
#     (horse-human / rps 는 softmax 로 풀었다. 2종이면 둘 다 가능하다)

import numpy as np
import time

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#1. 데이터
########## ImageDataGenerator 로 폴더에서 읽기 ##########
start_time1 = time.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 픽셀값 0~255 를 0~1 로 줄인다
)

path_train = './_data/image/man_woman/'   # 이 폴더의 하위 폴더 이름(man / woman)이 곧 클래스가 된다

xy_train = train_datagen.flow_from_directory(
    path_train,                 # 읽어올 상위 폴더
    target_size=(100,100),      # 사진마다 크기가 달라도 이 크기로 맞춰서 읽는다
    batch_size=27167,           # 전체 장수 -> 묶음이 1개가 되어 xy_train[0] 에 전 데이터가 담긴다
    class_mode='binary',        # 0 / 1 한 칸 -> y 가 (N,)   ('categorical' 이면 (N, 2))
    color_mode='rgb',           # 컬러 -> x 가 (N, 100, 100, 3)
    shuffle=True,               # 남 / 여가 한쪽으로 몰리지 않도록 순서를 섞는다
)
# 실행 결과 : Found 27167 images belonging to 2 classes.

# 클래스 번호는 폴더 이름 알파벳순으로 케라스가 정한다 -> 외우지 말고 찍어서 확인한다
# 이 번호가 곧 sigmoid 예측값의 방향이 된다 (1 에 가까우면 woman)
print(xy_train.class_indices)   # {'man': 0, 'woman': 1}

# flow_from_directory 까지는 파일 목록만 훑은 상태다
# [0] 을 꺼내는 순간 실제로 이미지를 열어 numpy 배열로 바꾼다 -> 시간이 걸리는 지점
x = xy_train[0][0]              # (27167, 100, 100, 3)
y = xy_train[0][1]              # (27167,)

end_time1 = time.time()
print('데이터 변환 시간 :', round(end_time1 - start_time1, 2), '초')

print(x.shape, y.shape)         # (27167, 100, 100, 3) (27167,)

########## train / test 분리 ##########
# test 폴더가 따로 없으므로 직접 나눈다
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=42,
    shuffle=True,
    stratify=y,                 # y 가 0 / 1 이라 원핫 변환 없이 바로 넣을 수 있다 (softmax 였다면 np.argmax 필요)
)

print(x_train.shape, y_train.shape)     # (21733, 100, 100, 3) (21733,)
print(x_test.shape, y_test.shape)       # (5434, 100, 100, 3) (5434,)

########## npy 로 저장 ##########
# 27167장을 읽는 데 30초 가까이 걸린다. 한 번 저장해 두면 keras47_03 / keras49_02 에서 1초 안에 불러온다
np_path = './_data/man_woman_npy/'
np.save(np_path + 'keras46_03_x_train.npy', arr=x_train)
np.save(np_path + 'keras46_03_y_train.npy', arr=y_train)
np.save(np_path + 'keras46_03_x_test.npy', arr=x_test)
np.save(np_path + 'keras46_03_y_test.npy', arr=y_test)

print('npy 저장 완료 :', np_path)
