# keras50_flow2_next.py
# keras50_flow1.py 베이스 - 사진 한 장 대신 fashion_mnist 한 장을 100장으로 증폭한다
#
# [ keras50_flow1 과 달라지는 점 ]
#  flow1 : 이미지 한 장을 batch_size=1 로 넣고 next 를 5번 불러 5장을 꺼냈다
#  flow2 : 같은 이미지를 np.tile 로 100장 복사해 넣고, batch_size=100 으로 한 번의 .next() 에 100장을 받는다
#
# [ flow 에 y 를 같이 넣으면 ]
#  flow(x, y) 의 결과는 (x 묶음, y 묶음) 튜플이다 -> [0] 이 x, [1] 이 y
#  여기서는 정답이 필요 없어서 np.zeros 로 자리만 채운 가짜 y 를 넣었다
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#################### 여기부터 증폭이다 ####################

# 켜 둔 옵션은 사진을 조금씩 바꾸는 규칙이다. 꺼 둔 옵션은 이 사진에는 어울리지 않아 남겨만 둔다
datagen = ImageDataGenerator(
    rescale=1./255,             # 픽셀값 0~255 를 0~1 로 (imshow 도 0~1 범위를 그대로 그린다)
    horizontal_flip=True,       # 좌우반전 - 사람 / 사물은 뒤집어도 같은 대상이다
    # vertical_flip=True,       # 상하반전 - 얼굴 사진에는 어울리지 않아 끔
    width_shift_range=0.1,      # 좌우로 최대 10% 이동
    # height_shift_range=0.1,   # 위아래로 이동
    rotation_range=15,          # -15 ~ +15 도 범위에서 회전
    # zoom_range=1.2,           # 확대 / 축소
    # shear_range=0.7,          # 한 점을 고정하고 기울이기
    fill_mode='nearest',        # 이동 / 회전으로 생긴 빈 자리를 가장 가까운 픽셀값으로 채운다
)

# 이미지 1장을 100장으로 증폭시키는 방법 (복사만 하면 100장이 완전히 똑같다)
augment_size = 100
# print(x_train.shape)          # (60000, 28, 28)
# print(x_train[0].shape)       # (28, 28)

# np.tile 은 배열을 옆으로 이어 붙인다 -> 2차원 (28, 28) 을 그대로 넣으면 가로로 100번 붙는다
# aaa = np.tile(x_train[0], augment_size)
# print(aaa.shape)              # (28, 2800)

# 단순 복붙 -> 그대로 사용하면 X (과적합 될 수 있기 때문에 변환해야 함)
aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1) # (100, 28, 28, 1)
print(aaa.shape)                # (100, 28, 28, 1)

xy_data = datagen.flow(
    # 한 장을 784 칸 1줄로 편 뒤 100번 이어 붙이고 (78400,) -> 100장짜리 4차원으로 다시 접는다
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1), # (100, 28, 28, 1)
    np.zeros(augment_size),     # augment_size(100개) 크기만큼 0으로 채움 -> 정답이 필요 없어 자리만 채운 가짜 y
    batch_size=augment_size,    # 100장을 한 묶음으로 -> .next() 한 번에 100장이 모두 나온다
    shuffle=False,              # 순서를 섞지 않는다 (x 와 y 의 짝 유지)
).next()                        # 복사본 100장이 각각 다르게 변형되어 나온다

print(xy_data)
print(type(xy_data))    # <class 'tuple'>
# print(xy_data.shape)  # AttributeError: 'tuple' object has no attribute 'shape'
print(len(xy_data))     # 2 -> x, y

print(xy_data[0].shape) # (100, 28, 28, 1)
print(xy_data[1].shape) # (100,)

# 100장 중 49장을 7 x 7 격자로 그려서 같은 사진이 조금씩 다르게 변형됐는지 확인한다
plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][i], cmap='gray')

plt.show()