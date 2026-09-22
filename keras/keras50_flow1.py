# keras50_flow1.py
# 사진 한 장을 ImageDataGenerator 로 변형해 여러 장으로 늘린다 (증폭 / augmentation)
#
# [ 왜 증폭을 하는가 ]
#  훈련 데이터가 적으면 모델이 그 사진들을 통째로 외워 버린다 (과적합)
#  원본을 조금씩 바꾼 사진을 계속 만들어 주면 "같은 대상이지만 조금 다른 모습" 을 배우게 된다
#
# [ flow 와 flow_from_directory 의 차이 ]
#  flow_from_directory : 폴더를 읽어 x, y 를 만든다          (keras44 ~ keras47 에서 쓴 방식)
#  flow                : 이미 만들어 둔 배열을 넣는다         (이 파일)
#  -> 여기서는 keras48 처럼 사진 한 장을 배열로 만든 뒤 flow 에 넣는다
#
# [ 지금까지와 달라지는 점 ]
#  keras44 ~ keras47 에서는 rescale 만 쓰고 나머지 옵션은 원본을 왜곡하므로 꺼 뒀다
#  여기서는 변형이 목적이므로 뒤집기 / 이동 / 회전 옵션을 일부러 켠다

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

path = 'c:/furiosa_study/_data/image/'
img = load_img(path + 'image.png', target_size=(100,100))   # 사진 한 장을 원하는 크기로 연다
print(img)                        # <PIL.Image.Image image mode=RGB size=100x100 ...>
print(type(img))                  # <class 'PIL.Image.Image'> -> imshow에 넣으면 출력되는 타입

# 사진이 제대로 열렸는지 눈으로 확인하고 싶을 때만 주석을 푼다
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)           # PIL 이미지 -> numpy 배열 (수치화)
print(arr)                        # 픽셀값이 0 ~ 255 로 들어 있다
print(arr.shape)                  # (100, 100, 3)
print(type(arr))                  # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0) # 맨 앞에 축을 추가 -> '1장짜리 묶음'
print(arr)
print(arr.shape)                  # (1, 100, 100, 3) -> flow 와 predict 가 요구하는 4차원 형태

# 예측용 npy 로 저장하는 코드 (keras48 에서 이미 저장했으므로 여기서는 쓰지 않는다)
# np_path = './_data/kaggle_cat_dog_npy/'
# np.save(np_path + 'keras48_man.npy', arr=arr)

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

# flow 는 배열을 받아 '변형된 이미지를 계속 만들어 주는 반복자' 를 돌려준다
# 꺼낼 때마다 옵션 범위 안에서 새로 변형된 사진이 나온다 (원본 파일은 바뀌지 않는다)
it = datagen.flow(arr,
                  batch_size=1,     # 한 번에 1장씩 꺼낸다
                  )

# print(it)              # <keras.preprocessing.image.NumpyArrayIterator object ...>
# print(it.next())       # Python 3.10 까지
# print(next(it))        # Python 3.11 이후
# print(next(it).shape)  # (1, 100, 100, 3) -> ImageDataGenerator 를 거친 변경된 형태

# 변형된 사진 5장을 한 줄로 나란히 그려서 옵션이 어떻게 적용됐는지 눈으로 확인한다
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):

    # batch = it.next()
    batch = next(it)              # (1, 100, 100, 3) - 꺼낼 때마다 다른 변형이 나온다
    # print(batch.shape)
    batch = batch.reshape(100,100,3)  # imshow 는 3차원을 받는다 -> 맨 앞 묶음 축을 뺀다
    ax[i].imshow(batch)
    # ax[i].axis('off')             # 눈금을 숨기고 싶을 때 주석을 푼다

plt.show()
