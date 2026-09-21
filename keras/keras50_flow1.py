# keras48_img_to_array.py 베이스

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

path = 'c:/furiosa_study/_data/image/'
img = load_img(path + 'image.png', target_size=(100,100))
print(img) # <PIL.Image.Image image mode=RGB size=100x100 at 0x1AEF94F9690>
print(type(img)) # <class 'PIL.Image.Image'> -> imshow에 넣으면 출력되는 타입

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)           # 수치화된 데이터로 변경
print(arr)
print(arr.shape)                  # (100, 100, 3)
print(type(arr))                  # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0) # 차원 증가
print(arr)
print(arr.shape)                  # (1, 100, 100, 3) -> predict에 넣을 수 있는 형태

# np_path = './_data/kaggle_cat_dog_npy/'
# np.save(np_path + 'keras48_man.npy', arr=arr)

#################### 여기부터 증폭이다 ####################

datagen = ImageDataGenerator(
    rescale=1./255,             # 필수
    horizontal_flip=True,       # 수평 뒤집기 (필수x), 좌우반전
    # vertical_flip=True,       # 수직 뒤집기 (필수x), 상하반전
    width_shift_range=0.1,      # 평형 이동 (필수x)
    # height_shift_range=0.1,   # (필수x)
    rotation_range=15,          # r각도 조절 (정해진 각도만큼 이미지 회전) (필수x)
    # zoom_range=1.2,           # (필수x)
    # shear_range=0.7,          # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동 (필수x)
    fill_mode='nearest',        # (필수x)
)

it = datagen.flow(arr,
                  batch_size=1,
                  )

# print(it)              # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001EEFAE2BDF0>
# print(it.next())       # Python 3.10 까지
# print(next(it))        # Python 3.11 이후
# print(next(it).shape)  # (1, 100, 100, 3) -> ImageDataGenerator 를 거친 변경된 형태

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):

    # batch = it.next()
    batch = next(it)
    # print(batch.shape)
    batch = batch.reshape(100,100,3)
    ax[i].imshow(batch)
    # ax[i].axis('off')
    
plt.show()