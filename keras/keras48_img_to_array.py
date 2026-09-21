# keras48_img_to_array.py
# 사진 한 장을 모델에 넣을 수 있는 형태로 바꿔서 npy 로 저장한다
#
# [ 왜 필요한가 ]
#  keras44 ~ keras47 에서는 ImageDataGenerator 가 '폴더 전체' 를 읽어 x, y 를 만들어 줬다
#  하지만 내 사진 한 장을 예측할 때는 정답(y)도 없고 폴더 구조도 없다
#  -> load_img 로 한 장을 열고, img_to_array 로 숫자로 바꾸고, 차원을 하나 늘려 predict 형태로 만든다
#
# [ predict 에 넣으려면 4차원이어야 한다 ]
#  모델은 언제나 (장수, 가로, 세로, 채널) 을 받는다. 한 장만 넣어도 '1장짜리 묶음' 이어야 한다
#  (100, 100, 3) -> np.expand_dims -> (1, 100, 100, 3)
#
# [ 주의 : 스케일링은 여기서 하지 않는다 ]
#  img_to_array 결과는 0 ~ 255 그대로다
#  훈련할 때는 ImageDataGenerator(rescale=1./255) 로 0 ~ 1 을 넣었으므로
#  예측하는 쪽(keras49)에서 /255. 를 해 줘야 훈련 때와 같은 범위가 된다

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np
import matplotlib.pyplot as plt

path = 'c:/furiosa_study/_data/image/'
img = load_img(path + 'image.png', target_size=(100,100))   # 훈련 때와 같은 크기로 읽어야 한다
print(img)                        # <PIL.Image.Image image mode=RGB size=100x100 ...>
print(type(img))                  # <class 'PIL.Image.Image'> -> imshow에 넣으면 출력되는 타입

# 사진이 제대로 열렸는지 눈으로 확인하고 싶을 때만 주석을 푼다
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)           # PIL 이미지 -> numpy 배열 (수치화)
print(arr)                        # 픽셀값이 0 ~ 255 로 들어 있다 (숫자가 많아 출력이 길다)
print(arr.shape)                  # (100, 100, 3)
print(type(arr))                  # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0) # 맨 앞에 축을 하나 추가 -> '1장짜리 묶음' 이 된다
print(arr)
print(arr.shape)                  # (1, 100, 100, 3) -> predict에 넣을 수 있는 형태

# 한 번 바꿔 두면 예측할 때마다 다시 읽지 않아도 된다
np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path + 'keras48_man.npy', arr=arr)
