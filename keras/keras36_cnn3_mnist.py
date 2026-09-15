# keras36_cnn3_mnist.py
# 이미지 데이터 스케일링 : 픽셀값 0 ~ 255 를 그대로 넣지 않고 작은 범위로 줄인다
#  방법 1) x / 255.      -> 0 ~ 1   (MinMax / MaxAbs 와 같은 결과)
#  방법 2) (x - 127.5) / 127.5 -> -1 ~ 1
# 이미지는 모든 컬럼(픽셀)의 범위가 0 ~ 255 로 같아서 scaler 없이 나눗셈만으로 스케일링이 된다

# keras36_cnn2_mnist_imshow.py 베이스

import numpy as np
import pandas as pd

from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

########## 스케일링 1 ##########
# x_train = x_train/255. # . 의미 -> float 형태로 출력, 이미지 데이터라서 255로 나눈 것 -> 이 상태에서 Min-Max나 MaxAbs와 동일
# x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0

########## 스케일링 2 ##########
x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5
print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test))   # 1.0 -1.0