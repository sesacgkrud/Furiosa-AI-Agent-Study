# keras36_cnn2_mnist_imshow.py
# mnist(손글씨 숫자 0~9) 데이터를 불러와서 모양과 값을 눈으로 확인한다
# (60000, 28, 28) : 28 x 28 짜리 흑백 이미지 6만 장, 값은 0 ~ 255
# plt.imshow 로 한 장을 그려 보고 y_train 의 정답과 맞는지 본다

import numpy as np
import pandas as pd

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)
print(x_train[0])
# print(x_train[0][0])

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

print(pd.value_counts(y_test))
# 1    1135
# 2    1032
# 7    1028
# 3    1010
# 9    1009
# 4     982
# 0     980
# 8     974
# 6     958
# 5     892
# Name: count, dtype: int64

import matplotlib.pyplot as plt
plt.imshow(x_train[50000], 'gray')
plt.show()