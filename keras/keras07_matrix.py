# keras07_matrix.py
# numpy shape 읽는 법 : 대괄호 [ 가 열린 깊이 = 차원 수, 각 자리의 숫자 = 그 차원의 원소 개수
# 모델의 input_dim / input_shape 를 맞추려면 데이터의 shape 부터 정확히 볼 수 있어야 한다

import numpy as np

x1 = np.array([1,2,3]) # 스칼라 데이터 3개 있는 1차원 데이터
print("x1 =", x1.shape) # x1 = (3,)

x2 = np.array([[1,2,3]])
print("x2 =", x2.shape) # x2 = (1, 3)

x3 = np.array([[1,2], [3,4]])
print("x3 =", x3.shape) # x3 = (2, 2)

x4 = np.array([[1,2], [3,4], [5,6]])
print("x4 =", x4.shape) # x4 = (3, 2)

x5 = np.array([[[1,2], [3,4], [5,6]]])
# x5 = np.array([[[1,2], [3,4], [5,6,]]]) # , 의미 : 더 입력할 게 있으면 추가해도 됨 / 에러 발생X
print("x5 =", x5.shape) # x5 = (1, 3, 2)

x6 = np.array([[[1,2,], [3,4]], [[5,6], [7,8]]])
print("x6 =", x6.shape) # x6 = (2, 2, 2)

x7 = np.array([[[[[1,2,3,4,5], [6,7,8,9,10]]]]])
print("x7 =", x7.shape) # x7 = (1, 1, 1, 2, 5)

x8 = np.array([[[1,2,3]], [[4,5,6]]])
print("x8 =", x8.shape) # x8 = (2, 1, 3)

x9 = np.array([[[[1]]], [[[2]]]])
print("x9 =", x9.shape) # x9 = (2, 1, 1, 1)