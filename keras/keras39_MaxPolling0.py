# keras39_MaxPolling0.py
# MaxPooling2D 를 summary 로만 확인하는 파일 (훈련 없음)
# keras38_padding_stride_0.py 베이스
#
# MaxPooling2D : 정해진 크기의 구역에서 가장 큰 값 하나만 남기고 버려서 이미지를 줄이는 층
#   pool_size default = (2,2), strides 를 안 주면 pool_size 와 같은 값이 된다
#   -> 2x2 네 칸 중 최대값 1개만 남긴다 = 가로 세로가 정확히 절반이 된다
#   가장 큰 값 = 그 구역에서 특징이 가장 강하게 나온 자리라서, 그 값만 남겨도 특징이 유지된다
#
# strides=2 로 줄이는 것과 무엇이 다른가
#   strides=2   : 커널을 건너뛰며 계산 -> 훑지 않고 지나친 자리는 아예 보지 않는다
#   MaxPooling  : 전부 훑은 뒤(strides=1) 그 결과 중 최대값만 남긴다 -> 다 보고 나서 줄인다
#   또 MaxPooling 은 연산 없이 고르기만 하므로 파라미터가 0 이다
#
# 통상적으로 Conv2D 다음에 붙여서 "특징 뽑기 -> 크기 줄이기" 를 반복한다

import numpy as np
import pandas as pd

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1),
                 strides=1,                          # default
                 padding='same',                     # padding 적용 -> 크기가 줄지 않는다
                 ))                                  # 출력 : (None, 10, 10, 10)
model.add(MaxPooling2D())                            # 출력 : (None, 5, 5, 10)  pool_size (2,2) -> 절반. param 0
                                                     # 통상적으로 MaxPooling2D는 Conv2D 다음 사용

model.add(Conv2D(filters=9, kernel_size=(3,3),       # 앞 층 출력이 5x5 로 줄어든 상태에서 들어간다
                 strides=2,                          # 두 칸씩 건너뛴다. 자투리는 버린다
                 padding='valid',                    # 패딩 없음
                 ))                                  # 출력 : (None, 2, 2, 9)  floor((5-3)/2) + 1 = 2

model.summary()
# _________________________________________________________________
#  Layer (type)                  Output Shape            Param #
# =================================================================
#  conv2d (Conv2D)               (None, 10, 10, 10)      50
#
#  max_pooling2d (MaxPooling2D)  (None, 5, 5, 10)        0        <- 고르기만 해서 파라미터 0
#
#  conv2d_1 (Conv2D)             (None, 2, 2, 9)         819
#
# =================================================================
# Total params: 869
# Trainable params: 869
# Non-trainable params: 0

# keras38(Conv2D 의 strides=2 로 줄인 경우)과 출력 크기 / 파라미터 수가 869 로 같지만,
# keras38 은 건너뛰며 계산했고 여기는 전부 계산한 뒤 최대값을 골라서 줄였다는 점이 다르다

# 주의 : MaxPooling2D 는 절반씩 줄이므로 너무 많이 넣으면 가로 세로가 1 까지 내려가고
#        그 뒤에 또 넣으면 "Negative dimension size" 에러가 난다