# keras38_padding_stride_0.py
# padding 과 strides 가 출력 크기를 어떻게 바꾸는지 summary 로만 확인하는 파일 (훈련 없음)
#
# padding : 이미지 가장자리에 0 을 둘러서 크기가 줄어드는 것을 막는 옵션
#   'valid' (default) - 패딩 없음. 출력이 (입력 - 커널 + 1) 로 줄어든다
#   'same'            - 출력 가로 세로가 입력과 같아지도록 0 을 둘러준다 (strides=1 일 때)
#   패딩을 주면 가장자리 픽셀도 커널 중앙에 올 수 있어서, 테두리 정보가 덜 버려진다
#
# strides : 커널이 한 번에 몇 칸씩 움직일지 (default 1)
#   strides=2 로 두면 건너뛰며 훑어서 출력 크기가 대략 절반이 된다
#   대신 훑는 영역이 겹치지 않아 정보가 듬성듬성해진다 -> 크기를 줄일 목적이 아니면 권하지 않는다
#   나누어떨어지지 않는 자투리는 그냥 버려진다 (valid 기준)
#
# 출력 크기 계산
#   padding='same'  : ceil(입력 / strides)
#   padding='valid' : floor((입력 - 커널) / strides) + 1

import numpy as np
import pandas as pd

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1), # strides/padding 둘 다 기본값이면 (9,9,10)
                #  strides=1,                        # default
                 strides=2,                          # 두 칸씩 건너뛴다 -> 중첩이 없어서 크기 줄일 때만 쓴다
                #  padding='valid',                  # default (패딩 없음)
                 padding='same',                     # 패딩 적용 -> strides=1 이었다면 (None, 10, 10, 10)
                 ))                                  # 실제 출력 : (None, 5, 5, 10)   same + strides=2 -> ceil(10/2) = 5

model.add(Conv2D(filters=9, kernel_size=(3,3),       # 앞 층 출력이 5x5 -> 기본값이면 (3,3,9)
                #  strides=1,
                 strides=2,                          # 두 칸씩 건너뛴다. 나누어떨어지지 않는 자투리는 버린다
                #  padding='valid',
                 padding='valid',                    # 패딩 없음
                 ))                                  # 실제 출력 : (None, 2, 2, 9)   floor((5-3)/2) + 1 = 2

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 5, 5, 10)          50        <- (2x2x1 + 1) x 10
#
#  conv2d_1 (Conv2D)           (None, 2, 2, 9)           819       <- (3x3x10 + 1) x 9
#
# =================================================================
# Total params: 869
# Trainable params: 869
# Non-trainable params: 0

# 파라미터 수는 padding / strides 와 무관하다 (커널 크기와 채널 수로만 정해진다)
# padding / strides 가 바꾸는 것은 출력 가로 세로뿐이다