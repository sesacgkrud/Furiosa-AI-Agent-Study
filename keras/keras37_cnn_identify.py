# keras37_cnn_identify.py
# keras36_cnn1.py 베이스 - Conv2D 에 넘긴 숫자들이 각각 무엇을 뜻하는지 하나씩 확인한다
#
# Conv2D(10, (2,2), input_shape=(5,5,1))
#   10          = 필터(커널) 개수 -> 출력 채널 수가 된다
#   (2,2)       = 커널 크기 (kernel_size). 이 크기의 창이 이미지 위를 한 칸씩 훑는다
#   input_shape = (가로, 세로, 채널). 첫 층에만 쓴다
#                 마지막 값이 채널 : 흑백 1 / 컬러 3
#                 3차원이라 input_dim 으로는 표현할 수 없다
#
# 출력 크기  : padding 없이 (2,2) 커널이 지나가면 가로 세로가 1씩 줄어든다  5x5 -> 4x4 -> 3x3
# 파라미터 수 : (커널 가로 x 커널 세로 x 입력 채널 + 1) x 필터 개수
#               conv2d   : (2 x 2 x 1  + 1) x 10 =  50   <- 입력 채널 1
#               conv2d_1 : (2 x 2 x 10 + 1) x  5 = 205   <- 입력 채널 10 (앞 층의 필터 개수)
#               + 1 은 필터마다 하나씩 붙는 bias

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))
model.add(Conv2D(5, (2,2)))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50
#
#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205
#
# =================================================================
# Total params: 255
# Trainable params: 255
# Non-trainable params: 0
