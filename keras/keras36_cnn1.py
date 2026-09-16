# keras36_cnn1.py
# CNN 첫 실습 : Conv2D 층을 쌓고 summary 로 출력 모양과 파라미터 수만 확인한다
# Conv2D(필터 개수, (커널 크기), input_shape=(가로, 세로, 채널))
#  - 커널(필터)이 이미지 위를 한 칸씩 훑으면서 특징을 뽑는다
#  - (2,2) 커널이 지나가면 가로 세로가 1씩 줄어든다 : 5x5 -> 4x4 -> 3x3
#  - 파라미터 수 = (커널 가로 x 커널 세로 x 입력 채널 + 1) x 필터 개수

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))   # 10 = 필터 개수, (2,2) = 커널 크기, (5,5,1) = (가로, 세로, 채널)
model.add(Conv2D(5, (2,2)))                         # 두 번째 층부터는 input_shape 를 쓰지 않는다 (앞 층 출력이 그대로 입력)

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50        
                                                                 
#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205       
                                                                 
# =================================================================
# Total params: 255
# Trainable params: 255
# Non-trainable params: 0