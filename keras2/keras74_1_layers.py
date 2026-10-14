# [목적] VGG16 을 층으로 넣은 모델에서 model.weights / model.trainable_weights 개수를 세어
#        vgg16.trainable = False 가 어느 가중치까지 동결하는지 확인한다 (Day28 keras71 의 가중치 개수 세기를 VGG16 에 적용)
#
#  - 가중치는 층마다 kernel(W) · bias(b) 2개씩 -> 가중치가 있는 층 수 x 2
#  - VGG16 (include_top=False) : 가중치가 있는 층은 Conv2D 13개 (MaxPooling2D 5개 · InputLayer 는 가중치 없음)
#  - 붙인 층 : Flatten (가중치 없음) + Dense 2개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.applications import VGG16

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

vgg16.trainable = False
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

print(len(model.weights))           # 30 = 13 (vgg16 Conv2D) * 2 (w, b) + 2 (dense) * 2 (w, b)
print(len(model.trainable_weights)) # 4  = Dense 2개 * 2 (w, b) -> vgg16 의 26개는 동결되어 빠진다
