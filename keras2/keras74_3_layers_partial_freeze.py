# keras74_2_layers.py 베이스
# [목적] 모델 전체나 vgg16 통째가 아니라 원하는 층 하나만 골라 동결한다 (부분 동결)
#  - model.layers[인덱스].trainable = False : 인덱스는 model.layers 순서 (vgg16 = 0, Flatten = 1, Dense(100) = 2, ...)
#  - Dense 를 5개로 늘리고 index 2 인 첫 번째 Dense(100) 만 동결해서 summary · 가중치 개수 · 표로 확인한다

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
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(10, activation='softmax'))

model.layers[2].trainable = False # Flatten() 포함한 갯수로 index 2에 해당하는 layers 동결
model.summary()
# Total params: 14,788,738 (56.41 MB)
# Trainable params: 22,750 (88.87 KB)         <- Dense(90) 9,090 + Dense(80) 7,280 + Dense(70) 5,670 + Dense(10) 710
# Non-trainable params: 14,765,988 (56.33 MB) <- vgg16 14,714,688 + 동결한 Dense(100) 51,300

print(len(model.weights))           # 36 = 13 (vgg16 Conv2D) * 2 (w, b) + 5 (dense) * 2 (w, b)
print(len(model.trainable_weights)) # 8  = 동결하지 않은 Dense 4개 * 2 (w, b)

import pandas as pd
pd.set_option('max_colwidth', None)

layers = [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in model.layers
]

results = pd.DataFrame(
    layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']
)

print(results)
#    Layer Type Layer Name  Layer Trainable
# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense            False
# 3       Dense    dense_1             True
# 4       Dense    dense_2             True
# 5       Dense    dense_3             True
# 6       Dense    dense_4             True

