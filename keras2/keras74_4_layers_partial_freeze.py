# keras74_3_layers_partial_freeze.py 베이스
# [목적] Sequential 에 넣지 않고 VGG16 모델 자체의 layers 를 보고, VGG16 안의 층 하나만 골라 동결한다
#  - VGG16 을 Sequential 에 넣으면 vgg16 (Functional) 한 줄로만 보이지만, VGG16 자체의 model.layers 는 19개 층이 다 보인다
#  - model.layers[11] = block4_conv1 (index 0 은 input_layer) -> 이 층 하나만 동결

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

# model.trainable = False

model.layers[11].trainable = False # VGG16 의 index 11 = block4_conv1 하나만 동결
model.summary()
# Total params: 14,714,688 (56.13 MB)
# Trainable params: 13,534,528 (51.63 MB)
# Non-trainable params: 1,180,160 (4.50 MB)   <- block4_conv1 (3 x 3 x 256 + 1) x 512 = 1,180,160

print(len(model.weights))           # 26 = 13 (Conv2D) * 2 (w, b)
print(len(model.trainable_weights)) # 24 = 동결한 block4_conv1 의 w, b 2개가 빠진다

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
#       Layer Type    Layer Name  Layer Trainable
# 0     InputLayer   input_layer             True
# 1         Conv2D  block1_conv1             True
# 2         Conv2D  block1_conv2             True
# 3   MaxPooling2D   block1_pool             True
# 4         Conv2D  block2_conv1             True
# 5         Conv2D  block2_conv2             True
# 6   MaxPooling2D   block2_pool             True
# 7         Conv2D  block3_conv1             True
# 8         Conv2D  block3_conv2             True
# 9         Conv2D  block3_conv3             True
# 10  MaxPooling2D   block3_pool             True
# 11        Conv2D  block4_conv1            False
# 12        Conv2D  block4_conv2             True
# 13        Conv2D  block4_conv3             True
# 14  MaxPooling2D   block4_pool             True
# 15        Conv2D  block5_conv1             True
# 16        Conv2D  block5_conv2             True
# 17        Conv2D  block5_conv3             True
# 18  MaxPooling2D   block5_pool             True
# MaxPooling2D · InputLayer 는 가중치가 없어서 trainable 이 True 여도 훈련할 것이 없다
