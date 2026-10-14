# keras74_1_layers.py 베이스
# [목적] 개수만 보던 keras74_1 에서 한 걸음 더 나가 model.layers 를 돌면서 층마다 trainable 여부를 pandas 표로 확인한다
#  - model.layers : 모델에 들어 있는 층 리스트 (vgg16 은 통째로 층 1개 Functional 로 들어간다)
#  - type(layer).__name__ : 층 종류 이름 (Functional / Flatten / Dense) , layer.name : 층 이름 , layer.trainable : 훈련 여부
#  - pd.set_option('max_colwidth', None) : 표의 칸 내용이 길어도 ... 으로 자르지 않고 다 보여준다

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

print(len(model.weights))           # 30
print(len(model.trainable_weights)) # 4

import pandas as pd
pd.set_option('max_colwidth', None)

# 층마다 (종류, 이름, trainable) 튜플을 만들어 리스트로 모은다 (리스트 컴프리헨션)
layers = [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in model.layers
]

results = pd.DataFrame(
    layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']
)

print(results)
# vgg16.trainable = False 로 vgg16 한 줄만 False, 직접 붙인 Flatten · Dense 는 True
#    Layer Type Layer Name  Layer Trainable
# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense             True
# 3       Dense    dense_1             True

