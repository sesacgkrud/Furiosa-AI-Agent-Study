# [목적] model.weights / model.trainable_weights 로 모델 안의 가중치를 직접 꺼내 보고,
#        model.trainable = False (동결) 를 하면 무엇이 달라지는지 확인한다
#
# 흐름
#  1) Dense 3 → 2 → 1 의 작은 모델을 만든다 (훈련은 하지 않는다)
#  2) model.weights 출력 → 층마다 kernel(가중치 W) 과 bias(b) 가 들어 있다
#  3) model.trainable = False 로 동결한 뒤 다시 출력 → 가중치는 그대로 있지만 "훈련할 가중치" 목록이 비어 버린다
#  4) summary 의 Trainable params / Non-trainable params 가 바뀌는 것을 확인한다
#  → 이 동결 기능이 keras73 에서 VGG16 같은 미리 훈련된 모델을 가져다 쓸 때 쓰인다

import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

print(tf.__version__)   # 2.9.3

random.seed(333)        # Python 초기값(랜덤값) 고정
np.random.seed(333)     # Numpy 초기값(랜덤값) 고정
tf.random.set_seed(333) # Tensorflow 초기값(랜덤값) 고정
# → 처음 만들어지는 가중치(랜덤값) 가 실행할 때마다 같게 나온다

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 2)                 8

#  dense_2 (Dense)             (None, 1)                 3

# =================================================================
# Total params: 17
# Trainable params: 17
# Non-trainable params: 0
# _________________________________________________________________
# 파라미터 = (입력 수 x 출력 수) + bias 출력 수
#  dense   : 1 x 3 + 3 = 6
#  dense_1 : 3 x 2 + 2 = 8
#  dense_2 : 2 x 1 + 1 = 3

print('=============== model.weights ===============')
print(model.weights)
# [dense/kernel (1, 3), dense/bias (3,), dense_1/kernel (3, 2), dense_1/bias (2,), dense_2/kernel (2, 1), dense_2/bias (1,)]
#  - kernel : 가중치 W, 처음에는 랜덤값  (dense/kernel = [[ 0.13603318, -0.03480017,  0.7743634 ]])
#  - bias   : 처음에는 모두 0          (dense/bias   = [0., 0., 0.])
print('========== model.trainable weights ==========')
print(model.trainable_weights)        # 아직 동결 전이라 model.weights 와 똑같다
print('=============================================')
print(len(model.weights))            # 6  ← 층 3개 x (kernel, bias)
print(len(model.trainable_weights))  # 6

######################## 동결 ########################
model.trainable = False              # 동결 ★★★★★
######################## 동결 ########################
print('=============================================')
print(len(model.weights))            # 6
print(len(model.trainable_weights))  # 0

print('========== model.trainable weights ==========')
print(model.trainable_weights)       # [] -> 훈련할 가중치 목록이 비어 있다 -> fit 을 해도 가중치가 바뀌지 않는다 (훈련 불가)

print('=============== model.weights ===============')
print(model.weights)                 # 가중치 6개는 그대로 존재 (값도 동결 전과 같다) -> 가중치는 있지만 훈련 X

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 2)                 8

#  dense_2 (Dense)             (None, 1)                 3

# =================================================================
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17
# _________________________________________________________________
#  -> Total params 는 17 그대로, 17개 전부 Non-trainable 로 옮겨 갔다
#  -> 초기 가중치로 고정된 상태 (fit 을 해도 절대 변하지 않음, keras71_trainable_weights2 에서 확인)
