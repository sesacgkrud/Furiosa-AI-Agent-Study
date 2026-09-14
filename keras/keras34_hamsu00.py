from tensorflow.keras.models import Sequential, Model # Model -> 함수형 모델이라고 생각
from tensorflow.keras.layers import Dense, Dropout, Input

#2-1. 모델 구성 (순차적)
model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dropout(0.2))

model.add(Dense(9))
model.add(Dropout(0.2))

model.add(Dense(1))

model.summary()

########## ########## ########## ########## ##########

#2-2. 모델 구성 (함수형)
input1 = Input(shape=(3,))
dense1 = Dense(10, name='ys1')(input1) # name -> 임의로 지은 것, 없어도 됨 / (input1) -> 으로 연결
drop1 = Dropout(0.2)(dense1)

dense2 = Dense(9, name='ys2')(drop1)
drop2 = Dropout(0.2)(dense2)

output1 = Dense(1)(drop2)

# 모델에 대한 정의
model2 = Model(inputs=input1, outputs=output1) # 어디서부터 어디까지인지 범위 정의

model2.summary()