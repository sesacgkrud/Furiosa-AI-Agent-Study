from tensorflow.keras.datasets import imdb

import numpy as np
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=1000,     # 많이 나온 단어 1,000개만 사용 -> 단어 번호 0 ~ 999
)

# print(x_train)                       # list([1, 17, 6, 194, 337, ... 22, 4, 204, 131, 9])]
print(x_train.shape, y_train.shape)    # (25000,) (25000,)
print(x_test.shape, y_test.shape)      # (25000,) (25000,)
print(y_train)                         # [1 0 0 ... 0 1 0]
print(np.unique(y_train))              # [0 1]

print(type(x_train))                   # <class 'numpy.ndarray'>
print(type(x_train[0]))                # <class 'list'>

print('리뷰의 최대 길이 :', max(len(i) for i in x_train))         # 2494
print('리뷰의 최소 길이 :', min(len(i) for i in x_train))         # 11
print('리뷰의 평균 길이 :', sum(map(len, x_train))/len(x_train))  # 238.71

# 전처리 (pad_sequences)
# 평균 길이 238.71 -> maxlen=250 이면 대부분의 리뷰를 거의 다 담음, 기본값 pre -> 앞을 0으로 채우고 긴 리뷰는 앞쪽을 자름
x_train = pad_sequences(x_train, maxlen=250)
x_test = pad_sequences(x_test, maxlen=250)
print(x_train.shape, x_test.shape)     # (25000, 250) (25000, 250)

# y 는 이미 0 / 1 -> 이진분류라서 원핫인코딩 하지 않음

# 2. 모델 구성
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=100, input_length=250))   # num_words=1000 -> 단어 번호 0 ~ 999
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 긍정 / 부정 -> 이진분류
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
    restore_best_weights=True,
)

path = './_save/keras62/'

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=path + 'keras62_2_mcp.keras',
    verbose=1,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128,
          validation_split=0.2,
          verbose=1,
          callbacks=[es, mcp],
          )
end_time = time.time()

model.save_weights(path + 'keras62_2_save.weights.h5')   # 가중치만 저장 (restore_best_weights 로 되돌린 최고 epoch 가중치)

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss[0])
print("acc :", round(loss[1], 4))

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)            # 0.5 기준 -> 0 / 1
print("acc_score :", accuracy_score(y_test, y_predict))
print("걸린 시간 :", round(end_time - start_time, 2), "초")

# [목표] acc_score : 0.6 이상

# loss : 0.32410532236099243
# acc : 0.8619
# acc_score : 0.86188
# 걸린 시간 : 35.69 초