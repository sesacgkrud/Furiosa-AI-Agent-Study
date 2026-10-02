from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

import numpy as np
import time
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,     # 단어사전 갯수, 빈도수의 갯수가 높은 단어 순으로 1,000개 뽑겠다.
    # maxlen=1000,      # 단어 갯수의 최대 길이 제한 1000
    test_split=0.2,
)

print(x_train) # list([1, 53, 751, 26, 14, ... 13, 78, 80, 467, 17, 12])
print(x_train.shape, y_train.shape)     # (8982,) (8982,) -> 전체 데이터 갯수
print(x_test.shape, y_test.shape)       # (2246,) (2246,)
print(y_train)                          # [ 3  4  3 ... 25  3 25]
print(np.unique(y_train))               # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
                                        #  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print(type(x_train))                    # <class 'numpy.ndarray'>
print(type(x_train[0]))                 # <class 'list'>

print(len(x_train[0]), len(x_train[1])) # 87 56
print('뉴스기사의 최대 길이 :', max(len(i) for i in x_train))         # 2376
print('뉴스기사의 최소 길이 :', min(len(i) for i in x_train))         # 13
print('뉴스기사의 평균 길이 :', sum(map(len, x_train))/len(x_train))  # 145.53

# 전처리 (pad_sequences)
# padding / truncating 기본값 pre -> 앞을 0으로 채우고, 긴 기사는 앞쪽을 자름
#   -> LSTM 은 마지막 단어 쪽 상태를 출력하므로 실제 단어가 뒤에 붙어 있는 pre 가 유리
x_train = pad_sequences(x_train, maxlen=200) # truncating = pre (default)
x_test = pad_sequences(x_test, maxlen=200)   # truncating = pre (default)
print(x_train.shape, x_test.shape)           # (8982, 200) (2246, 200)

# y 원핫인코딩
y_train = to_categorical(y_train)            # 0 ~ 45 -> 46칸 One-Hot
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)           # (8982, 46) (2246, 46)

# 2. 모델 구성
model = Sequential()
model.add(Embedding(1000, 200))              # input_dim = num_words 1000 (단어 번호 0 ~ 999), output_dim = 단어 하나를 200칸 벡터로
model.add(LSTM(128))                         # (None, 200, 200) -> (None, 128)  마지막 시점 출력만 -> 2차원
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))   # 클래스 46개 -> 다중분류
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
)

path = './_save/keras62/'

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,                # val_loss 가 가장 좋았던 epoch 의 모델만 덮어쓰며 저장
    filepath=path + 'keras62_1_mcp.keras',
    verbose=1,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64,
          validation_split=0.2,
          verbose=1,
          callbacks=[es, mcp],
          )
end_time = time.time()

# 가중치만 저장 (구조는 저장 안 됨 -> 불러올 때는 같은 모델을 만든 뒤 load_weights)
# restore_best_weights=True 라서 val_loss 가 가장 좋았던 epoch 의 가중치가 저장됨 -> mcp 와 같은 가중치
model.save_weights(path + 'keras62_1_save.weights.h5')

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss[0])
print("acc :", round(loss[1], 4))

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)   # 46칸 확률 -> 가장 큰 칸의 번호
y_true = np.argmax(y_test, axis=1)         # y_test 도 원핫 (2246, 46) -> 정답 번호로 되돌림
print("acc_score :", accuracy_score(y_true, y_predict))
print("걸린 시간 :", round(end_time - start_time, 2), "초")

# [목표] acc_score : 0.67 이상

# loss : 1.1545485258102417
# acc : 0.7418
# acc_score : 0.7417631344612645
# 걸린 시간 : 64.92 초