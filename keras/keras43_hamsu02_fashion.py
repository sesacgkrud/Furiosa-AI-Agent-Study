# keras43_hamsu02_fashion.py
# keras41_dnn2_fashion.py (Sequential DNN) 를 함수형(Model) 으로 바꾼 파일
#
# 함수형은 모델 종류(DNN / CNN)와 상관없이 '#2. 모델 구성' 을 쓰는 문법만 다르다
#  Sequential : model.add() 로 위에서 아래로 자동 연결
#  함수형     : Input 층을 따로 만들고, 층마다 뒤에 (앞층) 을 붙여 직접 연결한 뒤
#               Model(inputs=시작, outputs=끝) 으로 범위를 정해 완성
# -> 층 구성을 그대로 옮겼으므로 keras41_dnn2 와 Total params 가 같아야 제대로 변환한 것

import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model                   # Model -> 함수형 모델
from tensorflow.keras.layers import Dense, Dropout, Input   # Input -> 함수형 모델의 입력층
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test))   # 255 0

########## 스케일링 ##########
# keras41_dnn2 와 같은 조건으로 비교하려고 베이스 파일처럼 주석 처리해 두었다
# x_train = (x_train - 127.5)/127.5
# x_test = (x_test - 127.5)/127.5
# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

########## 2차원으로 reshape ##########
# Dense 는 (샘플, 컬럼) 2차원 입력 -> 28x28 그림 한 장을 784개 컬럼으로 편다
x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)
print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

########## y 원핫 인코딩 ##########
ohe = OneHotEncoder(sparse_output=False)   # 희소행렬이 아니라 numpy 배열로 받는다

y_train = y_train.reshape(-1,1)    # OneHotEncoder 는 2차원만 받으므로 (60000,) -> (60000, 1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)   # train 으로 기준(클래스 목록)을 만든다
y_test = ohe.transform(y_test)         # test 는 그 기준으로 변환만

print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)

#2. 모델 구성 (함수형) - keras41_dnn2 와 층 / Dropout 위치가 똑같다
# 변수 이름은 층마다 새로 붙인다 (dense1 / dense2 ...)
# 앞 층과 같은 이름을 다시 쓰면 그 변수가 덮어써져서 해당 층이 모델 연결에서 빠진다
input1 = Input(shape=(784,))                      # 입력층 : 컬럼 784개 (Sequential 의 input_shape=(784,))
dense1 = Dense(512, activation='relu')(input1)    # (input1) -> input1 뒤에 연결
dense2 = Dense(256, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
drop1 = Dropout(0.4)(dense4)                      # dense4 출력의 40% 를 훈련 때마다 끈다

dense5 = Dense(64, activation='relu')(drop1)      # Dropout 다음 층은 drop1 에 연결
dense6 = Dense(64, activation='relu')(dense5)
dense7 = Dense(32, activation='relu')(dense6)
dense8 = Dense(32, activation='relu')(dense7)
drop2 = Dropout(0.3)(dense8)

dense9 = Dense(16, activation='relu')(drop2)
dense10 = Dense(16, activation='relu')(dense9)
drop3 = Dropout(0.2)(dense10)

# 다중 분류(원핫) 출력층은 softmax 가 반드시 있어야 칸별 확률(합 = 1)이 나온다
output1 = Dense(10, activation='softmax')(drop3)

model = Model(inputs=input1, outputs=output1)   # 시작(input1) ~ 끝(output1) 범위를 정해서 모델 완성

model.summary()
# Total params: 599,178  <- keras41_dnn2(Sequential) 과 같아야 제대로 변환한 것
#  -> 784 x 512 + 512 = 401,920 으로 첫 층 한 곳이 전체의 67% 를 차지한다

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

# 목표가 정확도라서 val_loss 가 아니라 val_acc 를 기준으로 잡는다
#  - loss 는 낮을수록 좋고 acc 는 높을수록 좋으므로 mode 를 'max' 로 같이 바꿔야 한다
es = EarlyStopping(
    monitor='val_acc',      # val_loss 기준이면 loss 가 가장 낮은 epoch 를 고르는데, 그 epoch 가 acc 최고점은 아니다
    mode='max',             # acc 는 높을수록 좋으므로 max
    patience=25,            # acc 는 들쭉날쭉해서 loss 기준보다 patience 를 넉넉히 준다
    restore_best_weights=True,  # 멈춘 뒤 val_acc 가 가장 높았던 가중치로 되돌린다
    verbose=1,
)

start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
          )

end_time = time.time()


#4. 평가 예측
print('========== model.evaluate ==========')
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

# predict 결과는 (10000, 10) 짜리 확률 -> argmax 로 가장 큰 자리(예측 숫자)를 뽑는다
# y_test 도 원핫이라 똑같이 되돌려서 accuracy_score 에 넣는다
y_predict = model.predict(x_test)
y_predict=  np.argmax(y_predict, axis=1).reshape(-1,1) # reshape 없어도 됨
y_test = np.argmax(y_test, axis=1).reshape(-1,1)       # reshape 없어도 됨

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time, 2), '초')

# ===== CPU 기록 =====
# loss : 0.26822808384895325
# acc : 0.9261999726295471
# accuracy_score : 0.9262
# 소요 시간 : 951.0 초

# ===== CPU 기록 =====  <- MaxPooling 적용
# loss : 0.2809465229511261
# acc : 0.9279000163078308
# accuracy_score : 0.9279
# 소요 시간 : 763.36 초

# ===== GPU 기록 =====
# loss : 0.26894062757492065
# acc : 0.9258000254631042
# accuracy_score : 0.9258
# 소요 시간 : 380.58 초

# ===== GPU 기록 =====  <- MaxPooling 적용
# loss : 0.2832645773887634
# acc : 0.9229999780654907
# accuracy_score : 0.923
# 소요 시간 : 116.61 초

# ===== GPU 기록 ===== <- GlobalAveragePooling2D 적용
# loss : 0.2701389491558075
# acc : 0.9247999787330627
# accuracy_score : 0.9248
# 소요 시간 : 118.45 초

# ===== GPU 기록 ===== <- DNN
# Epoch 75: early stopping
# loss : 0.43086478114128113
# acc : 0.866100013256073
# accuracy_score : 0.8661
# 소요 시간 : 77.2 초

# ===== GPU 기록 ===== <- 함수형 적용
# loss : 0.48146337270736694
# acc : 0.8787999749183655
# accuracy_score : 0.8788
# 소요 시간 : 102.52 초