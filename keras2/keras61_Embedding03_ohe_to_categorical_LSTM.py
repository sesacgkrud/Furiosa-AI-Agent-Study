# keras61_Embedding03_ohe_to_categorical_LSTM.py
# [목표] 단어 번호를 to_categorical 로 One-Hot 해서 (15, 5, 31) 로 만들고 LSTM 으로 분류한다.
#   - One-Hot -> 단어마다 자기 칸이 따로 있어서 번호 크기 문제가 사라짐 (바보 26 과 잘생겼다 28 이 완전히 다른 단어로 구별됨)
#   - 대신 단어 하나가 31칸짜리 벡터가 됨 (단어 종류가 늘어날수록 칸 수도 그만큼 늘어남)
#     자세한 이유와 01 ~ 04 를 이 순서로 진행한 의도는 keras61_Embedding04_important.py 맨 위에 정리
#   - 흐름 : 토큰화 -> padding (15, 5) -> 예측 문장과 이어 붙여 One-Hot (16, 5, 31) -> 다시 나눔 -> 분리 -> LSTM -> 평가 -> 예측
#   - 결과 : test acc 0.4 ~ 0.8 (test 문장 단어 대부분이 훈련 때 한 번도 안 나온 단어라 맞힐 근거가 없음)
#            '개똥이 잘생겼다' 는 거의 0 (부정) -> 훈련 데이터에서 '개똥이' 가 부정 문장 ('개똥이 바보') 에만 나오기 때문

# [참고] 이 파일에서 쓰지 않는 import : Dropout, MinMaxScaler, StandardScaler (실행에는 영향 없음)
import numpy as np
import time

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

#1. 데이터
docs = [                                   # 훈련용 문장 15개 (짧은 영화 리뷰) -> 문장 하나가 데이터 1개
    '너무 재미있다', '참 최고에요', '참 잘 만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '벌로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밌네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다',
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])   # 1 = 긍정, 0 = 부정 (긍정 7개, 부정 8개)

token = Tokenizer()                        # 단어 사전을 만들고 문장을 번호로 바꿔 주는 도구 (인스턴스 생성)
token.fit_on_texts(docs)                   # 15문장 전체로 단어 사전을 만듦 -> 단어 31개 (번호 1 ~ 31)
# print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘': 5,
# '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10,
# '한': 11, '번': 12, '더': 13, '보고': 14, '싶어요': 15,
# '글쎄': 16, '벌로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20,
# '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '개똥이': 25,
# '바보': 26, '말똥이': 27, '잘생겼다': 28, '길동이': 29, '또': 30, '구라친다': 31}

x = token.texts_to_sequences(docs)         # 위에서 만든 단어 사전으로 문장을 단어 번호 리스트로 바꿈 (문장 15개 -> 리스트 15개)
print(x)                                   # 문장마다 단어 수가 달라서 리스트 길이가 제각각 (1개 ~ 5개) -> 아래 padding 이 필요한 이유
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15],[16], [17],
# [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26], [27, 28], [29, 30, 31]]

#################### padding ####################
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 문장마다 단어 수가 다름 (1개 ~ 5개) -> 모델은 같은 길이만 받을 수 있어서 모자란 자리를 0 으로 채움
padded_x = pad_sequences(x,                # padding 할 대상
                         padding='post',   # post -> 뒤를 0으로 채움 / default 는 pre (앞을 0으로 채움)
                         maxlen=5,         # 문장 길이를 5로 맞춤 -> 5보다 긴 문장은 잘림
                         truncating='post' # 잘라낼 때 뒤쪽을 자름 / default 는 pre (앞쪽 잘림)
                         )

print(padded_x)                            # [[ 2  3  0  0  0] [ 1  4  0  0  0] ... -> 모자란 뒷자리가 0 으로 채워짐
print(padded_x.shape)                      # (15, 5)

#################### 새 문장 (예측용) ####################
x_predict = ['개똥이 잘생겼다']           # 훈련 문장에 없는 새 문장 ('개똥이', '잘생겼다' 단어는 각각 훈련 데이터에 있음)

# 훈련 때와 똑같이 토큰화 -> padding 순서로 맞춰야 함
x_predict = token.texts_to_sequences(x_predict)   # fit_on_texts 는 다시 하지 않음 (단어 번호가 바뀜)
print(x_predict)                                  # [[25, 28]]
x_predict = pad_sequences(x_predict, padding='post', maxlen=5, truncating='post')   # 훈련 데이터와 같은 조건으로 padding
print(x_predict)                                  # [[25 28  0  0  0]]

#################### One-Hot Encoding (to_categorical) ####################
# 단어 번호를 숫자 크기 그대로 쓰면 26(바보) 과 28(잘생겼다) 이 비슷한 값이 됨
# -> 단어 하나를 31칸짜리 벡터로 바꿔서 단어끼리 크기 차이가 없게 만듦

# x_predict 만 따로 to_categorical 하면 가장 큰 번호가 28 이라 29칸만 생김 (훈련 데이터는 32칸)
# -> 훈련 데이터와 이어 붙여서 한 번에 바꾼 뒤 다시 나눔
all_x = np.concatenate([padded_x, x_predict])
print(all_x.shape)                         # (16, 5) -> 15문장 + 예측 문장 1개

all_x = to_categorical(all_x)
print(all_x.shape)                         # (16, 5, 32) -> 0번 칸까지 생김 (0 = padding)

all_x = all_x[:, :, 1:]                    # 0번 칸 제거 -> padding 자리는 전부 0 인 벡터가 됨
print(all_x.shape)                         # (16, 5, 31)

padded_x = all_x[:15]                      # 앞의 15개 = 훈련 / 평가용
x_predict = all_x[15:]                     # 마지막 1개 = 개똥이 잘생겼다
print(padded_x.shape, x_predict.shape)     # (15, 5, 31) (1, 5, 31) -> (데이터 수, timesteps, feature)

x_train, x_test, y_train, y_test = train_test_split(
    padded_x, labels,
    train_size=0.7,
    random_state=14,                       # 02 와 같은 분할 (결과 비교용)
    stratify=labels,
)

print(x_train.shape, x_test.shape) # (10, 5, 31) (5, 5, 31)
print(y_train.shape, y_test.shape) # (10,) (5,)

# One-Hot 은 이미 0 과 1 뿐이라 스케일링 하지 않음

#2. 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(5, 31)))     # (timesteps, feature) -> 단어 하나를 31칸짜리 벡터로 받음 (02 는 1칸)
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'],
              )

# 훈련 데이터가 10개뿐이라 validation 을 떼지 않고 loss 로 감시
# patience 를 크게 줘서 loss 를 끝까지 낮춤 -> 예측값이 0.5 근처에서 멈추지 않고 0 또는 1 쪽으로 확실해짐
es = EarlyStopping(
    monitor='loss',                        # 훈련 loss 를 감시
    mode='min',                            # loss 는 낮을수록 좋음
    patience=100,                          # 100 epoch 동안 최저 loss 가 갱신되지 않으면 멈춤
    restore_best_weights=True,             # 멈춘 시점이 아니라 loss 가 가장 낮았던 epoch 의 가중치로 되돌림
)

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=2,   # 훈련 10개를 2개씩 -> 1 epoch 에 가중치 5번 갱신
          verbose=1,
          callbacks=[es],
          )
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)      # [loss, acc] 리스트로 돌려줌 -> loss[0] = loss, loss[1] = acc
print("loss :", loss[0])
print("acc :", round(loss[1], 4))

y_predict = model.predict(x_test)          # test 5문장의 sigmoid 확률 (0 ~ 1)
y_predict = np.round(y_predict)            # 0.5 기준 반올림 -> 0 / 1
acc_score = accuracy_score(y_test, y_predict)   # 직접 계산한 정확도 -> 위의 evaluate acc 와 같은 값

print("acc_score :", acc_score)
print("걸린 시간 :", round(end_time - start_time, 2), "초")

#################### 새 문장 예측 ####################
y_pred = model.predict(x_predict)                 # (1, 1) 짜리 sigmoid 확률 -> 1 에 가까우면 긍정, 0 에 가까우면 부정
# [참고] One-Hot 은 '바보' 와 '잘생겼다' 를 완전히 다른 단어로 구별함
#        -> 그런데 훈련 데이터에서 '개똥이' 는 부정 문장에만 나와서 '개똥이' 를 부정 신호로 배움 -> 거의 0 (부정)
#        -> 01, 02 처럼 값이 0 ~ 1 사이를 오가지 않고 부정 쪽으로 일정하게 나옴
print("개똥이 잘생겼다 :", round(y_pred[0][0], 4))
