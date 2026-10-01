# keras61_Embedding04_important.py
# [목표] Embedding 층의 사용법과 파라미터 수를 확인한다. (model.summary 까지 / 훈련은 다음 단계)
#   - Embedding : 단어 번호 (정수) 를 받아서, 그 단어만의 "학습되는 벡터" (output_dim 칸) 로 바꿔 주는 층
#   - 입력 (15, 5) 단어 번호 그대로 -> 출력 (15, 5, 100) 단어 하나당 100칸짜리 벡터
#
# ============================== keras61 01 ~ 04 를 이 순서로 진행한 이유 (선생님 의도) ==============================
# 01 ~ 03 은 실제로 쓰는 방법이 아님 -> "단어를 숫자로 어떻게 표현해야 하는가" 를 한 단계씩 문제를 겪으면서
# Embedding 이 왜 필요한지 직접 확인하게 하려는 흐름
#
#   01 DNN  (15, 5)       : 단어 번호를 숫자 그대로 넣음
#       -> 문제 1. 번호 크기에 의미가 없음 (바보 26, 잘생겼다 28 -> 숫자로는 거의 같은데 뜻은 정반대)
#       -> 문제 2. DNN 은 단어 순서를 고려하지 못함
#
#   02 LSTM (15, 5, 1)    : 순서를 보도록 LSTM 으로 바꿈 -> 문제 2 해결
#       -> 하지만 단어 하나가 여전히 숫자 1개 (번호) -> 문제 1 은 그대로
#          ('개똥이 잘생겼다' 결과가 실행할 때마다 0 ~ 1 사이에서 흔들림)
#
#   03 One-Hot + LSTM (15, 5, 31) : 단어마다 자기 칸을 줌 -> 문제 1 해결 (단어끼리 크기 차이 없음)
#       -> 새로운 문제 3. 칸 수 = 단어 종류 수 -> 실제 문장 데이터는 단어가 수만 개라 단어 하나가 수만 칸이 됨
#                         그중 1칸만 1 이고 나머지는 전부 0 -> 메모리 낭비가 큼
#       -> 새로운 문제 4. 모든 단어 사이의 거리가 똑같음 -> '재미있다' 와 '재밌네요' 처럼 뜻이 비슷한 단어도
#                         '재미있다' 와 '지루해요' 만큼 서로 남남 -> 단어끼리 비슷한 정도를 표현할 수 없음
#
#   04 Embedding (15, 5) -> (15, 5, 100) : 문제 1, 3, 4 를 한 번에 해결
#       -> 단어 번호를 그대로 받아서 (One-Hot 을 직접 만들 필요 없음) 정해진 크기의 벡터로 바꿈
#          단어가 수만 개여도 벡터 크기는 output_dim (예: 100) 으로 고정 -> 문제 3 해결
#       -> 벡터 값은 가중치라서 훈련하면서 학습됨 -> 비슷하게 쓰이는 단어는 비슷한 벡터가 됨 -> 문제 4 해결
#       -> 사실 Embedding = "One-Hot (03) 다음에 bias 없는 Dense 를 붙인 것" 과 같은 계산
#          One-Hot 32칸 x Dense 100 = 3,200 개 가중치 = Embedding(32, 100) 의 Param 3,200 과 같음
#          다만 One-Hot 을 실제로 만들지 않고 번호로 해당 줄만 꺼내 쓰기 때문에 훨씬 가볍고 빠름
#   => 그래서 01 ~ 03 은 Embedding 의 필요성과 원리 (= One-Hot + Dense) 를 이해시키기 위한 사전 단계
# ======================================================================================================================

# [참고] 이 파일에서 쓰지 않는 import : time, Dropout, LSTM, to_categorical, EarlyStopping, accuracy_score, MinMaxScaler, StandardScaler, train_test_split (실행에는 영향 없음)
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
# Embedding 은 단어 번호 (정수) 를 그대로 받음 -> 01 ~ 03 과 달리 스케일링, reshape, One-Hot 을 하지 않음

#2. 모델 구성
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
model = Sequential()

#################### 임베딩1 ####################
# input_dim = 단어 사전 크기 = 단어 종류 31개 + padding 0번 1개 = 32 (= len(token.word_index) + 1)
#   이보다 작으면 큰 번호의 단어가 사전 범위를 벗어남
#   -> CPU 에서는 에러, GPU 에서는 에러 없이 0 벡터로 처리돼서 틀린 줄도 모르고 지나감
model.add(Embedding(input_dim=32, output_dim=100, input_length=5))
                    # 단어 사전 크기 (단어 종류 + 1), 출력 차원 (단어 하나를 몇 칸짜리 벡터로 만들지), 입력 시퀀스 길이 (padding 한 길이)
model.add(SimpleRNN(10))                   # Embedding 출력 (None, 5, 100) 이 3차원이라 RNN 계열에 바로 넣을 수 있음 (reshape 필요 없음)
model.add(Dense(1))                        # [참고] 이진 분류로 훈련하려면 activation='sigmoid' + binary_crossentropy 를 써야 함
model.summary()

######################### Embedding 적용 후 (SimpleRNN 없이 Embedding -> Dense) #########################
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3200

#  dense (Dense)               (None, 5, 1)              101

# =================================================================
# Total params: 3,301
# Trainable params: 3,301
# Non-trainable params: 0
# _________________________________________________________________
# Embedding Param = input_dim x output_dim = 32 x 100 = 3,200 (bias 없음 -> 단어마다 100칸짜리 벡터 1줄씩, 32줄짜리 표)
# Dense Param = (100 + 1) x 1 = 101
# Dense 는 마지막 축에만 적용됨 -> 단어 5개 각각에 출력 1개씩 (None, 5, 1) -> 문장 하나에 답 1개가 아니라서 RNN 이 필요

######################### simpleRNN 적용 후 #########################
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3200

#  simple_rnn (SimpleRNN)      (None, 10)                1110

#  dense (Dense)               (None, 1)                 11

# =================================================================
# Total params: 4,321
# Trainable params: 4,321
# Non-trainable params: 0
# _________________________________________________________________
# SimpleRNN Param = units x (units + feature + 1) = 10 x (10 + 100 + 1) = 1,110 (feature = Embedding 의 output_dim 100)
# Dense Param = (10 + 1) x 1 = 11
# SimpleRNN 이 단어 5개를 순서대로 읽고 마지막 상태 1개 (None, 10) 만 출력 -> 문장 하나에 답 1개 (None, 1)


# 임베딩2, 임베딩3 은 임베딩1 과 "같은 모델을 다르게 쓰는 방법" -> 한 번에 하나만 사용
#   -> 셋을 같이 쓰면 임베딩1 모델 뒤에 Embedding, SimpleRNN, Dense 가 계속 이어 붙음 (Dense 출력 뒤에 또 Embedding)
#   -> 써 보고 싶은 방법만 주석을 풀고 위의 임베딩1 을 주석 처리해서 사용

#################### 임베딩2 ####################
# model = Sequential()
# model.add(Embedding(input_dim=32, output_dim=100))   # input_length 생략 가능
#                     # 단어 사전 크기, 출력 차원
# model.add(SimpleRNN(10))
# model.add(Dense(1))
# model.summary()
# [참고] input_length 를 안 쓰면 summary 의 Embedding Output Shape 가 (None, None, 100) 으로 나옴
#        -> 문장 길이를 "아직 모름" 상태로 두고, 실제 데이터 (15, 5) 가 들어올 때 5 로 맞춰짐
#        -> Param 수는 input_dim x output_dim 이라 input_length 와 상관없이 똑같이 3,200


#################### 임베딩3 ####################
# model = Sequential()
# model.add(Embedding(32, 100))                 # input_dim, output_dim 이름 생략 가능 (순서 : 앞이 input_dim, 뒤가 output_dim)
# model.add(Embedding(32, 100, 5))              # 에러 -> ValueError: Could not interpret initializer identifier: 5
#                                               # 세 번째 자리는 input_length 가 아니라 embeddings_initializer (가중치 초기값 방법) 자리라서
#                                               # 5 를 초기값 방법 이름으로 읽으려다 실패함
# model.add(Embedding(32, 100, input_length=5)) # input_length 는 이름을 직접 써야 함
# model.add(SimpleRNN(10))
# model.add(Dense(1))
# model.summary()
