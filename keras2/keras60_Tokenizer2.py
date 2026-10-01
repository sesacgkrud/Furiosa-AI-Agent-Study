# keras60_Tokenizer2.py
# [목표] 문장 2개를 하나의 단어 사전으로 토큰화하고, np.concatenate 로 이어 붙여서 One-Hot Encoding 3가지를 만든다.
#   - 문장이 여러 개면 fit_on_texts 에 한꺼번에 넣어서 "공통 단어 사전" 을 만들어야 함
#     -> 두 문장에 같이 나온 '마구', '잘생겼다' 가 같은 번호를 받음
#   - 문장마다 길이가 달라서 (14개, 10개) 바로 배열로 만들 수 없음 -> 이번엔 이어 붙여서 해결
#     (문장 구분 없이 단어 24개를 한 줄로 세운 것 / 문장 단위를 유지하는 방법은 keras61 의 padding)

from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '개똥이는 기관사를 좋아한다. 말똥이는 잘생겼다. 길동이는 마구 마구 더 잘생겼다.'

token = Tokenizer()                   # 인스턴스(객체) = 클래스(), 인스턴스 생성
                                      # Tokenizer = 단어 사전을 만들고 (fit_on_texts) 문장을 번호로 바꿔 주는 (texts_to_sequences) 도구
token.fit_on_texts([text1, text2])    # 두 문장을 같이 넣어서 하나의 단어 사전을 만듦

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '잘생겼다': 4, '나는': 5, '지금': 6, '맛있는': 7, '김밥을': 8,
#  '엄청': 9, '먹었다': 10, '개똥이는': 11, '기관사를': 12, '좋아한다': 13, '말똥이는': 14, '길동이는': 15, '더': 16}
# -> '마구' 는 두 문장 합쳐서 6번 나와서 1번 / '잘생겼다' 는 2번 나와서 진짜, 매우 다음인 4번
print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 6),
#              ('먹었다', 1), ('개똥이는', 1), ('기관사를', 1), ('좋아한다', 1), ('말똥이는', 1), ('잘생겼다', 2), ('길동이는', 1), ('더', 1)])

x = token.texts_to_sequences([text1, text2])   # 공통 단어 사전으로 두 문장을 각각 번호 리스트로 바꿈 (문장 2개 -> 리스트 2개)
print(x)                              # [[5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10], [11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]

# 두 문장의 길이가 달라서 바로 (2, n) 배열로 만들 수 없음 -> np.concatenate 로 이어 붙여 1차원으로
x = np.concatenate(x)
print(x, x.shape)                     # (24,)

##### One-Hot Encoding 3가지 만들기 #####
# 단어 24개 x 단어 종류 16개 -> 세 방법 모두 (24, 16)

########## OneHot Encoding 1. (tensorflow) ##########
from tensorflow.keras.utils import to_categorical
x1 = to_categorical(x)                # 가장 큰 번호 (16) 를 보고 0 ~ 16 = 17칸을 만듦
print('\n===== to_categorical One-Hot Encoding =====')
print(x1)
print(x1.shape)                       # (24, 17) -> 0번 인덱스 열까지 생김 (단어 인덱스는 1부터 시작)
x1 = x1[:, 1:]                        # 쓰지 않는 0번 열 제거
print(x1.shape)                       # (24, 16)


########## OneHot Encoding 2. (pandas) ##########
x2 = pd.get_dummies(x, dtype=int)     # 1차원 데이터를 넣어야 함 / dtype=int -> True, False 대신 1, 0 으로 출력
print('\n===== pandas One-Hot Encoding =====')
print(x2)                             # 열 이름이 단어 번호 (1 ~ 16) 인 DataFrame
print(x2.shape)                       # (24, 16)


########## OneHot Encoding 3. (sklearn) ##########
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)   # sparse_output=False -> 바로 볼 수 있는 numpy 배열로 출력

# 2차원 형태로 넣어야 함
x3 = x.reshape(-1, 1)                 # (24,) -> (24, 1)
x3 = ohe.fit_transform(x3)

print('\n===== sklearn One-Hot Encoding =====')
print(x3)
print(x3.shape)                       # (24, 16)
