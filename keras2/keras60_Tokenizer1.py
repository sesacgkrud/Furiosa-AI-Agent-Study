# keras60_Tokenizer1.py
# [목표] 문장 1개를 Tokenizer 로 단어 번호(정수)로 바꾸고, 그 번호를 One-Hot Encoding 3가지 방법으로 바꿔 본다.
#   - 자연어(글자)는 모델에 바로 넣을 수 없음 -> 단어마다 번호를 붙여서 숫자로 바꾸는 게 첫 단계 (토큰화)
#   - 그런데 번호는 "이름표" 일 뿐 크기에 의미가 없음 (마구=1, 진짜=2 라고 진짜가 마구의 2배가 아님)
#     -> 번호를 그대로 쓰지 않고 One-Hot 으로 바꿔서 단어끼리 크기 차이가 없게 만드는 게 이 파일의 핵심
#   - 흐름 : 문장 -> fit_on_texts (단어 사전 만들기) -> texts_to_sequences (문장을 번호로) -> One-Hot

from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'

token = Tokenizer()                   # 인스턴스(객체) = 클래스(), 인스턴스 생성
                                      # Tokenizer = 단어 사전을 만들고 (fit_on_texts) 문장을 번호로 바꿔 주는 (texts_to_sequences) 도구
token.fit_on_texts([text])            # 단어 사전 만들기 -> 리스트로 넣어야 함 (문장 여러 개를 넣는 게 기본 형태)
                                      # 띄어쓰기 기준으로 자르고, 마침표 같은 문장부호는 자동으로 지움 ('먹었다.' -> '먹었다')

print(token.word_index)               # {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
                                      # 번호 규칙 : 많이 나온 단어부터 1번 (마구 4번 -> 1, 진짜 / 매우 2번 -> 2, 3)
                                      #             나온 횟수가 같으면 문장에서 먼저 나온 단어가 앞 번호
                                      #             0번은 비워 둠 (나중에 padding 의 0 으로 씀)
print(token.word_counts)              # OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])
                                      # 단어별 나온 횟수 (문장에 나온 순서대로)

x = token.texts_to_sequences([text])  # 위에서 만든 단어 사전으로 문장의 단어를 하나씩 번호로 바꿈 (문장 1개 -> 리스트 1개)
print(x)                              # [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
                                      # -> 숫자로는 바뀌었지만 번호일 뿐이라 크기에 의미가 없음 -> 아직 모델에 넣기엔 부족

# x 는 2차원 리스트 [[...]] -> 1차원 (14,) 로 펴서 사용
x = np.array(x).reshape(-1,)
print(x.shape)                        # (14,)

##### One-Hot Encoding 3가지 만들기 #####
# 세 방법 모두 결과는 같음 (단어 14개 x 단어 종류 9개 = (14, 9))
# 차이점 : to_categorical -> 0번 칸까지 만들어서 직접 지워야 함
#          get_dummies / OneHotEncoder -> 실제로 있는 값만 칸으로 만듦 (0 이 없으니 처음부터 9칸)

########## OneHot Encoding 1. (tensorflow) ##########
from tensorflow.keras.utils import to_categorical
x1 = to_categorical(x)                # 가장 큰 번호 (9) 를 보고 0 ~ 9 = 10칸을 만듦
print('\n===== to_categorical One-Hot Encoding =====')
print(x1)
print(x1.shape)                       # (14, 10) -> 0번 인덱스 열까지 생김 (단어 인덱스는 1부터 시작)
x1 = x1[:, 1:]                        # 쓰지 않는 0번 열 제거
print(x1.shape)                       # (14, 9)


# ########## OneHot Encoding 2. (pandas) ##########
# x2 = pd.get_dummies(x, dtype=int)     # 1차원 데이터를 넣어야 함 / dtype=int -> True, False 대신 1, 0 으로 출력
# print('\n===== pandas One-Hot Encoding =====')
# print(x2)                             # 열 이름이 단어 번호 (1 ~ 9) 인 DataFrame
# print(x2.shape)                       # (14, 9)


# ########## OneHot Encoding 3. (sklearn) ##########
# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(sparse_output=False)   # sparse_output=False -> 바로 볼 수 있는 numpy 배열로 출력

# # 2차원 형태로 넣어야 함
# x3 = x.reshape(-1, 1)                 # (14,) -> (14, 1)
# x3 = ohe.fit_transform(x3)            # 결과는 2차원 (14, 9) -> 단어 1개당 한 줄씩

# print('\n===== sklearn One-Hot Encoding =====')
# print(x3)
# print(x3.shape)                       # (14, 9)
