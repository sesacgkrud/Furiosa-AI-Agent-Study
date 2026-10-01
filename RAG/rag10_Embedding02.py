# rag10_Embedding01.py 베이스

# [목적] dimensions 옵션으로 임베딩 벡터의 차원 (숫자 개수) 을 직접 줄일 수 있는지 확인한다.
#   - text-embedding-3-small 은 기본 1536 차원 -> dimensions=5 를 주면 5 차원으로 나옴
#   - 왜 줄이나 : 문서가 수만 개면 벡터도 수만 개를 저장하고 비교해야 함
#                 -> 차원이 작을수록 저장 공간과 검색 속도가 유리함
#                 -> 대신 너무 작으면 문장 의미를 담을 칸이 부족해서 비슷한 문장을 잘 못 찾음 (5 는 확인용, 실제로는 수백 ~ 1536 사용)
#   - keras61_Embedding04 의 Embedding(output_dim=100) 에서 "단어 하나를 몇 칸짜리 벡터로 만들지" 를 정한 것과 같은 개념
#
# [rag10_Embedding01.py 와 달라진 점]
#   - model 을 text-embedding-3-small 로 고정
#   - OpenAIEmbeddings 에 dimensions=5 추가 -> 출력 벡터 길이가 5 로 바뀜
#   (아래 LCEL 설명과 ChatOpenAI, PromptTemplate import 는 베이스 파일에서 남은 것 -> 이 파일에서는 사용 안 함)

# LCEL = Langchain Expression Language
# Chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip() # strip() 공백이나 줄바꿈을 무시 (오탈자로 인한 에러 방지용)
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = '삼성전자의 창업주는 누구인가요?'

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    dimensions=5,                                  # 출력 벡터 차원을 5 로 지정 (text-embedding-3 계열 모델에서만 사용 가능)
)

vector = embeddings.embed_query(prompt)
print(vector)                                      # 숫자 5개짜리 리스트
print('==============================')
print('임베딩 벡터의 차원: ', len(vector)) # dimensions 조절 가능 -> 임베딩 벡터의 차원:  5
