# rag08_LCEL02.py 베이스

# [목적] 문장을 OpenAI 임베딩 모델로 "숫자 벡터" 로 바꿔 보고, 모델에 따라 벡터 차원이 몇인지 확인한다.
#   - RAG (검색 증강 생성) 의 첫 단계 : 문서와 질문을 벡터로 바꿔 두면 "뜻이 비슷한 문장 = 벡터가 가까운 문장" 으로 찾을 수 있음
#     -> 질문과 가장 비슷한 문서를 찾아서 LLM 에게 같이 넘겨주는 게 RAG 의 기본 흐름
#   - keras61_Embedding04 의 Embedding 층과 같은 개념 (단어 / 문장 -> 벡터)
#     차이 : keras 의 Embedding 은 우리 데이터 15문장으로 처음부터 학습 (output_dim=100 을 직접 정함)
#            OpenAI 임베딩은 엄청난 양의 글로 이미 학습된 모델을 API 로 빌려 씀 (문장 하나 -> 1536 / 3072 차원)
#
# [이전 파일 (rag08, rag09) 과 달라진 점]
#   - 이전 : ChatOpenAI (대화 모델) -> 질문을 넣으면 "답변 문장" 이 나옴
#   - 이번 : OpenAIEmbeddings (임베딩 모델) -> 문장을 넣으면 "숫자 리스트 (벡터)" 가 나옴 -> 답변을 만들지 않음
#   - 그래서 PromptTemplate, chain (prompt | model | output_parser) 을 쓰지 않음
#     (아래 LCEL 설명과 ChatOpenAI, PromptTemplate import 는 베이스 파일에서 남은 것 -> 이 파일에서는 사용 안 함)

# LCEL = Langchain Expression Language
# Chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip() # strip() 공백이나 줄바꿈을 무시 (오탈자로 인한 에러 방지용)
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = '삼성전자의 창업주는 누구인가요?'          # PromptTemplate 이 아니라 그냥 문자열 -> 이 문장 자체를 벡터로 바꿀 대상

from langchain_openai import OpenAIEmbeddings      # 문장 -> 벡터로 바꿔 주는 OpenAI 임베딩 모델 (ChatOpenAI 대신 사용)
embeddings = OpenAIEmbeddings(
    # model = 'text-embedding-3-small',            # 작은 모델 -> 1536 차원 (빠르고 저렴)
    model = 'text-embedding-3-large',              # 큰 모델   -> 3072 차원 (문장 의미를 더 세밀하게 표현, 더 비쌈)
    api_key=api_key,
    base_url=base_url,                             # ChatOpenAI 와 마찬가지로 사제 키를 쓸 때는 key 와 URL 을 함께 넣어야 함
)

vector = embeddings.embed_query(prompt)            # 문장 1개 -> 벡터 1개 (float 숫자들의 리스트)
                                                   # (참고) 문서 여러 개를 한 번에 바꿀 때는 embed_documents([문장1, 문장2, ...]) 사용
print(vector)                                      # [0.0123, -0.0456, ...] 처럼 숫자가 3072개 나열됨 -> 숫자 하나하나에 사람이 읽을 수 있는 의미는 없음
print('==============================')
# 아래 두 줄은 같은 코드 -> model 을 small / large 로 바꿔 가며 실행한 결과를 각각 기록해 둔 것 (지금은 large 라 둘 다 3072 출력)
print('임베딩 벡터의 차원: ', len(vector)) # 벡터 갯수 출력 -> 임베딩 벡터의 차원:  1536 (model -> small)
print('임베딩 벡터의 차원: ', len(vector)) # 벡터 갯수 출력 -> 임베딩 벡터의 차원:  3072 (model -> large)
