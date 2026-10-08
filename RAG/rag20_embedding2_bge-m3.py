# rag10_Embedding01.py 베이스

# [목적] OpenAI API 대신 HuggingFace 에 공개된 임베딩 모델 BAAI/bge-m3 을 내 PC(CPU) 에서 직접 돌려 문장을 벡터로 바꾼다
#   - HuggingFaceEmbeddings(model_name=...) : 처음 실행할 때 모델을 내려받아 ~/.cache/huggingface 에 저장하고 (캐시 약 4.3GB), 이후에는 저장된 모델을 쓴다
#   - API 키 · 요금 없이 임베딩할 수 있다 (대신 내 PC 에서 계산하므로 OpenAI 보다 느리다)
#   - BAAI/bge-m3 : BAAI(베이징 인공지능 연구원) 의 다국어 임베딩 모델, 한국어 문장도 바로 벡터로 바꾼다
#   - 사용법은 OpenAIEmbeddings 와 같다 -> embed_query(문장) 으로 벡터 1개
#   - 벡터 차원 1024 (OpenAI text-embedding-3-small 1536 / large 3072), 벡터 길이(norm) 가 1 로 정규화되어 나온다

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = '삼성전자의 창업주는 누구인가요?'

# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(
#     # model = 'text-embedding-3-small',
#     model = 'text-embedding-3-large',
#     api_key=api_key,
#     base_url=base_url,
# )

# pip install langchain_huggingface
# ImportError: Could not import sentence_transformers python package. Please install it with `pip install sentence-transformers`.
# 해결 방법 : pip install sentence-transformers
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-m3',                  # 다국어 임베딩 모델 (1024 차원)
    model_kwargs={
        'device' : 'cpu',                          # GPU 대신 CPU 로 계산
        # 'local_files_only' : True,               # 모델을 내려받은 뒤에는 True 로 두면 인터넷 연결 없이 저장된 모델만 쓴다
    }
)

vector = embeddings.embed_query(prompt)
print(vector)                                      # [-0.0663, -0.0078, -0.0158, ...] 숫자 1024개
print('==============================')
print('임베딩 벡터의 차원: ', len(vector))          # 임베딩 벡터의 차원:  1024
