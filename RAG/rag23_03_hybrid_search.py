# rag23_02_vector_search.py 베이스
# [변경] BM25 검색기 + 벡터 검색기를 EnsembleRetriever 로 합쳐 hybrid 검색을 한다
#   - BM25 (키워드가 글자 그대로 같은 문서) + 벡터 (의미가 가까운 문서) → 두 방식의 장점을 같이 쓴다
#   - 벡터 DB 는 rag23_02 에서 저장한 Chroma12 를 Chroma(...) 로 불러온다 (from_documents 로 다시 저장하지 않는다)
#   - EnsembleRetriever(retrievers=[...], weights=[...])
#       retrievers : 합칠 검색기 리스트 (각자 k=3 개씩 찾는다)
#       weights    : 검색기마다 결과를 얼마나 중요하게 볼지 (0.5, 0.5 → 같은 비중)
#     → 두 검색기의 순위에 weights 를 곱해 점수를 합친 뒤 높은 순으로 정렬, 겹치는 문서는 한 번만 남긴다
#       (3개 + 3개 중 '삼성전자는 AI 반도체 기술을 연구한다.' 가 겹쳐서 결과는 5개)

from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = '삼성전자의 반도체 사업'

#################### BM25 검색 ####################

from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(texts)   # hybrid 에 넣을 BM25 검색기
bm25_retriever.k = 3

# bm25_docs = bm25_retriever.invoke(query)
# print('========== BM25 검색 결과 ==========')
# for doc in bm25_docs:
#     print(doc.page_content)

############################################################
############################################################
######################### 벡터 검색 #########################

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

#01. 임베딩 모델
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',                                 # 저장할 때와 같은 임베딩 모델을 써야 검색이 맞는다
    api_key=api_key,
    base_url=base_url,
)

DB_PATH = './_db/Chroma12/'

# rag23_02 에서 저장한 DB 불러오기 (Chroma() 는 문서를 넣지 않고 폴더 · 컬렉션만 연결)
vector_store = Chroma(
    embedding_function=embeddings,
    # documents=texts,
    persist_directory=DB_PATH,
    collection_name='croma12'
)

vector_retriever = vector_store.as_retriever(
    search_kwargs={'k':3}
)

# vector_docs = vector_retriever.invoke(query)

# print('========== Vector DB 검색 결과 ==========')
# for doc in vector_docs:
#     print(doc.page_content)

############################################################
############################################################
######################## hybrid 검색 ########################
from langchain_classic.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever,
    ],
    weights=[0.5,0.5] # 검색 중요도
)

hybrid_docs = hybrid_retriever.invoke(query)
print('========== hybrid DB 검색 결과 ==========')
for doc in hybrid_docs:
    print(doc.page_content)
    # ========== hybrid DB 검색 결과 ==========
    # 삼성전자는 AI 반도체 기술을 연구한다.
    # 삼성전자는 반도체를 생산하는 기업이다.
    # 반도체는 컴퓨터의 핵심 부품이다.
    # 테슬라는 전기자동차를 제조한다.
    # 삼성전자 사업 전망

    # 삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다.
    # 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다.
    # 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.

    # - 1등 'AI 반도체 기술' 문장 : BM25 1등 + 벡터 2등 → 두 검색기에 모두 있어서 점수가 가장 높다
    # - 나머지는 BM25 결과(반도체는 ... / 테슬라는 ...)와 벡터 결과(삼성전자는 반도체를 ... / 삼성전자 사업 전망)가 순위대로 섞인다
    # - BM25 의 0점 문장(테슬라)도 함께 들어온다 → 키워드 검색이 약한 한국어 문장에서는 weights 를 조절해 볼 수 있다
