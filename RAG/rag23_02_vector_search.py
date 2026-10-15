# rag23_01_bm25_search.py 베이스
# [변경] BM25 검색은 주석 처리하고, 같은 문서 · 같은 질문을 벡터 검색(Chroma + OpenAI 임베딩)으로 찾는다
#   - BM25 는 글자가 같은 단어만 보지만, 벡터 검색은 문장을 임베딩 벡터로 바꿔 의미가 가까운 문서를 찾는다
#     → '삼성전자의' / '삼성전자는' 처럼 조사가 달라도 의미가 같으면 가깝게 나온다
#   - Chroma.from_documents : texts 8개를 임베딩해서 ./_db/Chroma12/ 의 'croma12' 컬렉션에 저장하고 vector_store 를 만든다
#   - as_retriever(search_kwargs={'k':3}) : 질문과 가장 가까운 문서 3개를 가져오는 검색기
#   - 여기서 저장한 DB 는 rag23_03 hybrid 검색에서 그대로 불러와 쓴다

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

# bm25_retriever = BM25Retriever.from_documents(texts)
# bm25_retriever.k = 3

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
    model='text-embedding-3-small',                                 # 문서 저장 · 질문 검색 모두 이 모델로 벡터를 만든다
    api_key=api_key,
    base_url=base_url,
)

DB_PATH = './_db/Chroma12/'                                         # rag12 에서 만든 DB 폴더

# 같은 폴더 · 같은 컬렉션 이름에 from_documents 하면 기존 문서는 그대로 두고 texts 8개가 추가된다
# (실행할 때마다 8개씩 다시 추가된다)
vector_store = Chroma.from_documents(
    embedding=embeddings,
    documents=texts,
    persist_directory=DB_PATH,
    collection_name='croma12'
)

vector_retriever = vector_store.as_retriever(
    search_kwargs={'k':3}                                           # 질문과 가까운 문서 3개
)

vector_docs = vector_retriever.invoke(query)

print('========== Vector DB 검색 결과 ==========')
for doc in vector_docs:
    print(doc.page_content)

    # ========== Vector DB 검색 결과 ========== <- 기존 저장되어 있는 DB에서까지 참고해서 결과 나옴 (persist_directory · collection_name)
    # 삼성전자는 반도체를 생산하는 기업이다.
    # 삼성전자는 AI 반도체 기술을 연구한다.
    # 삼성전자 사업 전망

    # 삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다.
    # 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다.
    # 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.

    # - 1등 '삼성전자는 반도체를 생산하는 기업이다.' 는 BM25 에서 0점이던 문장 → 조사가 달라도 의미로 찾는다
    # - 3번째는 texts 가 아니라 이전 rag12 실습 때 Chroma12 에 저장된 samsung 텍스트 청크
    #   (croma12 컬렉션 : 이전 실습 청크 208개 + texts 8개 = 216개)
    # - BM25 결과에 있던 '테슬라는 전기자동차를 제조한다.' 처럼 관계없는 문장은 나오지 않는다
