# rag11_Chroma01_save.py 베이스

# [목적] rag11_Chroma01_save 에서 저장한 Chroma DB 를 불러와 similarity_search 로 검색한다
#   문서 불러오기 / 청킹 / 저장 부분은 주석 처리 → 임베딩 API 를 다시 호출하지 않는다

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip() # strip() 공백이나 줄바꿈을 무시 (오탈자로 인한 에러 방지용)
# You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'
base_url = 'https://monogpt.kr/api/monorouter/v1'

# #01. 데이터 불러온다.
# path = './_data/rag_data/'

# # UnicodeDecodeError: 'cp949' codec can't decode byte 0xec in position 0: illegal multibyte sequence
# loader1 = TextLoader(path + 'samsung_outlook.txt', encoding='utf-8')
# loader2 = TextLoader(path + 'nvidia_outlook.txt', encoding='utf-8')

# #02. 문서를 자른다. (청킹)
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=100,                              # 자를 때 중복되는 구간
#     separators=['\n\n', '\n', ' ', '']              # 통상적인 default
# )

# split_doc1 = loader1.load_and_split(text_splitter)  # 청크 300, 오버랩 100 (위에서 설정한) 기준으로 데이터 분 (samsung)
# split_doc2 = loader2.load_and_split(text_splitter)  # 청크 300, 오버랩 100 (위에서 설정한) 기준으로 데이터 분 (nvidia)

#03. 문서 길이 확인
# print(split_doc1)
# print(len(split_doc1), len(split_doc2))             # 9 9 (Document와 page_content -> 청크 크기)

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,                                 # 출력 벡터 차원을 5 로 지정 (text-embedding-3 계열 모델에서만 사용 가능)
)

DB_PATH = './_db/Chroma11/'

#04. 불러오기
# from_documents (저장) 대신 Chroma(...) 로 이미 저장된 폴더를 연다
#   persist_directory / collection_name 이 저장할 때와 같아야 같은 DB 를 찾는다
#   embedding_function : 질문 문장을 벡터로 바꿀 때 쓴다 (저장할 때와 같은 모델이어야 벡터를 비교할 수 있다)
db = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='croma11',
)

# 저장된 데이터 확인
print('==============================')
aaa = db.similarity_search('삼성전자 사업 전망에 대해 알려줘', k=2) # 질문과 벡터가 가장 가까운 청크 2개를 골라준다, default = 4
# print(db.get())
print(aaa)