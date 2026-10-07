# rag11_Chroma01_save.py 베이스

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

# pip install faiss-cpu
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip() # strip() 공백이나 줄바꿈을 무시 (오탈자로 인한 에러 방지용)
# You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'
base_url = 'https://monogpt.kr/api/monorouter/v1'

#01. 데이터 불러온다.
path = './_data/rag_data/'

# UnicodeDecodeError: 'cp949' codec can't decode byte 0xec in position 0: illegal multibyte sequence
loader1 = TextLoader(path + 'samsung_outlook.txt', encoding='utf-8')
loader2 = TextLoader(path + 'nvidia_outlook.txt', encoding='utf-8')

#02. 문서를 자른다. (청킹)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,                              # 자를 때 중복되는 구간
    separators=['\n\n', '\n', ' ', '']              # 통상적인 default
)

split_doc1 = loader1.load_and_split(text_splitter)  # 청크 300, 오버랩 100 (위에서 설정한) 기준으로 데이터 분할 (samsung)
split_doc2 = loader2.load_and_split(text_splitter)  # 청크 300, 오버랩 100 (위에서 설정한) 기준으로 데이터 분할 (nvidia)

#03. 문서 길이 확인
# print(split_doc1)
print(len(split_doc1), len(split_doc2))             # 9 9 (Document와 page_content -> 청크 크기)

#04. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,                                 # 출력 벡터 차원을 5 로 지정 (text-embedding-3 계열 모델에서만 사용 가능)
)

#05. FAISS
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query('Hello World!'))) # 검색 기반 : 유클리드 거리
# faiss_index = faiss.IndexFlatL2(1536)
# print('FAISS 인덱스 초기화 준비 완료')

# FAISS 벡터 저장소의 벡터 차원 수 (임베딩 차원 수)
# print(faiss_index.d) # 1536

# faiss_db = FAISS(
#     embedding_function=embeddings,
#     index=faiss_index,
#     docstore=InMemoryDocstore(), # 메모리 기반으로 작업하겠다는 것
#     index_to_docstore_id={},
# )

# 저장된 문서의 갯수 확인
# print(faiss_db.index.ntotal) # 0

#################### 준비 완료 ####################

db = FAISS.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings,
)

DB_PATH = './_db/Faiss17'
db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index17'
)