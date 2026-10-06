# rag10_Embedding02.py 베이스

# [목적] 텍스트 파일을 불러와 청크로 자르고, OpenAI 임베딩으로 벡터를 만들어 Chroma (벡터 DB) 에 저장한다
#   TextLoader → RecursiveCharacterTextSplitter → OpenAIEmbeddings → Chroma.from_documents
#   persist_directory 를 주면 DB 가 폴더(./_db/Chroma11/) 에 파일로 남는다 → rag11_Chroma02_load 에서 다시 임베딩하지 않고 불러온다

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

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,                                 # 출력 벡터 차원을 5 로 지정 (text-embedding-3 계열 모델에서만 사용 가능)
)

DB_PATH = './_db/Chroma11/'

#04. 저장
db = Chroma.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name='croma11',
)

print('Chroma 문서 저장 끝')