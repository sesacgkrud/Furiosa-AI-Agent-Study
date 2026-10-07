# rag17_FAISS_1_save.py 베이스

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

'''
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
'''

#04. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,                                 # 출력 벡터 차원을 5 로 지정 (text-embedding-3 계열 모델에서만 사용 가능)
)
'''
#05. FAISS
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query('Hello World!'))) # 검색 기반 : 유클리드 거리
# faiss_index = faiss.IndexFlatL2(1536)
print('FAISS 인덱스 초기화 준비 완료')

# FAISS 벡터 저장소의 벡터 차원 수 (임베딩 차원 수)
print(faiss_index.d) # 1536

faiss_db = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore=InMemoryDocstore(), # 메모리 기반으로 작업하겠다는 것
    index_to_docstore_id={},
)

# 저장된 문서의 갯수 확인
print(faiss_db.index.ntotal) # 0

#################### 준비 완료 ####################

db = FAISS.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings,
)
'''

DB_PATH = './_db/Faiss17'
# db.save_local(
#     folder_path=DB_PATH,
#     index_name='faiss_index17'
# )

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

print('==============================')
# 문서 저장소 ID 확인
print(db.index_to_docstore_id)
# {0: '894417ce-b66b-4967-99b1-07dd4b0863a2', 1: '5a0c79d3-2b4c-47d2-ab22-16183b744679', 2: '22f3864d-8dd9-4978-8542-2170c1067520',
#  3: '26e17e74-f0d7-46cc-9280-725a44ce56ce', 4: '174d60a5-122b-4035-bb66-59e88a6aa74f', 5: 'a8dca308-ec1d-4b84-bd43-f322dc711d7a',
#  6: '77d34eca-8063-48be-acae-8bea10f5cc88', 7: '4d3b33db-d941-4f7d-a298-e4e9db4ff35a', 8: 'c12938c5-88f2-4bfc-ab3a-9bb04f317b81',
#  9: '24ba29fd-1538-4ea0-ab0b-3a80fd973aec', 10: '3e785374-9120-4c3c-b312-e9ff05756913', 11: '08a84470-acf6-4a8f-9785-0e22bc0c4f2f',
#  12: 'e5ac0f81-5135-41c5-89df-a8ac1e7ed052', 13: '9b7ed2d5-6e72-4a42-974a-992cba8aba5f', 14: '37be1d9d-e297-49c4-a039-fe3ea8485b56',
#  15: '398f408b-a415-408e-95c8-afcb2e051f57', 16: '7d938225-23dc-42ed-94fd-b8a55512d949', 17: '779b4dd9-d2ee-4359-8b66-691525514302'}
print('==============================')
# 저장된 결과 확인
print(db.docstore._dict)
# {'894417ce-b66b-4967-99b1-07dd4b0863a2':
#  Document(id='894417ce-b66b-4967-99b1-07dd4b0863a2',
#           metadata={'source': './_data/rag_data/samsung_outlook.txt'},
#           page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다.' \
#           '여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.'),
#           ...
#           ...
#  '779b4dd9-d2ee-4359-8b66-691525514302':
#  Document(id='779b4dd9-d2ee-4359-8b66-691525514302',
#           metadata={'source': './_data/rag_data/nvidia_outlook.txt'},
#           page_content='게임용 GPU, 전문 시각화, 자동차 분야는 데이터센터 외의 사업 기반을 제공한다. 다만 인공지능 데이터센터 사업의 비중이 커질수록 실적은 대형 고객의 설비 투자 변화에 더 민감해질 수 있다.' \
#           '중장기 전망을 평가할 때에는 인공지능 시장의 성장, 소프트웨어 생태계의 지속성, 고객의 투자 수익성, 경쟁사의 대체 기술, 제품 공급 능력과 전력 비용을 함께 검토해야 한다. 이 문서는 RAG 실습을 위한 교육용 자료이며 투자 권유가 아니다.')
# }
print('==============================')
# 유사도 검색
aaa = db.similarity_search('삼성전자 창업주에 대해 알려줘', k=2)
print(aaa)
# [Document(id='894417ce-b66b-4967-99b1-07dd4b0863a2', metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='삼성전자 사업 전망\n\n삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다. 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.'), Document(id='c12938c5-88f2-4bfc-ab3a-9bb04f317b81', metadata={'source': './_data/rag_data/samsung_outlook.txt'}, page_content='삼성전자의 중장기 전망은 인공지능 메모리의 경쟁력, 첨단 공정의 수율 개선, 파운드리 고객 확대, 스마트폰의 제품 차별화에 달려 있다. 위험 요인으로는 반도체 가격 하락, 세계 경기 둔화, 공급망 차질, 환율 변동, 수출 규제와 경쟁 심화를 들 수 있다. 전망을 분석할 때에는 성장 산업에 참여하고 있다는 점과 그 기회가 실제 매출 및 이익으로 이어지는지를 함께 평가해야 한다. 이 문서는 RAG 실습을 위한 교육용 자료이며 투자 권유가 아니다.')]