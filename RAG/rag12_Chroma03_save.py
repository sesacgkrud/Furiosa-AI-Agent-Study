# rag11_Chroma01_save.py 베이스

# [목적] 폴더 안의 txt 파일 전부를 glob 으로 찾아 한 번에 불러오고, Chroma 에 저장한 뒤 검색기(retriever) 로 검색한다
#   rag11 과 달라진 점
#     - 파일 이름을 하나씩 적지 않고 glob('*.txt') 로 목록을 만든다 → 반복문으로 TextLoader
#     - loader.load() 로 Document 리스트를 모은 뒤 split_documents 로 한 번에 청킹 (chunk_overlap 100 → 10)
#     - similarity_search 외에 as_retriever(search_kwargs={'k':2}) → retriever.invoke(query)

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

# 폴더에서 텍스트 파일 목록 가져오기
from glob import glob

path = './_data/rag_data/'
txt_files = glob(os.path.join(path, '*.txt'))

print(txt_files) # ['./_data/rag_data\\2026_AI_for_All.txt', './_data/rag_data\\nvidia_outlook.txt', './_data/rag_data\\samsung_outlook.txt']

#01. 데이터 불러온다.
path = './_data/rag_data/'

data = []
for text_file in txt_files:
    loader = TextLoader(text_file, encoding='utf-8')
    # data.append(loader)
    data += loader.load()

print('==========')
# print(data[0])
print('==========')
print(len(data))    # 3
print(data[0].page_content)

char_count = [len(doc.page_content) for doc in data]
print(char_count)   # [8158, 2049, 1898] 각각 문자의 갯수

#02. 문서를 자른다. (청킹)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,                                 # 300개 언저리로 잘림, 반드시 300개로 잘리는 것 아님
    chunk_overlap=10,                               # 자를 때 중복되는 구간
    separators=['\n\n', '\n', ' ', '']              # 통상적인 default
)

texts = text_splitter.split_documents(data)
print('생성된 텍스트 청크 수:', len(texts))                                # 생성된 텍스트 청크 수: 52
print('각 청크의 길이 :', list(len(text.page_content) for text in texts)) # 각 청크의 길이 : [259, 282, 282, 128, 276, ...]
# list() 없이 (len(...) for ...) 만 print 하면 값이 아니라 <generator object <genexpr> at 0x...> 가 출력된다

print('첫번째 청크의 내용: ', texts[0].page_content)      # 내용(값)만 나옴
print('첫번째 청크의 길이: ', len(texts[0].page_content)) # 259, chunk_size 미만 또는 언저리만큼 나옴

print('두번째 청크의 내용: ', texts[1].page_content)      # 내용(값)만 나옴
print('두번째 청크의 길이: ', len(texts[1].page_content)) # 282, chunk_size 미만 또는 언저리만큼 나옴

#03. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,                                 # 출력 벡터 차원을 5 로 지정 (text-embedding-3 계열 모델에서만 사용 가능)
)

sample_text = '삼성전자의 창업자는 누구인가요?'
vector = embeddings.embed_query(sample_text)
print(vector)
print(len(vector)) # 1536

DB_PATH = './_db/Chroma12/'

#04. 저장
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name='croma12',
)

print(f'벡터 저장소에 저장된 문서 수: {vector_store._collection.count()}') # 벡터 저장소에 저장된 문서 수: 52

query = '삼성전자의 창업자는 누구인가요?'
result = vector_store.similarity_search(query)
print(f'검색 결과의 길이: {len(result)}') # 검색 결과의 길이: 4 (default)

#################### Retrievers ####################
###################### 검색기 #######################
retriever = vector_store.as_retriever(search_kwargs={'k':2})
print(retriever) # tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x000001DAFE45FF10> search_kwargs={'k': 2}
aaa = retriever.invoke(query)
print(f'검색된 관련 문서 수: {len(aaa)}')
print(f'첫번째 관련 문서 내용 미리보기: {aaa[0].page_content[:50]}')

# 검색된 관련 문서 수: 2
# 첫번째 관련 문서 내용 미리보기: 삼성전자 사업 전망
# 삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사

# retriever 는 질문과 가장 가까운 청크를 "원문 그대로" 돌려줄 뿐, 질문에 대한 답을 만들지 않는다
#   [:50] 으로 앞 50글자만 출력해서 중간에 잘려 보인다
#   rag_data 3개 파일에는 창업자 내용이 없다 → 그래도 유사도 검색은 항상 가장 가까운 k 개를 돌려준다 ("삼성전자" 가 들어간 청크)
#   답 문장을 만들려면 검색된 청크를 LLM(ChatOpenAI) 에 같이 넘기는 단계가 더 필요하다
