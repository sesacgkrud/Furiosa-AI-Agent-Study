# rag15_Chroma_gradio.py 베이스

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

#03. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    # dimensions=5,                                 # 출력 벡터 차원을 5 로 지정 (text-embedding-3 계열 모델에서만 사용 가능)
)

DB_PATH = './_db/Chroma12/'

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='croma12',
)

print(f'벡터 저장소에 저장된 문서 수: {vector_store._collection.count()}') # 벡터 저장소에 저장된 문서 수: 208

# query = '삼성전자의 창업자는 누구인가요?'
# result = vector_store.similarity_search(query)
# print(f'검색 결과의 길이: {len(result)}') # 검색 결과의 길이: 4 (default)

#################### Retrievers ####################
###################### 검색기 #######################
retriever = vector_store.as_retriever(search_kwargs={'k':2})
print(retriever) # tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x000001DAFE45FF10> search_kwargs={'k': 2}
# aaa = retriever.invoke(query)
# print(f'검색된 관련 문서 수: {len(aaa)}')
# print(f'첫번째 관련 문서 내용 미리보기: {aaa[0].page_content[:50]}')

##################### 모델 연결 #####################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0,      # 0 -> 있는 그대로 | 1 -> 다양하고 창의적으로
                        # temperature : 다음 단어를 고를 때 얼마나 무작위로 고를지 정하는 값
                        #   0 -> 확률이 가장 높은 단어만 골라서, 같은 질문에는 거의 항상 같은 답이 나온다 (사실 확인 · RAG 답변에 적합)
                        #   1 -> 확률 분포 그대로 뽑아서, 확률이 낮은 단어도 가끔 골라 물을 때마다 표현이 달라진다 (아이디어 · 글쓰기에 적합)
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

# response = model.invoke('엔비디아는 어떤 기업인가요?')
# print('model의 답변 :', response.content)

# print('===========================================')
# query_with_context = f'''
#     {aaa[0].page_content}\n\n
#     위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
# '''

# response = model.invoke(query_with_context)
# print('model의 응답 :', response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template('''

    다음 컨텍스트를 바탕으로 질문에 답변해 주세요.
    컨텍스트 밖의 지식은 절대 쓰지 말고, 없으면 정확히 '주어진 정보로는 답변할 수 없습니다.'만 출력해주세요.

    컨텍스트 : {context}
    질문 : {input}
    답변 : 

''')

# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)  # prompt | model
rag_chain = create_retrieval_chain(retriever, docu_chain) # 검색 | docu_chain

'''
# 체인 실행
query = '삼성전자의 창업자는 누구인가요?'
response = rag_chain.invoke({'input' : query})

print(response)
print('====================== keys() =====================')
print(response.keys())                                           # dict_keys(['input', 'context', 'answer'])
print('====================== context =====================')
print(response['context'][0].page_content)                       # 삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다. 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다. 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.
print('====================== answer =====================')
print(response['answer'])                                        # 주어진 정보로는 답변할 수 없습니다.
'''

#################### Gradio 챗봇 ####################
import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({'input' : message})
    return response['answer']

# Gradio 인터페이스 생성
demo = gr.ChatInterface(fn=answer_invoke, title='AI Chat Bot')

# Gradio 실행
demo.launch()
# demo.launch(share=True) # 배포 URL 제공