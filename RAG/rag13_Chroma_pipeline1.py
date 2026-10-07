# rag12_Chroma04_load.py 베이스

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
query = '엔비디아는 어떤 기업인가요?'
result = vector_store.similarity_search(query)
print(f'검색 결과의 길이: {len(result)}') # 검색 결과의 길이: 4 (default)

#################### Retrievers ####################
###################### 검색기 #######################
retriever = vector_store.as_retriever(search_kwargs={'k':2})
print(retriever) # tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x000001DAFE45FF10> search_kwargs={'k': 2}
aaa = retriever.invoke(query)
print(f'검색된 관련 문서 수: {len(aaa)}')
print(f'첫번째 관련 문서 내용 미리보기: {aaa[0].page_content[:50]}')

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

# response = model.invoke('삼성전자의 창업자는 누구인가요?')
# model의 답변 : 삼성전자의 창립자는 여러 사람의 협업으로 형성된 대기업 그룹 삼성의 초기 창업 주체를 말합니다. 삼성그룹의 기원은 1938년 이병철 회장(1910-1987)이 처음 설립한 삼성상회에서 시작되었고, 이병철 회장이 삼성의 주도적 리더로 성장시키며 삼성그룹으로 확장했습니다. 따라서 삼성전자의 "창업자"를 하나로 특정하기보다는, 삼성전자는 이병철 회장과 삼성그룹의 초기 창립과 성장 과정에서 탄생한 기업으로 보는 것이 일반적입니다.

# 참고로:
# - 1938년: 이병철이 대구에서 삼성상회를 설립
# - 이후 삼성전자는 반도체, 디지털 가전 등으로 사업을 확대하며 삼성그룹의 핵심 계열사 중 하나가 됨

# 추가로 더 알고 싶은 부분이 있으면 말씀해 주세요.

response = model.invoke('엔비디아는 어떤 기업인가요?')

# model의 답변 : 엔비디아(NVIDIA, Incorporated)는 미국의 다국적 기술 기업으로, 주로 그래픽 처리 유닛(GPU) 개발로 유명합니다. 다음은 주요 특징들입니다:

# - 핵심 사업 분야
#   - GPU 생산: 게임용 그래픽 카드(GeForce 시리즈)로 잘 알려져 있으며, 데이터센터용 고성능 GPU도 제공합니다.
#   - AI와 머신러닝: 딥러닝 모델의 학습과 추론에 최적화된 GPU와 소프트웨어 스택(NVIDIA CUDA, cuDNN 등)을 제공.
#   - 자율주행, 로봇, 시뮬레이션: 자동차용 컴퓨팅 플랫폼(NVIDIA Drive), 엔터프라이즈용 GPU 인스턴스, 시뮬레이션 소프트웨어.
#   - 인프라 소프트웨어: 데이터센터용 GPU 가속화 솔루션, 클라우드 GPU 서비스.

# - 주요 제품 및 기술
#   - GPU 아키텍처: 현재는 Ada Lovelace, Hopper 등 최신 아키텍처를 바탕으로 GPU를 출시.
#   - 소프트웨어 생태계: CUDA(병렬 컴퓨팅 플랫폼), cuDNN(딥러닝 라이브러리), RTX 기술(레이 트레이싱), NVIDIA AI Enterprise 등.
#   - 하드웨어 플랫폼: GeForce(게이밍), Quadro/RTX 데이터센터용, HGX/Leap 서버 모듈 등.

# - 기업적 맥락
#   - 설립: 1993년 설립, 본사는 미국 캘리포니아주 산타클라라에 위치.
#   - 시장 위치: GPU 산업의 선두주자 중 하나. 특히 AI, 데이터센터, 클라우드 컴퓨팅 시장에서 중요한 파트너 역할.
#   - 인수: ARM 인수 시도 등 전략적 움직임으로도 주목받았으나 규제 이슈 등으로 최종 대규모 인수는 무산된 바 있음.

# - 산업 영향
#   - 게임 그래픽과 VR/AR 경험 향상.
#   - AI 개발자 생태계 촉진: 대규모 모델 학습과 실시간 추론 가속화.
#   - 데이터센터의 비용·에너지 효율 개선에 기여.

# 간단히 말해, 엔비디아는 GPU와 AI 컴퓨팅을 중심으로 그래픽, 자율주행, 데이터센터, 클라우드 인프라 전반에 걸친 고성능 컴퓨팅 솔루션을 제공하는 선도 기업입니다. 필요하시면 특정 분야(예: 게임용 GPU 비교, 데이터센터 GPU 종류, CUDA 생태계)로 더 자세히 설명해 드리겠습니다.

print('model의 답변 :', response.content)


print('===========================================')
query_with_context = f'''
    {aaa[0].page_content}\n\n
    위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
'''

response = model.invoke(query_with_context)
print('model의 응답 :', response.content)
# model의 응답 : 삼성전자의 창업자는 이병철(李秉喆)입니다. 그는 1938년에 삼성을 창립했고, 이후 삼성그룹의 다각화된 사업군을 이끌며 회사의 성장 기반을 마련했습니다.
# model의 응답 : 엔비디아(NVIDIA)는 그래픽처리장치(GPU)와 이를 활용하는 소프트웨어, 시스템 솔루션을 제공하는 기술기업입니다. 과거에는 게임용 그래픽카드로 널리 알려졌으나, 현재는 대규모 인공지능(AI) 데이터센터용 가속기와 관련 소프트웨어 생태계를 중심으로 사업을 확장하고 있습니다. AI 모델의 개발과 운영에 필요한 고성능 연산 자원 수요가 증가함에 따라 GPU 기반의 가속기, AI 프레임워크 및 인프라 소프트웨어를 포함한 생태계를 제공하며, 데이터센터용 HPC(고성능 컴퓨팅) 솔루션, 자율주행, 그래픽 렌더링 등 다양한 분야로 사업 영역을 확장하고 있습니다.