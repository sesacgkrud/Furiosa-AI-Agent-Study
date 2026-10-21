# lg08_Memory02.py 베이스

# [목적]
# lg09_Parallel_Execution_absorption1 의 병렬 실행 → 병합 그래프에 실제 웹 검색(Tavily)과 LLM 분석을 넣는다
#
#                 ┌──▶ search   : TavilySearch 로 웹 검색          ──┐
#        START ───┤                                                    ├──▶ merge : 두 결과를 합쳐 assistant 메시지로 ──▶ END
#                 └──▶ analysis : LLM 이 기존 지식으로 질문 분석    ──┘
#
# - 질문은 messages 로 받는다 → 각 노드에서 state['messages'][-1].content 로 마지막(사용자) 메시지를 꺼낸다
# - search, analysis 는 State 의 search_result / analysis_result 키에 따로 결과를 쓰고,
#   merge 가 두 결과를 '=====' 로 구분해 합친 뒤 messages 에 ('assistant', result) 로 추가한다
# - Tool 을 LLM 에 bind_tools 로 맡기지 않고, search 노드에서 web_search.invoke 로 직접 검색한다
#   → 검색은 항상 실행되고, 검색과 LLM 분석이 동시에 진행된다

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    max_tokens=8000,                # gpt-5.6-terra 는 답변 전에 추론(reasoning) 토큰을 먼저 쓴다
                                    # 2000 이면 추론에 다 써서 분석 답변(response.content)이 빈 문자열이 된다 (finish_reason : 'length')
    api_key=api_key,
    base_url=base_url,
)

#################### 0. Tool 정의 ####################

from langchain_tavily import TavilySearch

web_search = TavilySearch(
    max_results=5,
)

tools = [web_search]
llm_with_tools = llm.bind_tools(tools)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

#################### 1. State 정의 ####################

class State(TypedDict):
    messages : Annotated[list, add_messages]
    search_result : str                 # search 노드 결과 (Tavily 검색 결과)
    analysis_result : str               # analysis 노드 결과 (LLM 분석 답변)

#################### 2. Node 정의 #####################

# 검색 에이전트
def search_agent(state : State):
    print('검색 에이전트 실행')
    question = state['messages'][-1].content    # 마지막 메시지 = 사용자 질문

    result = web_search.invoke({                # Tool 을 노드 안에서 직접 실행
        'query' : question
    })
    return {'search_result' : str(result)}      # 검색 결과 딕셔너리 → 문자열 (State 키가 str)

# 분석 에이전트
def analysis_agent(state : State):
    print('분석 에이전트 실행')
    question = state['messages'][-1].content

    response = llm.invoke(
        f'다음 질문을 기존 지식을 바탕으로 분석해줘.\n{question}'
    )
    return {'analysis_result' : response.content}

# 결과 통합 에이전트
def merge_agent(state : State):
    result = (
        state['search_result'] + '\n\n' +
        '==============================' + '\n' +
        state['analysis_result']
    )
    return {'messages' : [('assistant', result)]}  # add_messages 로 messages 에 AI 답변으로 추가

#################### 3. Graph 구성 ####################

builder = StateGraph(State)

builder.add_node('search', search_agent)
builder.add_node('analysis', analysis_agent)
builder.add_node('merge', merge_agent)

########## 병렬 실행 ##########

builder.add_edge(START, 'search')
builder.add_edge(START, 'analysis')

########## 병합 ##########

builder.add_edge('search', 'merge')
builder.add_edge('analysis', 'merge')

builder.add_edge('merge', END)

# 컴파일
graph = builder.compile()

#################### 4. 그래프 실행 ####################

query = '엔비디아의 미래 전망에 대해 알려줘'

response = graph.invoke(
    {'messages' : [('user', query)]},
)

print('Assistant :', response['messages'][-1].content)
# 검색 에이전트 실행 / 분석 에이전트 실행
# Assistant : {'query': '엔비디아의 미래 전망에 대해 알려줘', ..., 'results': [{'url': ..., 'title': ..., 'content': ...}, ...]}
#
# ==============================
# 엔비디아의 미래 전망은 전반적으로 AI 인프라 시장의 핵심 수혜자라는 강한 강점과, ...

#################### 그래프 구조 출력 ####################

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
