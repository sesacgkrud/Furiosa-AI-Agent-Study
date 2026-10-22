# lg09_Parallel_Execution_absorption2_concat.py 베이스

# [목적]
# absorption2_concat 은 merge 노드에서 두 결과를 '=====' 로 단순히 이어 붙였다 (concat)
# → 이번에는 merge 노드에서도 LLM 을 한 번 더 불러, 검색 결과 + 분석 결과를 종합한 최종 답변을 만든다
#
#                 ┌──▶ search   : TavilySearch 로 웹 검색          ──┐
#        START ───┤                                                    ├──▶ merge : LLM 이 두 결과를 종합해 최종 답변 ──▶ END
#                 └──▶ analysis : LLM 이 기존 지식으로 질문 분석     ──┘
#
# - 그래프 구조(병렬 실행 → 병합)와 search / analysis 노드는 absorption2_concat 과 같고, merge 노드만 바뀐다
# - merge 프롬프트에 [웹 검색 결과] / [AI 분석 결과] 를 나눠 넣고, 검색 결과를 우선 참고하라고 지시한다
# - llm.invoke 의 결과(AIMessage)를 그대로 messages 에 넣는다 → add_messages 가 AI 답변으로 추가

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    max_tokens=8000,
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
    search_result : str
    analysis_result : str

#################### 2. Node 정의 #####################

# 검색 에이전트
def search_agent(state : State):
    print('검색 에이전트 실행')
    question = state['messages'][-1].content

    result = web_search.invoke({
        'query' : question
    })
    return {'search_result' : str(result)}

# 분석 에이전트
def analysis_agent(state : State):
    print('분석 에이전트 실행')
    question = state['messages'][-1].content

    response = llm.invoke(
        f'다음 질문을 기존 지식을 바탕으로 분석해줘.\n{question}'
    )
    return {'analysis_result' : response.content}

# 결과 통합 에이전트 (단순 concat 보다 강력한 merge)
def merge_agent(state : State):
    print('결과 통합 에이전트 실행')

    prompt = f'''
    다음 두 결과를 종합하여 최종 답변을 작성해줘.

    [웹 검색 결과]
    {state["search_result"]}

    [AI 분석 결과]
    {state["analysis_result"]}

    검색 결과를 우선적으로 참고하고,
    최신 정보와 기존 분석을 구분해서 설명해줘.

    '''

    response = llm.invoke(prompt)       # 두 결과를 LLM 이 다시 읽고 하나의 답변으로 정리

    return {'messages' : [response]}    # AIMessage 를 그대로 추가 (concat 때의 ('assistant', result) 와 같은 역할)

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
# 분석 에이전트 실행 / 검색 에이전트 실행 / 결과 통합 에이전트 실행     ← search · analysis 가 끝난 뒤 merge 실행
# Assistant : ## 종합 결론
# 엔비디아의 미래는 여전히 AI 인프라 확장의 최대 수혜 기업 중 하나라는 점에서 긍정적입니다. ...
# ## 1. 검색 결과 기준의 최신 흐름         ← 웹 검색 결과 (기사 · 출처 URL) 를 먼저 정리
# ## 2. 장기적으로 긍정적인 이유
# ## 3. 기존 분석 관점에서의 주요 위험     ← AI 분석 결과 (기존 지식) 를 구분해서 정리
# → concat 처럼 검색 결과 딕셔너리가 그대로 보이지 않고, 하나의 정리된 답변으로 출력된다

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
