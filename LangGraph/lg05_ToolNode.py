# tool calling : 외부 기능을 가져와서 사용하기

# [목적]
# LLM 이 혼자 답할 수 없는 질문(오늘 주가 같은 실시간 정보)을 웹 검색 Tool 로 찾아서 답하게 한다
#
# 흐름 : Tool 정의 (TavilySearch) → bind_tools 로 LLM 에 Tool 연결 → State / Node 정의
#        → chatbot 노드 + tools 노드 (ToolNode) → 조건부 Edge (tools_condition) → compile → invoke
#
#        START ──▶ chatbot ──(tools_condition)──▶ tool_calls 있음 ──▶ tools ──┐
#                    ▲                     └──▶ tool_calls 없음 ──▶ END    │
#                    └─────────────────────────────────────────────────────┘
#
# - bind_tools : LLM 이 필요하다고 판단하면 답변 대신 'Tool 을 이렇게 불러 달라' 는 tool_calls 를 돌려준다
# - ToolNode   : tool_calls 대로 Tool 을 실제로 실행하고, 결과를 ToolMessage 로 messages 에 추가하는 노드
# - tools 실행 후 다시 chatbot 으로 돌아가 검색 결과를 보고 최종 답변을 만든다
# - TavilySearch 는 .env 의 TAVILY_API_KEY 를 사용한다

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    max_tokens=2000,                # 검색 결과를 정리한 답변이 길어서 lg03 보다 늘림
    api_key=api_key,
    base_url=base_url,
)

#################### 0. Tool 정의 ####################

# pip install langchain-tavily
from langchain_tavily import TavilySearch

# 웹 검색 도구 생성
web_search = TavilySearch(
    max_results=5,                  # 검색 결과 최대 5개
)

# result = web_search.invoke({
#     'query' : '삼성전자의 오늘 주가 알려줘',
# })

# print(result)
# {'query': '삼성전자의 오늘 주가 알려줘',
#  'follow_up_questions': None,
#  'answer': None,
#  'images': [],
#  'results': [{
#      'url': 'https://kr.investing.com/equities/samsung-electronics-co-ltd',
#      'title': 'Samsung Electronics Co Ltd 오늘의 주가 | 005930 실시간 티커 - Investing.com',
#      'content': '### 주요 주주\n\n| 보유자 | 보유 비중 % | 보유 주식 수 | 기준일 | 가치 (1,000 단위) ...',
#      'score': 0.8646883, 'raw_content': None, 'id': '499674-00'
#      }, {
#          'url': 'https://ir.gsifn.io/sample/ir_search.html',
#          'title': '주가검색 - 삼성전자',
#          'content': '##### 삼성전자\n\n(2026/10/06)\n\n272,000\n\n전일대비\n:   4,000\n\n등락률(%)\n:...',
#          'score': 0.78886026,
#          'raw_content': None,
#          'id': '31026c-01'
#          },
#          {
#          'url': 'https://stock.mk.co.kr/price/home/KR7005930003',
#          'title': '삼성전자(005930)- 매일경제 마켓',
#          'content': '## 주가 정보\n\n 코스피\n\n  5,768.01 1.76%\n 코스닥\n\n  808.89 1.78%\n ...',
#          'score': 0.7520312,
#          'raw_content': None,
#          'id': 'fa3984-02'
#          }
#   ],
#          'response_time': 2.26,
#          'request_id': '633d7f48-d1ab-416b-aeaa-7caf8e898844'}

# print(result['results'][-1]['content'])


tools = [web_search]                        # LLM 이 사용할 수 있는 Tool 리스트
llm_with_tools = llm.bind_tools(tools)      # LLM 에 Tool 목록을 알려준다

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

#################### 1. State 정의 ####################

class State(TypedDict):
    messages : Annotated[list, add_messages]

#################### 2. Node 정의 #####################

def chatbot(state : State):
    # 답변(AIMessage) 또는 tool_calls 가 담긴 AIMessage 를 돌려준다
    return {'messages' : [llm_with_tools.invoke(state['messages'])]}

#################### 3. Graph 구성, Node 추가 ##########
# 그래프 생성
graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)

tool_node = ToolNode(tools)                 # tool_calls 를 받아 Tool 을 실행하는 미리 만들어진 노드

# 아래 둘 다 같은 의미, 편한 방법으로 사용하면 됨
# graph_builder.add_node('tools', tool_node)
graph_builder.add_node('tools', ToolNode(tools))

#################### 4. Edge 추가 #####################
graph_builder.add_edge(START, 'chatbot')

# graph_builder.add_edge('chatbot', 'tools')    # 일반 Edge 는 항상 tools 로 가서 끝나지 않는다
graph_builder.add_conditional_edges(            # 조건부 Edge : 함수 결과에 따라 다음 노드를 고른다
    'chatbot',
    tools_condition # tool_calls 이 있는지?
    # tool_calls 가 있으면 'tools' 반환
    # tool_calls 가 없으면 END 반환
    # tools_condition 에 END 포함되어 있음
)

graph_builder.add_edge('tools', 'chatbot')      # Tool 결과를 들고 다시 chatbot 으로

#################### 5. Compile ######################

graph = graph_builder.compile()

#################### 6. Graph 실행 ####################

query = '삼성전자의 오늘 주가를 알려줘'
response = graph.invoke({
    'messages' : [('user', query)]
})

# messages : [human 질문, ai (tool_calls), tool (검색 결과), ai (최종 답변)] → 마지막이 최종 답변
print('AI :', response['messages'][-1].content)
