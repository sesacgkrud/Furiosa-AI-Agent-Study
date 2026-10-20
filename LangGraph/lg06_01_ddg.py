# lg05_ToolNode.py 베이스

# [목적]
# lg05 의 웹 검색 Tool 을 TavilySearch 대신 DuckDuckGoSearchRun 으로 바꿔서 같은 그래프로 실행한다
# - DuckDuckGo 검색은 API 키 없이 사용할 수 있다 (pip install ddgs)
# - Tool 만 바뀌고 State / Node / Edge (ToolNode, tools_condition) 는 lg05 와 같다

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    max_tokens=2000,
    api_key=api_key,
    base_url=base_url,
)

#################### 0. Tool 정의 ####################

# from langchain_tavily import TavilySearch

# web_search = TavilySearch(
#     max_results=5,
# )

# pip install ddgs
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

web_search = DuckDuckGoSearchRun(           # DuckDuckGo 웹 검색 Tool (API 키 불필요), 결과를 문자열 하나로 돌려준다
    max_results=5,                          # DuckDuckGoSearchRun 에는 적용되지 않는 인자 (내부 DuckDuckGoSearchAPIWrapper 기본값이 5)
)

# result = web_search.invoke({
#     'query' : '삼성전자의 오늘 주가 알려줘'
# })

# print(result)

# exit()                                    # Tool 결과만 확인하고 아래 그래프는 실행하지 않을 때

tools = [web_search]
llm_with_tools = llm.bind_tools(tools)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

#################### 1. State 정의 ####################

class State(TypedDict):
    messages : Annotated[list, add_messages]

#################### 2. Node 정의 #####################

def chatbot(state : State):
    return {'messages' : [llm_with_tools.invoke(state['messages'])]}

#################### 3. Graph 구성, Node 추가 ##########

graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)

tool_node = ToolNode(tools)

# graph_builder.add_node('tools', tool_node)
graph_builder.add_node('tools', ToolNode(tools))

#################### 4. Edge 추가 #####################
graph_builder.add_edge(START, 'chatbot')

# graph_builder.add_edge('chatbot', 'tools')
graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition
)

graph_builder.add_edge('tools', 'chatbot')

#################### 5. Compile ######################

graph = graph_builder.compile()

#################### 6. Graph 실행 ####################

query = '삼성전자의 오늘 주가를 알려줘'
response = graph.invoke({
    'messages' : [('user', query)]
})

print('AI :', response['messages'][-1].content)
