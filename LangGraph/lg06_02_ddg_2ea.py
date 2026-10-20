# lg06_01_ddg.py 베이스

# [목적]
# TavilySearch 와 DuckDuckGoSearchRun 두 개의 웹 검색 Tool 을 함께 LLM 에 연결한다
# - tools = [web_search1, web_search2] → bind_tools 로 두 Tool 을 모두 알려준다
# - LLM 이 질문을 보고 어떤 Tool 을 부를지 (하나 또는 둘 다) 스스로 고른다
# - ToolNode(tools) 도 두 Tool 을 모두 가지고 있어서, tool_calls 의 Tool 이름(tavily_search / duckduckgo_search)으로 찾아 실행한다

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

from langchain_tavily import TavilySearch

web_search1 = TavilySearch(                 # Tool 1 : Tavily 웹 검색
    max_results=5,
)

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

web_search2 = DuckDuckGoSearchRun(          # Tool 2 : DuckDuckGo 웹 검색
    max_results=5,
)

tools = [web_search1, web_search2]          # Tool 2개를 한 리스트로
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
