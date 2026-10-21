# lg05_ToolNode.py 베이스

# [목적]
# lg05 의 ToolNode 그래프를 실행하지 않고, 그래프 구조를 PNG 이미지로 저장한다
# - graph.get_graph()       : compile 된 그래프의 노드 · Edge 구조
# - .draw_mermaid_png()     : 구조를 Mermaid 다이어그램 PNG (bytes) 로 만든다
# - open(..., 'wb')         : bytes 를 바이너리 쓰기 모드로 파일에 저장 → ./_save/image/lg07_ToolNode.png
# 저장된 그림 : __start__ → chatbot → (조건부, 점선) tools / __end__, tools → chatbot

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

# query = '삼성전자의 오늘 주가를 알려줘'
# response = graph.invoke({
#     'messages' : [('user', query)]
# })

# print('AI :', response['messages'][-1].content)

######################################################
# 그래프 이미지 저장
######################################################

path = './_save/image/'
png_data = graph.get_graph().draw_mermaid_png()     # 그래프 구조 → Mermaid PNG (bytes)

# 파일 입출력 시 사용
with open(path + 'lg07_ToolNode.png', 'wb') as file:    # 'wb' : 바이너리 쓰기, with 블록이 끝나면 자동으로 close
    file.write(png_data)

# 위와 같은 코드
# file = open(path + 'lg07_ToolNode1.png', 'wb')
# file.write(png_data)
# file.close()
