# lg07_ToolNode_graph01.py 베이스

# [목적]
# lg07_ToolNode_graph01 은 그래프 구조 PNG 를 파일(_save/image/)로 저장했다면,
# 이 파일은 파일로 저장하지 않고 matplotlib 창에 바로 그림으로 띄워서 확인한다
# - draw_mermaid_png() 결과는 PNG 파일 내용 그대로인 bytes 라서 plt.imshow 에 바로 넣을 수 없다
# - BytesIO(png_data)  : bytes 를 메모리 위의 파일처럼 다룰 수 있게 감싼다 (디스크에 저장하지 않음)
# - Image.open(...)    : PIL 로 PNG 를 열어 이미지 객체로 만든다 → plt.imshow 로 출력

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
# 그래프 이미지 출력
######################################################

# pip install matplotlib
import matplotlib.pyplot as plt
from PIL import Image                   # 이미지 열기 (pillow)
from io import BytesIO                  # bytes 를 파일처럼 읽게 해주는 메모리 버퍼

png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))     # bytes → 메모리 파일 → PIL 이미지

# 그림 출력
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')                         # x, y 축 눈금 숨기기
plt.show()
