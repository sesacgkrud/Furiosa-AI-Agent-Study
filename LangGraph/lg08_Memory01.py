# lg07_ToolNode_graph02.py 베이스

# [목적]
# 메모리(checkpointer) 없이 graph.invoke 를 두 번 따로 실행하면 이전 대화를 기억하지 못하는 것을 확인한다
# - 첫번째 invoke : '내 이름은 Mr.Song 이야'  → 반갑습니다, Mr. Song!
# - 두번째 invoke : '내 이름이 뭐라고 했지?'  → 아직 이름을 말씀해 주신 적이 없어요
# - invoke 할 때마다 State 가 새로 시작되어 messages 에 이번 질문 하나만 들어가기 때문
#   (add_messages 는 한 번의 실행 안에서만 메시지를 쌓아 준다) → lg08_Memory02 에서 InMemorySaver 로 해결

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

# 첫번째 질문
response = graph.invoke({
    'messages' : [('user', '내 이름은 Mr.Song 이야')]
})

print(response['messages'][-1].content)

# 두번째 질문
print('========== 두번째 질문 ==========')
response = graph.invoke({
    'messages' : [('user', '내 이름이 뭐라고 했지?')]
})

print(response['messages'][-1].content)