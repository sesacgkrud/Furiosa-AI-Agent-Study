# lg08_Memory01.py 베이스

# [목적]
# lg08_Memory01 에서 기억하지 못했던 이전 대화를 InMemorySaver 메모리로 기억하게 한다
# - 메모리 추가는 세 단계 : ① InMemorySaver import → ② compile(checkpointer=memory) → ③ invoke 에 config(thread_id) 전달
# - checkpointer : 그래프 실행이 끝날 때마다 State(messages) 를 thread_id 별로 저장해 둔다
# - 같은 thread_id 로 다시 invoke 하면 저장된 messages 를 불러와 새 질문을 이어 붙인다 (add_messages)
#   → 두번째 질문 '내 이름이 뭐라고 했지?' 에 'Mr.Song이라고 하셨어요.' 라고 답한다
# - InMemorySaver 는 메모리(RAM) 에만 저장하므로 프로그램을 종료하면 대화 기록이 사라진다

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

# 메모리 추가 첫번째 : 대화 State 를 메모리(RAM) 에 저장하는 checkpointer
from langgraph.checkpoint.memory import InMemorySaver

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

# 메모리 추가 두번째 : compile 할 때 checkpointer 로 연결
memory = InMemorySaver()

graph = graph_builder.compile(checkpointer=memory)

#################### 6. Graph 실행 ####################

# 메모리 추가 세번째 : 대화방 id (thread_id) 지정
# 같은 thread_id 끼리 대화가 이어지고, 'user02' 처럼 다른 id 로 invoke 하면 처음부터 새 대화가 된다
config = {'configurable' : {'thread_id' : 'user01'}}

# 첫번째 질문
response = graph.invoke(
    {'messages' : [('user', '내 이름은 Mr.Song 이야')]},
    config=config                       # 두 번 모두 같은 config 를 넘겨야 이전 대화를 불러온다
)

print(response['messages'][-1].content)

# 두번째 질문
print('========== 두번째 질문 ==========')
response = graph.invoke({
    'messages' : [('user', '내 이름이 뭐라고 했지?')]},
    config=config)

print(response['messages'][-1].content)