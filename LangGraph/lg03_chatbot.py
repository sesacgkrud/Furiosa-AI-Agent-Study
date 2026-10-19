# rag06_monogpt.py 베이스

# [목적]
# LangGraph 로 LLM 을 한 번 부르는 가장 단순한 챗봇 그래프를 만든다
#
# 흐름 : State 정의 → Node 정의 → Graph 구성 (add_node) → Edge 추가 (add_edge) → compile → invoke
#
#        START ──▶ chatbot ──▶ END
#
# - State : 그래프 안에서 노드끼리 주고받는 데이터 (TypedDict, lg01)
# - Node  : State 를 받아서 바꿀 값만 딕셔너리로 돌려주는 함수
# - Edge  : 노드 실행 순서 (START = 시작, END = 끝)

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    max_tokens=1000,            # 답변 최대 토큰 수
    api_key=api_key,
    base_url=base_url,
)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

#################### 1. State 정의 ###################

class State(TypedDict) :
    messages : Annotated[list, add_messages]    # 노드가 돌려준 메시지가 덮어쓰지 않고 이어 붙는다 (lg02)
    aaa : str                                   # messages 외에 다른 키도 둘 수 있다 (이번 실습에서는 사용 X)
    bbb : str
    ccc : str

# 위 코드와 같은 의미
# state = {
#     'messages' : [],
#     'aaa' : 'abc',
#     'bbb' : 'bca',
#     'ccc' : 'cab',
# }

#################### 2. Node 정의 ####################

def chatbot(state : State) :
    # 지금까지의 messages 를 LLM 에 넣고, 답변(AIMessage) 을 리스트로 돌려준다 → add_messages 로 추가된다
    return {'messages' : [llm.invoke(state['messages'])]}

#################### 3. Graph 구성 ###################

graph_01 = StateGraph(State)                # State 형식으로 그래프 생성
graph_01.add_node('chatbot', chatbot)       # ('노드 이름', 실행할 함수)

#################### 4. Edge 추가 ####################

graph_01.add_edge(START, 'chatbot')         # 시작 → chatbot
graph_01.add_edge('chatbot', END)           # chatbot → 끝

#################### 5. Graph Compile ###############

graph = graph_01.compile()                  # 실행할 수 있는 그래프로 만든다

#################### 6. Graph Execution #############

query = '삼성전자의 주가를 예측해줘'
response = graph.invoke({
    'messages' : [('user', query)]          # ('user', 질문) → HumanMessage 로 바뀌어 들어간다
})

print(response)                             # 최종 State 딕셔너리 {'messages' : [HumanMessage, AIMessage]}
# print(response['messages'][0])          # HumanMessage 출력
# print(response['messages'][1])          # AIMessage 출력
# print(response['messages'][1].content)  # AIMessage 내용 출력
# print(response['messages'][0].type)     # human
# print(response['messages'][1].type)     # ai
