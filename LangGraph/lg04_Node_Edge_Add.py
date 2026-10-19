# lg03_chatbot.py 베이스

# [목적]
# lg03 의 chatbot 노드 앞뒤에 prepare / finish 노드를 추가해서, 노드 3개를 Edge 로 순서대로 연결한다
#
#        START ──▶ prepare ──▶ chatbot ──▶ finish ──▶ END
#
# - 노드마다 돌려준 messages 가 add_messages 로 차례대로 쌓여, 최종 messages 는 4개가 된다
#   [0] human (질문) → [1] ai (prepare) → [2] ai (chatbot, LLM 답변) → [3] ai (finish)

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

#################### 1. State 정의 ###################

class State(TypedDict) :
    messages : Annotated[list, add_messages]
    aaa : str
    bbb : str
    ccc : str

#################### 2. Node 정의 ####################

def prepare(state : State) :                # LLM 을 부르지 않고 고정된 AIMessage 만 추가
    return {'messages' :
            [AIMessage(content='질문을 접수했습니다.')]
            }

def chatbot(state : State) :                # 질문 + prepare 메시지까지 들어간 messages 로 LLM 호출
    return {'messages' : [llm.invoke(state['messages'])]}

def finish(state : State) :                 # LLM 답변 뒤에 완료 메시지 추가
    return {'messages' :
                [AIMessage(content='답변 처리가 완료되었습니다.')]
            }

#################### 3. Graph 구성 ###################

graph_01 = StateGraph(State)
graph_01.add_node('prepare', prepare)
graph_01.add_node('chatbot', chatbot)
graph_01.add_node('finish', finish)

#################### 4. Edge 추가 ####################

graph_01.add_edge(START, 'prepare')         # 시작 → prepare
graph_01.add_edge('prepare', 'chatbot')     # prepare → chatbot
graph_01.add_edge('chatbot', 'finish')      # chatbot → finish
graph_01.add_edge('finish', END)            # finish → 끝

#################### 5. Graph Compile ###############

graph = graph_01.compile()

#################### 6. Graph Execution #############

query = '삼성전자의 주가를 예측해줘'
response = graph.invoke({
    'messages' : [('user', query)]
})

#################### 7. 결과 출력 ####################
for i, msg in enumerate(response['messages']):  # 쌓인 메시지를 순서 번호와 함께 출력
    print(f'[{i}] {msg.type} : {msg.content}')  # msg.type : human / ai

    # [0] human : 삼성전자의 주가를 예측해줘
    # [1] ai : 질문을 접수했습니다.
    # [2] ai : 삼성전자 주가를 정확히 예측하는 것은 불가능합니다. 특히 저는 실시간 주가·공시·환율·반도체 가격 데이터를 직접 확인할 수 없으므로 특정 목표주가를 단정해 드리기보다는, 주가를 좌우할 핵심 변수와 시나리오로 보는 것이 적절합니다.

    # ### 삼성전자 주가의 핵심 변수
    # 1. **메모리 반도체 업황**
    # - ... (생략)
    # 결론적으로 삼성전자는 **AI 메모리와 반도체 업황 회복의 수혜 가능성**이 크지만, 실제 주가의 핵심은 **HBM 경쟁력과 파운드리 개선이 기대를 충족하느냐**에 달려 있습니다. 단기 가격 예측보다 분할매수·손절 기준·투자 기간을 정해 두는 접근이 더 현실적입니다.

    # 원하시면 **단기 3개월 / 중기 1년 / 장기 3년** 중 원하는 기간을 기준으로, 더 구체적인 시나리오형 전망을 작성해 드릴 수 있습니다.
    # [3] ai : 답변 처리가 완료되었습니다.
