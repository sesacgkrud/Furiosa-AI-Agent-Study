# [목적]
# LangGraph 병렬 실행 (Parallel Execution) : 노드 두 개를 동시에 실행하고, 결과를 한 노드에서 합친다
# LLM · Tool 없이 정해진 문장만 돌려주는 노드로 병렬 실행 → 병합 흐름만 먼저 확인한다
#
#                 ┌──▶ search   (검색 노드) ──┐
#        START ───┤                            ├──▶ merge (결과 통합) ──▶ END
#                 └──▶ analysis (분석 노드) ──┘
#
# - 병렬 실행 : START 에서 add_edge 를 두 번 → search, analysis 가 같은 단계에서 함께 실행된다
#               (실행 순서는 정해져 있지 않아서 '분석 에이전트 실행' 이 먼저 찍히기도 한다)
# - 병합      : search → merge, analysis → merge → 두 노드가 모두 끝난 뒤 merge 가 한 번 실행된다
# - 노드마다 State 의 서로 다른 키(search_result / analysis_result)에 결과를 써서 덮어쓰지 않는다
#   → merge 에서 두 결과를 꺼내 final_result 로 합친다
# - State 에 선언하지 않은 키는 노드가 return 해도 State 에 저장되지 않으므로 사용할 키를 모두 선언한다
# - 그래프 구조 출력은 lg07_ToolNode_graph02 와 같은 방식 (draw_mermaid_png → BytesIO → Image.open → imshow)

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

#1. State 정의
class State(TypedDict):
    question : str                      # 사용자 질문 (invoke 입력)
    search_result : str                 # search 노드 결과
    analysis_result : str               # analysis 노드 결과
    final_result : str                  # merge 노드가 합친 최종 결과

#2. 검색 노드
def search_agent(state : State):
    print('검색 에이전트 실행')
    return {'search_result' : '엔비디아는 AI GPU 시장의 주요 기업이다.'}


#3. 분석 노드
def analysis_agent(state : State):
    print('분석 에이전트 실행')
    return {'analysis_result' : 'AI 반도체 수요 증가가 성장 요인이다.'}

#4. 결과 통합 노드
def merge_agent(state : State):
    result = (
        state['search_result'] + '\n' +
        state['analysis_result']
    )
    return {'final_result' : result}

#5. 그래프 구성
builder = StateGraph(State)

# 노드 이름을 생략하고 함수만 넣으면 함수 이름이 노드 이름이 된다 ('analysis_agent', 'merge_agent')
# builder.add_node('search_agent', search_agent)
# builder.add_node(analysis_agent)
# builder.add_node(merge_agent)

# add_edge 에는 함수 이름이 아니라 여기서 정한 노드 이름('search', 'analysis', 'merge')을 쓴다

builder.add_node('search', search_agent)
builder.add_node('analysis', analysis_agent)
builder.add_node('merge', merge_agent)

########## 병렬 실행 ##########
# START 에서 두 노드로 각각 Edge → 동시에 실행

builder.add_edge(START, 'search')
builder.add_edge(START, 'analysis')

########## 병합 ##########
# 두 노드 모두 merge 로 Edge → search, analysis 가 모두 끝나야 merge 실행

builder.add_edge('search', 'merge')
builder.add_edge('analysis', 'merge')

builder.add_edge('merge', END)

# 컴파일
graph = builder.compile()

# 실행
response = graph.invoke({
    'question' : '엔비디아의 미래 전망은?'
})

print(response['final_result'])
# 검색 에이전트 실행 / 분석 에이전트 실행 (순서는 실행마다 다를 수 있음)
# 엔비디아는 AI GPU 시장의 주요 기업이다.
# AI 반도체 수요 증가가 성장 요인이다.

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