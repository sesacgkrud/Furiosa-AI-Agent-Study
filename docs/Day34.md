# Day34 - LangGraph 그래프 이미지 출력 · InMemorySaver 대화 메모리 · 병렬 실행 (Parallel Execution)

**학습 기간:** 2026-10-21

> Day33에서 파일로 저장했던 그래프 구조 PNG를 `BytesIO` + `Image.open`으로 열어 **matplotlib 창에 바로 출력**했다. 이어서 Tool calling 챗봇에 `'내 이름은 Mr.Song 이야'` → `'내 이름이 뭐라고 했지?'`를 차례로 물어 **메모리 없이는 이전 대화를 기억하지 못하는 것**을 확인하고, `InMemorySaver`를 `compile(checkpointer=memory)`로 연결한 뒤 `config`의 `thread_id`로 invoke 해 **이름을 기억하게** 했다. 마지막으로 START에서 두 노드로 Edge를 나눠 **search · analysis 노드를 동시에 실행(병렬 실행)** 하고 merge 노드에서 결과를 합치는 그래프를 만들었다. 처음에는 정해진 문장으로 흐름만 확인하고, 다음에는 search 노드에 Tavily 웹 검색, analysis 노드에 LLM 분석을 넣었다. Day33의 `lg07_ToolNode_graph.py`는 `lg07_ToolNode_graph01.py`로 이름을 바꿨다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 그래프 이미지 출력 | `Image.open(BytesIO(png_data))` → `plt.imshow(img)`로 파일 저장 없이 바로 띄운다 |
| BytesIO | PNG bytes를 메모리 위의 파일처럼 감싼다 (디스크에 저장하지 않음) |
| 메모리 없음 | invoke마다 State가 새로 시작 → 두번째 질문에서 이름을 모른다 |
| InMemorySaver | `compile(checkpointer=InMemorySaver())` → 실행이 끝날 때마다 State를 저장 |
| thread_id | `config = {'configurable' : {'thread_id' : 'user01'}}` → 같은 id끼리 대화가 이어진다 |
| 병렬 실행 | `add_edge(START, 'search')` + `add_edge(START, 'analysis')` → 두 노드가 동시에 실행 |
| 병합 | `add_edge('search', 'merge')` + `add_edge('analysis', 'merge')` → 둘 다 끝나야 merge 실행 |
| State 키 나누기 | 노드마다 `search_result` / `analysis_result` 다른 키에 결과를 쓴다 |
| 웹 검색 + LLM 병렬 | search는 `web_search.invoke`, analysis는 `llm.invoke` → merge에서 합쳐 assistant 메시지로 |
| lg07 (이름 변경) | `lg07_ToolNode_graph.py` → `lg07_ToolNode_graph01.py` |

---

## 📖 핵심 학습 내용

### 1. 그래프 이미지를 창에 바로 출력 (lg07_ToolNode_graph02)

lg07_ToolNode_graph01(Day33)을 베이스로, 그래프 구조 PNG를 파일로 저장하지 않고 matplotlib으로 띄웠다.

```python
# pip install matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()

img = Image.open(BytesIO(png_data))

# 그림 출력
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

- `draw_mermaid_png()`의 결과는 PNG 파일 내용 그대로인 **bytes** → `plt.imshow`에 바로 넣을 수 없다
- `BytesIO(png_data)` : bytes를 **메모리 위의 파일**처럼 다룰 수 있게 감싼다 (디스크에 저장하지 않음)
- `Image.open(...)` : PIL로 PNG를 열어 이미지 객체로 만든다 → `plt.imshow(img)`
- `plt.axis('off')` : x, y 축 눈금을 숨긴다
- Day33의 `with open(..., 'wb')` 파일 저장 대신 실행할 때마다 창으로 확인한다

### 2. 메모리 없는 챗봇 (lg08_Memory01)

lg07_ToolNode_graph02의 Tool calling 그래프(chatbot · tools · tools_condition)를 그대로 두고, 질문 두 개를 따로 invoke 했다.

```python
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
```

```
반갑습니다, Mr. Song! 무엇을 도와드릴까요?
========== 두번째 질문 ==========
아직 이름을 말씀해 주신 적이 없어요. 알려주시면 기억해 둘게요.
```

- invoke할 때마다 State가 **새로 시작**되어 messages에는 이번 질문 하나만 들어간다
- `add_messages`는 **한 번 실행하는 동안에만** 메시지를 쌓는다 → 이전 invoke의 대화는 남지 않는다

### 3. InMemorySaver로 대화 기억하기 (lg08_Memory02)

메모리를 세 단계로 추가했다.

```python
# 메모리 추가 첫번째
from langgraph.checkpoint.memory import InMemorySaver

# 메모리 추가 두번째
memory = InMemorySaver()

graph = graph_builder.compile(checkpointer=memory)

# 메모리 추가 세번째
config = {'configurable' : {'thread_id' : 'user01'}}

# 첫번째 질문
response = graph.invoke(
    {'messages' : [('user', '내 이름은 Mr.Song 이야')]},
    config=config
)

print(response['messages'][-1].content)

# 두번째 질문
print('========== 두번째 질문 ==========')
response = graph.invoke({
    'messages' : [('user', '내 이름이 뭐라고 했지?')]},
    config=config)

print(response['messages'][-1].content)
```

```
반갑습니다, Mr. Song! 무엇을 도와드릴까요?
========== 두번째 질문 ==========
Mr. Song이라고 하셨어요.
```

- `InMemorySaver()` : 대화 State를 **메모리(RAM)** 에 저장하는 checkpointer
- `compile(checkpointer=memory)` : 그래프 실행이 끝날 때마다 State(messages)를 저장한다
- `config = {'configurable' : {'thread_id' : 'user01'}}` : 대화방 id
  - 같은 `thread_id`로 다시 invoke하면 저장된 messages를 불러와 새 질문을 이어 붙인다 (`add_messages`)
  - 두 invoke 모두 같은 `config`를 넘겨야 이전 대화를 불러온다
- 메모리(RAM)에만 저장하므로 프로그램을 종료하면 대화 기록이 사라진다

### 4. 병렬 실행 → 병합 (lg09_Parallel_Execution_absorption1)

LLM · Tool 없이 정해진 문장만 돌려주는 노드로 병렬 실행 흐름을 먼저 확인했다.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question : str
    search_result : str
    analysis_result : str
    final_result : str

def search_agent(state : State):
    print('검색 에이전트 실행')
    return {'search_result' : '엔비디아는 AI GPU 시장의 주요 기업이다.'}

def analysis_agent(state : State):
    print('분석 에이전트 실행')
    return {'analysis_result' : 'AI 반도체 수요 증가가 성장 요인이다.'}

def merge_agent(state : State):
    result = (
        state['search_result'] + '\n' +
        state['analysis_result']
    )
    return {'final_result' : result}

builder = StateGraph(State)

builder.add_node('search', search_agent)
builder.add_node('analysis', analysis_agent)
builder.add_node('merge', merge_agent)

########## 병렬 실행 ##########
builder.add_edge(START, 'search')
builder.add_edge(START, 'analysis')

########## 병합 ##########
builder.add_edge('search', 'merge')
builder.add_edge('analysis', 'merge')

builder.add_edge('merge', END)

graph = builder.compile()

response = graph.invoke({
    'question' : '엔비디아의 미래 전망은?'
})

print(response['final_result'])
```

```
분석 에이전트 실행
검색 에이전트 실행
엔비디아는 AI GPU 시장의 주요 기업이다.
AI 반도체 수요 증가가 성장 요인이다.
```

- **병렬 실행** : START에서 `add_edge`를 두 번 → search, analysis가 같은 단계에서 함께 실행된다 (출력 순서는 정해져 있지 않다)
- **병합** : 두 노드 모두 merge로 Edge → search, analysis가 **모두 끝난 뒤** merge가 한 번 실행된다
- 노드마다 State의 **서로 다른 키**(`search_result` / `analysis_result`)에 결과를 써서 서로 덮어쓰지 않는다
- State에 선언하지 않은 키는 노드가 return해도 저장되지 않는다 → 사용할 키를 모두 선언한다
- `add_edge`에는 함수 이름(`search_agent`)이 아니라 `add_node`에서 정한 **노드 이름**(`'search'`)을 쓴다
- `add_node(analysis_agent)`처럼 함수만 넣으면 함수 이름이 노드 이름이 된다
- 결과는 lg07_ToolNode_graph02와 같은 방식으로 그래프 구조를 출력해 확인했다

```
그래프 구조

           __start__
           ╱       ╲
     analysis     search        ← 동시에 실행
           ╲       ╱
             merge              ← 둘 다 끝나면 실행
               │
            __end__
```

### 5. 웹 검색 + LLM 분석 병렬 실행 (lg09_Parallel_Execution_absorption2)

lg08_Memory02를 베이스로 absorption1의 병렬 그래프에 실제 Tavily 웹 검색과 LLM 분석을 넣었다.

```python
llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    max_tokens=8000,
    api_key=api_key,
    base_url=base_url,
)

class State(TypedDict):
    messages : Annotated[list, add_messages]
    search_result : str
    analysis_result : str

# 검색 에이전트
def search_agent(state : State):
    print('검색 에이전트 실행')
    question = state['messages'][-1].content

    result = web_search.invoke({
        'query' : question
    })
    return {'search_result' : str(result)}

# 분석 에이전트
def analysis_agent(state : State):
    print('분석 에이전트 실행')
    question = state['messages'][-1].content

    response = llm.invoke(
        f'다음 질문을 기존 지식을 바탕으로 분석해줘.\n{question}'
    )
    return {'analysis_result' : response.content}

# 결과 통합 에이전트
def merge_agent(state : State):
    result = (
        state['search_result'] + '\n\n' +
        '==============================' + '\n' +
        state['analysis_result']
    )
    return {'messages' : [('assistant', result)]}

query = '엔비디아의 미래 전망에 대해 알려줘'

response = graph.invoke(
    {'messages' : [('user', query)]},
)

print('Assistant :', response['messages'][-1].content)
```

- 질문은 `messages`로 받고, 각 노드에서 `state['messages'][-1].content`로 마지막(사용자) 메시지를 꺼낸다
- search : `web_search.invoke({'query' : question})`로 **노드 안에서 Tool을 직접 실행** → 검색 결과 딕셔너리를 `str()`로 바꿔 저장
- analysis : `llm.invoke(...)`로 LLM이 기존 지식으로 분석한 답변을 저장
- merge : 두 결과를 `=====`로 구분해 합친 뒤 `('assistant', result)`로 messages에 추가 → `response['messages'][-1].content`
- 검색과 LLM 분석이 **동시에 진행**된다 (그래프 구조는 absorption1과 같다)
- State에 `search_result` / `analysis_result` 키가 없으면 merge에서 `KeyError: 'search_result'`가 난다
- `max_tokens=8000` : gpt-5.6-terra는 답변 전에 추론(reasoning) 토큰을 먼저 쓴다. 2000이면 추론에 다 써서(`reasoning_tokens: 2000`, `finish_reason: 'length'`) 분석 답변이 빈 문자열이 된다

```
검색 에이전트 실행 / 분석 에이전트 실행
Assistant : {'query': '엔비디아의 미래 전망에 대해 알려줘', ..., 'results': [{'url': ..., 'title': ..., 'content': ...}, ...]}

==============================
엔비디아의 미래 전망은 전반적으로 AI 인프라 시장의 핵심 수혜자라는 강한 강점과, ...
```

### 6. lg07_ToolNode_graph 파일 이름 변경

- `lg07_ToolNode_graph.py` → `lg07_ToolNode_graph01.py` (파일 저장 버전 01, 창 출력 버전 02)
- `docs/Day33.md`와 README 디렉토리 구조의 파일 이름도 함께 바꿨다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `LangGraph/lg07_ToolNode_graph02.py` | `Image.open(BytesIO(png_data))` + `plt.imshow`로 그래프 구조를 창에 바로 출력 |
| `LangGraph/lg08_Memory01.py` | 메모리 없이 invoke 두 번 → 두번째 질문에서 이름을 기억하지 못함 |
| `LangGraph/lg08_Memory02.py` | `InMemorySaver` + `compile(checkpointer=memory)` + `config`(`thread_id`)로 이전 대화 기억 |
| `LangGraph/lg09_Parallel_Execution_absorption1.py` | START → search · analysis 병렬 실행 → merge 병합 (정해진 문장으로 흐름 확인) |
| `LangGraph/lg09_Parallel_Execution_absorption2.py` | search(Tavily 웹 검색) · analysis(LLM 분석) 병렬 실행 → merge에서 합쳐 assistant 메시지로 |
| `LangGraph/lg07_ToolNode_graph01.py` | `lg07_ToolNode_graph.py`에서 이름 변경 (수정) |
| `docs/Day34.md` | Day34 학습 기록 신규 작성 |
| `docs/Day33.md` | lg07 파일 이름 변경 반영, 하단 nav에 Day34 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day33 lg07 이름, Day34) / 진행도(34일, 43%) 갱신 (수정) |

---

## 📊 실행 결과

| 파일 | 결과 |
|---|---|
| lg07_ToolNode_graph02 | matplotlib 창에 `__start__ → chatbot → (점선) tools / __end__, tools → chatbot` 그래프 출력 |
| lg08_Memory01 | 두번째 질문 : `아직 이름을 말씀해 주신 적이 없어요.` |
| lg08_Memory02 | 두번째 질문 : `Mr. Song이라고 하셨어요.` |
| lg09_absorption1 | search · analysis 실행 후 두 문장을 합친 `final_result` 출력, 그래프 `__start__ → analysis / search → merge → __end__` |
| lg09_absorption2 | Tavily 검색 결과 + `=====` + LLM 분석 답변이 하나의 assistant 메시지로 출력 |

---

## 💻 핵심 개념

### 그래프 이미지 저장 vs 출력

```
draw_mermaid_png()  →  PNG bytes
   ├─ open(path, 'wb').write(png_data)               파일로 저장 (lg07_ToolNode_graph01)
   └─ Image.open(BytesIO(png_data)) → plt.imshow    창에 바로 출력 (lg07_ToolNode_graph02)
```

### 메모리 (checkpointer)

```
메모리 없음                                   InMemorySaver + thread_id
invoke 1 : [human 이름은 Mr.Song]             invoke 1 : [human 이름은 Mr.Song] [ai 반갑습니다]  ──저장──▶ user01
invoke 2 : [human 내 이름은?]                 invoke 2 : user01 불러오기 + [human 내 이름은?]
           → 이름을 모름                                 → Mr. Song이라고 하셨어요

① from langgraph.checkpoint.memory import InMemorySaver
② graph = graph_builder.compile(checkpointer=InMemorySaver())
③ graph.invoke(입력, config={'configurable' : {'thread_id' : 'user01'}})
```

### 병렬 실행 → 병합

```
                ┌──▶ search   ──┐
       START ───┤               ├──▶ merge ──▶ END
                └──▶ analysis ──┘

add_edge(START, 'search')       add_edge('search', 'merge')
add_edge(START, 'analysis')     add_edge('analysis', 'merge')
   └ 같은 단계에서 동시 실행         └ 둘 다 끝나야 merge 한 번 실행

State 키 : search → search_result, analysis → analysis_result (다른 키) → merge 에서 합치기
```

---

## 성과

- 그래프 구조 PNG를 `BytesIO` · `Image.open` · `plt.imshow`로 파일 저장 없이 바로 확인했다
- 메모리 없는 챗봇이 invoke 사이의 대화를 기억하지 못하는 것을 확인했다
- `InMemorySaver` · `checkpointer` · `thread_id`로 이전 대화(이름)를 기억하는 챗봇을 만들었다
- START에서 두 노드로 Edge를 나눠 병렬 실행하고, merge 노드에서 결과를 합치는 그래프를 만들었다
- 웹 검색(Tavily)과 LLM 분석을 동시에 실행해 하나의 답변으로 합쳤다

---

## 💡 주요 학습 포인트

1. **PNG bytes는 `BytesIO`로 감싸서 연다**: `Image.open(BytesIO(png_data))` → `plt.imshow(img)`
2. **invoke마다 State는 새로 시작**: `add_messages`는 한 번 실행하는 동안에만 메시지를 쌓는다
3. **메모리 추가는 세 단계**: `InMemorySaver` import → `compile(checkpointer=memory)` → `config`의 `thread_id`
4. **같은 `thread_id`끼리 대화가 이어진다**: 두 invoke 모두 같은 `config`를 넘긴다
5. **`InMemorySaver`는 RAM에 저장**: 프로그램을 종료하면 대화 기록이 사라진다
6. **병렬 실행 = START에서 Edge 여러 개**: 같은 단계의 노드가 동시에 실행되고, 출력 순서는 정해져 있지 않다
7. **병합 = 여러 노드에서 한 노드로 Edge**: 앞 노드가 모두 끝나야 merge가 실행된다
8. **병렬 노드는 State의 다른 키에 쓴다**: `search_result` / `analysis_result`로 나눠야 서로 덮어쓰지 않는다
9. **State에 선언한 키만 저장된다**: 선언하지 않으면 return해도 버려져 `KeyError`가 난다
10. **`add_edge`에는 노드 이름**: 함수 이름이 아니라 `add_node('search', ...)`에서 정한 이름
11. **Tool은 노드 안에서 직접 실행할 수도 있다**: `web_search.invoke({'query' : question})` → 검색이 항상 실행된다
12. **추론 모델은 `max_tokens`를 넉넉히**: 추론 토큰이 한도를 다 쓰면 답변이 빈 문자열이 된다

---

[⬅️ Day33](Day33.md) · [🏠 전체 목차](../README.md)
