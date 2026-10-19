# Day32 - LangGraph 입문 (TypedDict State, add_messages Reducer, StateGraph Node · Edge 챗봇)

**학습 기간:** 2026-10-19

> LangGraph로 챗봇 그래프를 처음 만들었다. 먼저 그래프의 State를 정의할 때 쓰는 `TypedDict`를 일반 `Dict` 타입 힌트와 비교했고, State의 `messages`에 붙이는 Reducer `add_messages`가 메시지 리스트를 **덮어쓰지 않고 이어 붙이며, id가 같으면 바꾼다**는 것을 확인했다. 그다음 `StateGraph`에 `add_node`로 chatbot 노드를 넣고 `add_edge`로 `START → chatbot → END`를 연결해 `compile` · `invoke`로 LLM 답변을 받았다. 마지막으로 chatbot 앞뒤에 prepare / finish 노드를 추가해 `START → prepare → chatbot → finish → END` 순서로 실행하고, 노드마다 돌려준 메시지 4개가 순서대로 쌓이는 것을 출력했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| Dict 타입 힌트 | `Dict[str, str]` → 모든 값이 같은 자료형 (나이도 `'39'` 문자열) |
| TypedDict | `class Person(TypedDict)` → 키마다 자료형을 따로 정한 딕셔너리 형식, 결과는 그냥 dict |
| TypedDict 만들기 | `Person(name=..., age=..., job=...)` 또는 `person2 : Person = {...}` |
| Reducer | `messages : Annotated[list, add_messages]` → 새 메시지가 오면 덮어쓰지 않고 합친다 |
| add_messages | `add_messages(msg1, msg2)` → 두 리스트를 이어 붙임, **id가 같으면 새 메시지로 바뀜** |
| LangGraph 순서 | State 정의 → Node 정의 → `StateGraph` + `add_node` → `add_edge` → `compile` → `invoke` |
| Node | State를 받아 바꿀 값만 `{'messages' : [...]}`로 돌려주는 함수 |
| Edge | `START` · `END`와 노드 이름으로 실행 순서 연결 |
| 노드 추가 | prepare / finish 노드 → 최종 messages 4개 (human, ai, ai, ai) |

---

## 📖 핵심 학습 내용

### 1. Dict vs TypedDict (lg01_TypedDict)

```python
from typing import Dict, TypedDict

sample_dict : Dict[str, str] = {
    'name' : '김말똥',
    'age' : '39',
    'job' : '회사원'
}

class Person(TypedDict) :
    name : str
    age : int
    job : str

person = Person(name='홍길동', age=40, job='도적')
print(person)   # {'name': '홍길동', 'age': 40, 'job': '도적'}

person2 : Person = {'name' : '김개똥', 'age' : 20, 'job' : '전사'}
print(person2)  # {'name': '김개똥', 'age': 20, 'job': '전사'}
```

- `Dict[str, str]` : 키 · 값 자료형을 하나씩만 적을 수 있다 → 나이도 문자열 `'39'`
- `TypedDict` : 키마다 자료형을 따로 정한다 (`name : str`, `age : int`)
- 만드는 방법 2가지 : `Person(키=값)` / `변수 : Person = {...}` → 둘 다 결과는 그냥 딕셔너리
- LangGraph의 State를 이 `TypedDict` 형식으로 정의한다

### 2. add_messages Reducer (lg02_add_messages)

```python
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class MyData(TypedDict):
    messages : Annotated[list, add_messages] # Reducer

from langchain_core.messages import AIMessage, HumanMessage

msg1 = [HumanMessage(content='안녕', id='1')]
msg2 = [AIMessage(content='방가', id='2')]
results = add_messages(msg1, msg2)

msg3 = [HumanMessage(content='안녕', id='3')]
msg4 = [AIMessage(content='방가', id='3')]
results2 = add_messages(msg3, msg4)
```

```
results  : [HumanMessage(content='안녕', ..., id='1'), AIMessage(content='방가', ..., id='2', ...)]
results2 : [AIMessage(content='방가', ..., id='3', ...)]
```

- `Annotated[list, add_messages]` : messages는 list이고, 값이 들어올 때 **add_messages로 합치라는 표시** (Reducer)
- id가 다르면(`'1'`, `'2'`) 두 메시지가 순서대로 이어 붙는다
- id가 같으면(`'3'`, `'3'`) 추가되지 않고 **새 메시지(AIMessage)로 덮어쓴다** → 결과 1개
- `HumanMessage` · `AIMessage`는 Day27 대화 기억 챗봇에서 쓴 메시지 클래스

### 3. LangGraph 챗봇 (lg03_chatbot)

rag06_monogpt를 베이스로 같은 `ChatOpenAI` (monogpt, `gpt-5.6-terra`)에 `max_tokens=1000`을 넣고 그래프를 만들었다.

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. State 정의
class State(TypedDict) :
    messages : Annotated[list, add_messages]
    aaa : str
    bbb : str
    ccc : str

# 2. Node 정의
def chatbot(state : State) :
    return {'messages' : [llm.invoke(state['messages'])]}

# 3. Graph 구성
graph_01 = StateGraph(State)
graph_01.add_node('chatbot', chatbot)

# 4. Edge 추가
graph_01.add_edge(START, 'chatbot')
graph_01.add_edge('chatbot', END)

# 5. Graph Compile
graph = graph_01.compile()

# 6. Graph Execution
query = '삼성전자의 주가를 예측해줘'
response = graph.invoke({
    'messages' : [('user', query)]
})
```

- **State** : 그래프 안에서 노드끼리 주고받는 데이터. messages 외에 `aaa` · `bbb` · `ccc` 같은 다른 키도 둘 수 있다 (`state = {'messages' : [], 'aaa' : 'abc', ...}`와 같은 의미)
- **Node** : State를 받아서 **바꿀 키만** 딕셔너리로 돌려주는 함수 → chatbot은 LLM 답변을 `messages`에 돌려준다
- `StateGraph(State)` : State 형식으로 그래프를 만든다
- `add_node('chatbot', chatbot)` : (노드 이름, 실행할 함수)
- `add_edge(START, 'chatbot')` / `add_edge('chatbot', END)` : 시작 → chatbot → 끝
- `compile()` : 실행할 수 있는 그래프로 만든다 → `invoke`로 실행
- `invoke({'messages' : [('user', query)]})` : `('user', 질문)` 튜플이 HumanMessage로 들어간다
- `response` = 최종 State 딕셔너리
  - `response['messages'][0]` → HumanMessage (`.type` = `human`)
  - `response['messages'][1]` → AIMessage (`.type` = `ai`), `.content`가 LLM 답변

### 4. Node · Edge 추가 (lg04_Node_Edge_Add)

lg03을 베이스로 chatbot 앞뒤에 노드 2개를 추가했다.

```python
def prepare(state : State) :
    return {'messages' : 
            [AIMessage(content='질문을 접수했습니다.')]
            }
    
def chatbot(state : State) :
    return {'messages' : [llm.invoke(state['messages'])]}

def finish(state : State) : 
    return {'messages' : 
                [AIMessage(content='답변 처리가 완료되었습니다.')]
            }

graph_01 = StateGraph(State)
graph_01.add_node('prepare', prepare)
graph_01.add_node('chatbot', chatbot)
graph_01.add_node('finish', finish)

graph_01.add_edge(START, 'prepare')
graph_01.add_edge('prepare', 'chatbot')
graph_01.add_edge('chatbot', 'finish')
graph_01.add_edge('finish', END)

graph = graph_01.compile()

for i, msg in enumerate(response['messages']):
    print(f'[{i}] {msg.type} : {msg.content}')
```

- prepare / finish는 LLM을 부르지 않고 고정된 `AIMessage`만 돌려준다
- 노드마다 돌려준 메시지가 `add_messages`로 **차례대로 쌓인다**
- chatbot은 질문 + prepare 메시지까지 들어간 messages로 LLM을 부른다
- `enumerate`로 번호를 붙여 `msg.type` (human / ai)과 `msg.content`를 출력

```
[0] human : 삼성전자의 주가를 예측해줘
[1] ai : 질문을 접수했습니다.
[2] ai : 삼성전자 주가를 정확히 예측하는 것은 불가능합니다. ... (LLM 답변)
[3] ai : 답변 처리가 완료되었습니다.
```

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `LangGraph/lg01_TypedDict.py` | `Dict[str, str]` 타입 힌트 vs `TypedDict` 클래스, `Person(...)` / `person2 : Person = {...}` 두 가지 생성 |
| `LangGraph/lg02_add_messages.py` | `Annotated[list, add_messages]` Reducer, id가 다르면 이어 붙이고 같으면 덮어쓰는 것 확인 |
| `LangGraph/lg03_chatbot.py` | State 정의 → chatbot 노드 → `StateGraph` · `add_node` · `add_edge(START / END)` · `compile` · `invoke` 챗봇 |
| `LangGraph/lg04_Node_Edge_Add.py` | prepare / finish 노드 추가, `START → prepare → chatbot → finish → END`, 메시지 4개 출력 |
| `docs/Day32.md` | Day32 학습 기록 신규 작성 |
| `docs/Day31.md` | 하단 nav에 Day32 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(LangGraph, Day32) / 진행도(32일, 40%) 갱신 (수정) |

---

## 📊 실행 결과

| 파일 | 결과 |
|---|---|
| lg01 | `{'name': '홍길동', 'age': 40, 'job': '도적'}` / `{'name': '김개똥', 'age': 20, 'job': '전사'}` |
| lg02 | id `'1'` · `'2'` → 메시지 2개 / id `'3'` · `'3'` → AIMessage 1개 (덮어씀) |
| lg03 | `response['messages']` = [HumanMessage, AIMessage] |
| lg04 | `response['messages']` = [human, ai (prepare), ai (chatbot), ai (finish)] 4개 |

---

## 💻 핵심 개념

### LangGraph 만드는 순서

```
1. State 정의        class State(TypedDict) : messages : Annotated[list, add_messages]
2. Node 정의         def chatbot(state : State) : return {'messages' : [...]}
3. Graph 구성        graph_01 = StateGraph(State)  →  graph_01.add_node('chatbot', chatbot)
4. Edge 추가         graph_01.add_edge(START, 'chatbot')  /  graph_01.add_edge('chatbot', END)
5. Graph Compile     graph = graph_01.compile()
6. Graph Execution   response = graph.invoke({'messages' : [('user', query)]})
```

### 그래프 흐름

```
lg03   START ──▶ chatbot ──▶ END

lg04   START ──▶ prepare ──▶ chatbot ──▶ finish ──▶ END

       messages : [human 질문] + [ai 접수] + [ai LLM 답변] + [ai 완료]   (add_messages 로 차례대로 쌓임)
```

### add_messages

```
add_messages([Human id=1], [AI id=2])  →  [Human id=1, AI id=2]    (id 다름 → 이어 붙임)
add_messages([Human id=3], [AI id=3])  →  [AI id=3]                (id 같음 → 덮어씀)
```

---

## 성과

- `TypedDict`로 키마다 자료형을 정한 딕셔너리 형식을 만들고, 일반 `Dict` 타입 힌트와 비교했다
- `add_messages` Reducer가 메시지를 이어 붙이고, id가 같으면 덮어쓰는 것을 확인했다
- `StateGraph` · `add_node` · `add_edge` · `compile` · `invoke`로 LangGraph 챗봇을 처음 만들었다
- 노드를 3개로 늘려 Edge 순서대로 실행되고, 노드마다 메시지가 쌓이는 것을 출력으로 확인했다

---

## 💡 주요 학습 포인트

1. **TypedDict는 키마다 자료형을 정한 딕셔너리 형식**: 만든 결과는 그냥 dict이고, LangGraph State를 이 형식으로 정의한다
2. **Reducer = 값을 합치는 방법**: `Annotated[list, add_messages]`면 새 메시지가 기존 리스트에 추가된다
3. **add_messages는 id로 구분한다**: id가 같은 메시지는 추가되지 않고 새 메시지로 바뀐다
4. **Node는 바꿀 키만 돌려준다**: `{'messages' : [...]}`만 돌려줘도 나머지 State는 그대로다
5. **Edge로 실행 순서를 정한다**: `START`에서 시작해 `END`에서 끝나도록 노드를 연결한다
6. **compile 해야 실행할 수 있다**: `StateGraph`는 설계도, `compile()` 결과로 `invoke`한다
7. **`('user', query)`는 HumanMessage로 들어간다**: 결과 `response['messages']`에서 `.type`으로 human / ai를 구분한다
8. **노드를 추가하면 메시지가 순서대로 쌓인다**: lg04는 질문 1개 + 노드 3개 = 메시지 4개

---

[⬅️ Day31](Day31.md) · [🏠 전체 목차](../README.md)
