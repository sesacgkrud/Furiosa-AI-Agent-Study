# Day33 - LangGraph Tool Calling (TavilySearch · DuckDuckGo 웹 검색, ToolNode, tools_condition 조건부 Edge, 그래프 이미지 저장)

**학습 기간:** 2026-10-20

> LLM이 혼자 답할 수 없는 **오늘 주가** 같은 실시간 정보를 웹 검색 Tool로 찾아서 답하게 했다 (tool calling). `TavilySearch`로 웹 검색 Tool을 만들어 `llm.bind_tools(tools)`로 LLM에 연결하고, LangGraph에 Tool을 실행하는 `ToolNode`를 `tools` 노드로 넣었다. chatbot 다음 노드는 `add_conditional_edges('chatbot', tools_condition)`로 정해 **tool_calls가 있으면 tools, 없으면 END**로 가고, tools 실행 뒤에는 다시 chatbot으로 돌아와 검색 결과로 최종 답변을 만들었다. 같은 그래프에서 Tool만 API 키가 필요 없는 `DuckDuckGoSearchRun`으로 바꿔 보고, Tavily와 DuckDuckGo 두 Tool을 함께 연결해 봤다. 마지막으로 `graph.get_graph().draw_mermaid_png()`로 그래프 구조를 PNG 이미지로 저장했다. Day32의 lg04에는 `messages : list`로 두면 덮어쓰게 된다는 주석을 추가했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| tool calling | LLM이 외부 기능(Tool)을 가져와서 사용 → 실시간 정보도 답할 수 있다 |
| TavilySearch | `TavilySearch(max_results=5)` 웹 검색 Tool, `.env`의 `TAVILY_API_KEY` 사용 |
| bind_tools | `llm.bind_tools(tools)` → LLM이 필요하면 답변 대신 **tool_calls**를 돌려준다 |
| ToolNode | `add_node('tools', ToolNode(tools))` → tool_calls대로 Tool을 실행해 결과를 messages에 추가 |
| 조건부 Edge | `add_conditional_edges('chatbot', tools_condition)` → tool_calls 있으면 `'tools'`, 없으면 `END` |
| 다시 chatbot | `add_edge('tools', 'chatbot')` → 검색 결과를 보고 최종 답변 |
| DuckDuckGo | `DuckDuckGoSearchRun` → API 키 없이 웹 검색 (`pip install ddgs`) |
| Tool 2개 | `tools = [web_search1, web_search2]` → LLM이 어떤 Tool을 부를지 고른다 |
| 그래프 이미지 | `graph.get_graph().draw_mermaid_png()` + `open(..., 'wb')`로 PNG 저장 |
| lg04 (수정) | `# messages : list   # 이렇게 하면 계속 덮어써짐` 주석 추가 |

---

## 📖 핵심 학습 내용

### 1. Tavily 웹 검색 + ToolNode 그래프 (lg05_ToolNode)

```python
llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    max_tokens=2000,
    api_key=api_key,
    base_url=base_url,
)

# pip install langchain-tavily
from langchain_tavily import TavilySearch

web_search = TavilySearch(
    max_results=5,
)

tools = [web_search]
llm_with_tools = llm.bind_tools(tools)
```

- **tool calling** : LLM이 외부 기능(Tool)을 가져와서 사용하는 것 → LLM이 모르는 오늘 주가를 웹 검색으로 찾는다
- `TavilySearch(max_results=5)` : 웹 검색 결과를 최대 5개 가져오는 Tool (`.env`에 `TAVILY_API_KEY`)
- Tool만 따로 `web_search.invoke({'query' : ...})`로 실행해 보면 `query` · `results`(url, title, content, score) · `response_time` 딕셔너리가 나온다
- `llm.bind_tools(tools)` : LLM에 사용할 수 있는 Tool 목록을 알려준다 → 필요하다고 판단하면 **답변 대신 tool_calls**(어떤 Tool을 어떤 인자로 부를지)를 돌려준다

```python
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages : Annotated[list, add_messages]

def chatbot(state : State):
    return {'messages' : [llm_with_tools.invoke(state['messages'])]}

graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)

tool_node = ToolNode(tools)

# 아래 둘 다 같은 의미, 편한 방법으로 사용하면 됨
# graph_builder.add_node('tools', tool_node)
graph_builder.add_node('tools', ToolNode(tools))

graph_builder.add_edge(START, 'chatbot')

# graph_builder.add_edge('chatbot', 'tools')
graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition # tool_calls 이 있는지?
)

graph_builder.add_edge('tools', 'chatbot')

graph = graph_builder.compile()

query = '삼성전자의 오늘 주가를 알려줘'
response = graph.invoke({
    'messages' : [('user', query)]
})

print('AI :', response['messages'][-1].content)
```

- `ToolNode(tools)` : tool_calls에 적힌 Tool을 실제로 실행하고, 결과를 messages에 추가하는 **미리 만들어진 노드**
- `add_node('tools', tool_node)` / `add_node('tools', ToolNode(tools))` : 변수에 담든 바로 넣든 같은 의미
- `add_conditional_edges('chatbot', tools_condition)` : **조건부 Edge**, chatbot 다음 노드를 조건으로 고른다
  - tool_calls가 있으면 `'tools'` 반환
  - tool_calls가 없으면 `END` 반환 (tools_condition에 END 포함)
- `add_edge('chatbot', 'tools')`처럼 일반 Edge로 연결하면 항상 tools로만 간다
- `add_edge('tools', 'chatbot')` : Tool 결과를 들고 다시 chatbot으로 → 검색 결과로 최종 답변
- 최종 답변은 messages의 **마지막** → `response['messages'][-1].content`

### 2. DuckDuckGo 웹 검색 (lg06_01_ddg)

lg05를 베이스로 Tavily 부분은 주석 처리하고 Tool만 DuckDuckGo로 바꿨다.

```python
# pip install ddgs
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

web_search = DuckDuckGoSearchRun(
    max_results=5,
)

tools = [web_search]
llm_with_tools = llm.bind_tools(tools)
```

- `DuckDuckGoSearchRun` : API 키 없이 쓸 수 있는 웹 검색 Tool (`pip install ddgs`), 검색 결과를 문자열 하나로 돌려준다
- State · Node · Edge(ToolNode, tools_condition)는 lg05와 그대로 → **Tool만 바꿔 끼우면 된다**
- `max_results=5`는 `DuckDuckGoSearchRun`에는 적용되지 않는 인자다 (내부 `DuckDuckGoSearchAPIWrapper`의 기본값이 5라서 결과 개수는 같다)
- `exit()` : Tool 결과만 확인하고 그 아래 그래프는 실행하지 않을 때 사용

### 3. Tool 2개 연결 (lg06_02_ddg_2ea)

```python
from langchain_tavily import TavilySearch

web_search1 = TavilySearch(
    max_results=5,
)

from langchain_community.tools import DuckDuckGoSearchRun

web_search2 = DuckDuckGoSearchRun(
    max_results=5,
)

tools = [web_search1, web_search2]
llm_with_tools = llm.bind_tools(tools)
```

- `tools` 리스트에 Tool 두 개를 넣고 `bind_tools`로 함께 알려준다
- LLM이 질문을 보고 **어떤 Tool을 부를지 스스로 고른다** (하나만 또는 둘 다)
- `ToolNode(tools)`도 두 Tool을 모두 가지고 있어 tool_calls의 Tool 이름(`tavily_search` / `duckduckgo_search`)으로 찾아 실행한다

### 4. 그래프 이미지 저장 (lg07_ToolNode_graph01)

lg05를 베이스로 그래프 실행은 주석 처리하고 그래프 구조를 이미지로 저장했다.

```python
path = './_save/image/'
png_data = graph.get_graph().draw_mermaid_png()

# 파일 입출력 시 사용
with open(path + 'lg07_ToolNode.png', 'wb') as file:
    file.write(png_data)

# 위와 같은 코드
# file = open(path + 'lg07_ToolNode1.png', 'wb')
# file.write(png_data)
# file.close()
```

- `graph.get_graph()` : compile된 그래프의 노드 · Edge 구조
- `.draw_mermaid_png()` : 구조를 Mermaid 다이어그램 **PNG bytes**로 만든다
- `open(파일, 'wb')` : 바이너리(`b`) 쓰기(`w`) 모드 → bytes를 그대로 파일에 쓴다
- `with open(...) as file:` : 블록이 끝나면 자동으로 닫힌다 = `open` · `write` · `close` 세 줄과 같다
- `_save/`는 `.gitignore`라 이미지는 GitHub에 올리지 않는다

```
저장된 그림 (lg07_ToolNode.png)

        __start__
            │
         chatbot  ◀──────┐
         ┆      ┆        │
         ▼      ▼        │
     __end__   tools ────┘

   ┆ (점선) = 조건부 Edge (tools_condition)
```

### 5. lg04_Node_Edge_Add 주석 추가 (수정)

```python
class State(TypedDict) :
    # messages : list   # 이렇게 하면 계속 덮어써짐
    messages : Annotated[list, add_messages]
```

- Reducer 없이 `list`로만 두면 노드가 돌려준 messages가 이전 값을 **덮어쓴다** → `add_messages`를 붙여야 쌓인다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `LangGraph/lg05_ToolNode.py` | `TavilySearch` 웹 검색 Tool + `bind_tools`, `ToolNode` tools 노드, `add_conditional_edges` · `tools_condition`, tools → chatbot |
| `LangGraph/lg06_01_ddg.py` | Tool을 `DuckDuckGoSearchRun`으로 바꿔 같은 그래프 실행 (API 키 불필요) |
| `LangGraph/lg06_02_ddg_2ea.py` | `TavilySearch` + `DuckDuckGoSearchRun` Tool 2개를 함께 `bind_tools` · `ToolNode` |
| `LangGraph/lg07_ToolNode_graph01.py` | `get_graph().draw_mermaid_png()` + `with open(..., 'wb')`로 그래프 구조 PNG 저장 |
| `LangGraph/lg04_Node_Edge_Add.py` | `# messages : list   # 이렇게 하면 계속 덮어써짐` 주석 추가 (수정) |
| `docs/Day33.md` | Day33 학습 기록 신규 작성 |
| `docs/Day32.md` | 하단 nav에 Day33 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day33) / 진행도(33일, 41%) 갱신 (수정) |

---

## 📊 실행 결과

| 파일 | 결과 |
|---|---|
| lg05 | Tavily 검색 결과로 삼성전자 오늘 주가 답변 (`response['messages'][-1].content`) |
| lg06_01 | DuckDuckGo 검색 결과로 같은 질문 답변 |
| lg06_02 | Tavily · DuckDuckGo 두 Tool 중 LLM이 골라서 검색 후 답변 |
| lg07 | `_save/image/lg07_ToolNode.png` — `__start__ → chatbot`, chatbot에서 점선으로 `tools` / `__end__`, `tools → chatbot` |

---

## 💻 핵심 개념

### Tool calling 그래프

```
START ──▶ chatbot ──(tools_condition)──┬── tool_calls 있음 ──▶ tools (ToolNode) ──┐
            ▲                          └── tool_calls 없음 ──▶ END                 │
            └──────────────────────── 검색 결과를 들고 다시 ────────────────────────┘
```

### messages 흐름

```
[human]  삼성전자의 오늘 주가를 알려줘
[ai]     tool_calls : tavily_search(query=...)          ← chatbot (llm_with_tools)
[tool]   검색 결과                                       ← tools (ToolNode)
[ai]     오늘 삼성전자 주가는 ...                         ← chatbot, tool_calls 없음 → END
                                                          response['messages'][-1].content
```

### Tool 바꿔 끼우기

```
TavilySearch(max_results=5)         langchain_tavily       TAVILY_API_KEY 필요
DuckDuckGoSearchRun()               langchain_community    API 키 X (pip install ddgs)

tools = [Tool, ...]  →  llm.bind_tools(tools)  +  ToolNode(tools)    ← 같은 tools 리스트를 둘 다에
```

### Edge 종류

```
add_edge('tools', 'chatbot')                         항상 같은 다음 노드
add_conditional_edges('chatbot', tools_condition)    함수 결과('tools' / END)에 따라 다음 노드
```

---

## 성과

- `TavilySearch` 웹 검색 Tool을 `bind_tools`로 LLM에 연결해 실시간 주가 질문에 답하게 했다
- `ToolNode`와 `tools_condition` 조건부 Edge로 Tool이 필요할 때만 실행되는 그래프를 만들었다
- tools → chatbot으로 돌아와 검색 결과로 최종 답변을 만드는 흐름을 확인했다
- Tool을 `DuckDuckGoSearchRun`으로 바꿔도 그래프는 그대로 쓸 수 있는 것을 확인했다
- Tool 두 개를 함께 연결해 LLM이 Tool을 고르게 했다
- `draw_mermaid_png`로 그래프 구조를 이미지로 저장해 조건부 Edge(점선)를 눈으로 확인했다

---

## 💡 주요 학습 포인트

1. **tool calling = LLM이 외부 기능을 가져와서 사용**: LLM이 모르는 실시간 정보도 웹 검색 Tool로 답할 수 있다
2. **`bind_tools`는 Tool 목록을 알려줄 뿐**: LLM은 tool_calls만 돌려주고, 실제 실행은 `ToolNode`가 한다
3. **`ToolNode(tools)`는 미리 만들어진 노드**: `add_node('tools', ...)`로 넣고, 결과는 messages에 추가된다
4. **조건부 Edge**: `add_conditional_edges('chatbot', tools_condition)` → tool_calls 있으면 tools, 없으면 END
5. **tools 다음은 다시 chatbot**: `add_edge('tools', 'chatbot')`로 검색 결과를 보고 최종 답변을 만든다
6. **최종 답변은 마지막 메시지**: `response['messages'][-1].content`
7. **Tool만 바꿔 끼우면 된다**: Tavily → DuckDuckGo로 바꿔도 State · Node · Edge는 그대로
8. **Tool 여러 개도 리스트로**: `tools = [web_search1, web_search2]`, LLM이 어떤 Tool을 부를지 고른다
9. **인자가 실제로 적용되는지 확인**: `DuckDuckGoSearchRun(max_results=5)`의 `max_results`는 적용되지 않는다 (기본값 5)
10. **그래프 구조는 이미지로 확인**: `get_graph().draw_mermaid_png()` + `with open(..., 'wb')`
11. **Reducer가 없으면 덮어쓴다**: `messages : list`가 아니라 `Annotated[list, add_messages]` (lg04 주석)

---

[⬅️ Day32](Day32.md) · [🏠 전체 목차](../README.md) · [Day34 ➡️](Day34.md)
