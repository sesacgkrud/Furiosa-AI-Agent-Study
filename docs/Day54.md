# Day54 - 파이썬 데코레이터 (Decorator) 기본 구조 · @ 문법 & 함수 실행 시간 측정 timer 데코레이터 · RNGD LangChain Tool Calling (@tool · bind_tools · tool_calls)

**학습 기간:** 2026-11-18

> 기존 함수를 고치지 않고 기능을 덧붙이는 파이썬 데코레이터(Decorator) 를 만들어, 함수 실행 전 · 후에 문장을 출력했다 (p03_Decorator).
>
> `time.perf_counter()` 로 함수 실행 시간을 재는 `timer` 데코레이터를 만들고, `@timer` 하나로 서로 다른 두 함수에 똑같이 적용했다 (p04_Deco_timer).
>
> RNGD 에서는 furiosa-llm 서버(Qwen3-8B-FP8) 를 `--enable-auto-tool-choice --tool-call-parser hermes` 옵션으로 띄우고, LangChain `@tool` 데코레이터로 만든 함수를 `bind_tools` 로 등록해 LLM 이 돌려주는 `tool_calls` 를 확인했다 (f31 · f32). 이어서 tool_calls 의 args 로 도구를 실제 실행하고 (f33), 도구 3 개(add · multiply · subtract) 중 LLM 이 질문에 맞는 도구를 골라 쓰게 했다 (f34).

---

## 🎯 한눈에 보기

| 파일 | 데코레이터 | 출력 |
|---|---|---|
| p03_Decorator | `deco` : 실행 전 · 후 문장 | 함수 실행 전 → 안녕하세요. → 함수 실행 후 |
| p04_Deco_timer | `timer` : 실행 시간 측정 | hello 약 1.0008 초 / study 약 1.5005 초 |
| f31_tool_test | `@tool` add + `bind_tools` → tool_calls 확인 | name 'add', args {'aaa': 237341, 'bbb': 31321}, finish_reason 'tool_calls' |
| f32_tool_calls | tool_calls 를 for 문으로 꺼내기 | tool 이름 · 매개변수 · 호출 ID |
| f33_tool_invoke | `add.invoke(call["args"])` 로 실제 실행 | 21 + 32 = 53 |
| f34_multi_tools | 도구 3 개 중 LLM 이 선택 + `tool_map` 실행 | add 42 / multiply 96 / subtract 50 |

---

## 📖 핵심 학습 내용

### 1. 데코레이터 기본 구조 (p03_Decorator)

```python
# Decorator : 기존 함수를 수정하지 않고, 함수에 새로운 기능을 추가하는 문법

def deco(func):
    def wrapper():
        print("함수 실행 전")
        func()
        print("함수 실행 후")
    return wrapper

@deco
def hello2():
    print("안녕하세요.")

hello2()
# 함수 실행 전
# 안녕하세요.
# 함수 실행 후
```

- `deco(func)` : 꾸밀 함수를 인자로 받는다 (함수도 값처럼 넘길 수 있다)
- `wrapper()` : 안쪽 함수에서 `func()` 앞뒤에 추가 기능을 넣는다
- `return wrapper` : 실행 결과가 아니라 wrapper 함수 자체를 돌려준다 (괄호 없음)
- `@deco` 를 함수 위에 붙이면 `hello2 = deco(hello2)` 와 같다 → `hello2()` 를 부르면 wrapper 가 실행된다
- 데코레이터가 없는 `hello()` 는 "안녕하세요." 만 출력 → 같은 함수 코드에 앞뒤 기능만 추가된 것을 비교

### 2. 실행 시간 측정 timer 데코레이터 (p04_Deco_timer)

```python
import time

def timer(func):
    def wrapper():
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        print("걸린 시간: ", round(end - start, 6), "초")
    return wrapper

@timer
def hello():
    print("안녕하세요.")
    time.sleep(1)       # 1초 간 정지

@timer
def study():
    print("NPU 공부중이다.")
    time.sleep(1.5)

hello()     # 걸린 시간:  1.000771 초
study()     # 걸린 시간:  1.500456 초
```

- `time.perf_counter()` 를 `func()` 앞뒤로 찍어 차이를 걸린 시간으로 출력, `round(값, 6)` 소수 6 자리
- `time.sleep(초)` 만큼 멈춰서 hello 약 1 초, study 약 1.5 초 (print 등 실행 시간이 조금 더해지고 실행마다 조금씩 다르다)
- 시간 측정 코드를 함수마다 쓰지 않고 `@timer` 만 붙여 여러 함수에 공통 적용
- 실무 활용 : 함수 실행 시간 측정 · 로그 기록 · 사용자 권한 검사 · 예외 처리 · 함수 호출 횟수 기록

### 3. RNGD 서버 Tool Calling 옵션 (kingkeras.py 서버 명령)

```bash
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:0 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-overlap-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

- `--enable-auto-tool-choice` : 모델이 도구를 쓸지 스스로 고르게 허용
- `--tool-call-parser hermes` : 모델이 만든 도구 호출 문장을 OpenAI 형식 `tool_calls` 로 바꾸는 파서 (`--enable-auto-tool-choice` 에 필수)
- `--enable-overlap-scheduling` 은 Day51 에서 비교한 overlap 스케줄링

### 4. @tool 도구 정의 · bind_tools · tool_calls (f31 · f32)

```python
from langchain_core.tools import tool

@tool
def add(aaa: int, bbb: int) -> int:
    """두 정수를 더합니다."""
    return aaa + bbb

print(add.name)          # add
print(add.description)   # 두 정수를 더합니다.
print(add.args)          # 매개변수 형식 (aaa · bbb : integer)

llm_with_tools = llm.bind_tools([add])
response = llm_with_tools.invoke("237341과 31321을 add로 더해줘.")

print(response.tool_calls)
# [{'name': 'add', 'args': {'aaa': 237341, 'bbb': 31321}, 'id': 'call-...', 'type': 'tool_call'}]

for call in response.tool_calls:      # f32
    print(call["name"], call["args"], call["id"])
```

- `@tool` : 함수를 도구(StructuredTool) 로 바꾼다 — 함수명 → name, docstring → description, 타입 힌트 → args
- 오늘 배운 데코레이터(@) 문법이 LangChain 도구 정의에 그대로 쓰인다
- `bind_tools` : 요청마다 도구 목록을 함께 보내는 LLM 을 만든다
- LLM 은 답을 계산하지 않고 "add 를 이 값으로 불러 달라" 는 `tool_calls` 만 돌려준다 (`finish_reason: 'tool_calls'`)
- f31 response_metadata : completion_tokens 126 / prompt_tokens 177 / cached_tokens 176 (도구 설명이 든 앞부분 Prefix Cache 적중)

### 5. tool_calls 로 도구 실행 (f33)

```python
response = llm_with_tools.invoke("21과 32를 add로 더해줘.")

for call in response.tool_calls:
    result = add.invoke(call["args"])     # add(aaa=21, bbb=32)
    print("계산 결과: ", result)          # 53
```

- 역할 분담 : LLM = 어떤 도구를 어떤 값으로 부를지 결정 / 우리 코드 = 도구를 실제로 실행
- `도구.invoke(dict)` 로 args 딕셔너리를 그대로 넣어 실행

### 6. 도구 여러 개 중 선택 (f34)

```python
tools = [add, multiply, subtract]
tool_map = {tool.name : tool for tool in tools}    # {'add': add, 'multiply': ..., 'subtract': ...}

llm_with_tools = llm.bind_tools(tools)

for question in questions:
    response = llm_with_tools.invoke(question)
    for call in response.tool_calls:
        result = tool_map[call["name"]].invoke(call["args"])
```

| 질문 | 선택한 도구 | 매개 변수 | 실행 결과 |
|---|---|---|---|
| 15와 27을 더해줘 | add | {'a': 15, 'b': 27} | 42 |
| 12와 8을 곱해줘 | multiply | {'a': 12, 'b': 8} | 96 |
| 200에서 150을 빼줘 | subtract | {'a': 200, 'b': 150} | 50 |

- 도구 이름을 콕 집어 말하지 않아도 (f31 은 "add로") 질문 뜻에 맞는 도구를 LLM 이 골랐다
- `tool_map` : LLM 이 돌려준 이름 문자열로 실행할 도구 객체를 찾는 딕셔너리

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `python/p03_Decorator.py` | `deco(func)` · `wrapper` · `@deco` 로 함수 실행 전 · 후 출력 |
| `python/p04_Deco_timer.py` | `time.perf_counter` 로 실행 시간을 재는 `@timer` 를 hello · study 에 적용 |
| `furiosa02_rngd/f31_tool_test.py` | `@tool` add 를 `bind_tools` 로 등록, tool_calls · response_metadata 확인 |
| `furiosa02_rngd/f32_tool_calls.py` | tool_calls 를 for 문으로 돌며 이름 · 매개변수 · 호출 ID 출력 (f31 베이스) |
| `furiosa02_rngd/f33_tool_invoke.py` | tool_calls 의 args 로 `add.invoke` 실제 실행 → 53 (f31 베이스) |
| `furiosa02_rngd/f34_multi_tools.py` | add · multiply · subtract 3 개 등록, `tool_map` 으로 LLM 이 고른 도구 실행 |
| `furiosa02_rngd/kingkeras.py` | 서버 실행 명령에 `--enable-overlap-scheduling` · `--enable-auto-tool-choice` · `--tool-call-parser hermes` 추가 (수정) |
| `python/p02_lambda3.py` | Day48 의 `p03_lambda3.py` 를 lambda 실습 번호(p02) 에 맞춰 이름 변경 (내용 같음) |
| `docs/Day54.md` | Day54 학습 기록 신규 작성 |
| `docs/Day53.md` | 하단 nav에 Day54 링크 추가 (수정) |
| `docs/Day48.md` | `p03_lambda3.py` → `p02_lambda3.py` 이름 변경 반영 (수정) |
| `README.md` | 학습 일지(퓨리오사 NPU 구분) / 디렉토리 구조(python/ · furiosa02_rngd/ Day54, p02_lambda3) / 진행도(54일, 68%) 갱신 (수정) |

---

## 💻 핵심 개념

### 데코레이터

```
def deco(func):          : 함수를 인자로 받는다
    def wrapper():       : 추가 기능 + func() 실행
        ...
    return wrapper       : 꾸민 함수(wrapper) 를 돌려준다

@deco                    : hello2 = deco(hello2)
def hello2(): ...
```

### 실행 시간 측정

```
start = time.perf_counter()
func()
end = time.perf_counter()
end - start              : 걸린 시간 (초)
time.sleep(1)            : 1 초 멈춤
```

### Tool Calling 흐름

```
@tool 함수 ──▶ bind_tools([...]) ──▶ invoke(질문)
                                       │
                                       ▼
              LLM : tool_calls [{'name', 'args', 'id'}]  (계산은 안 함)
                                       │
                                       ▼
              우리 코드 : tool_map[name].invoke(args) ──▶ 실행 결과
서버 옵션 : --enable-auto-tool-choice --tool-call-parser hermes
```

---

## 성과

- 함수를 인자로 받아 wrapper 를 돌려주는 데코레이터를 직접 만들고 `@` 문법으로 적용했다
- 실행 시간 측정 데코레이터를 만들어 여러 함수에 코드 수정 없이 공통 기능을 붙였다
- RNGD 서버에서 `@tool` · `bind_tools` 로 Tool Calling 을 실행해, LLM 이 고른 도구와 매개변수로 함수를 실제 실행했다 (도구 3 개 중 선택까지)

---

## 💡 주요 학습 포인트

1. **Decorator**: 기존 함수를 수정하지 않고 기능을 추가하는 문법
2. **함수를 인자로**: `deco(func)` 처럼 함수를 값으로 넘기고, 안쪽 함수(wrapper) 를 돌려준다
3. **@ 문법**: `@deco` = `hello2 = deco(hello2)`
4. **timer**: `time.perf_counter` 앞뒤 차이로 실행 시간 측정
5. **활용**: 시간 측정 · 로그 · 권한 검사 · 예외 처리 · 호출 횟수 기록을 여러 함수에 공통 적용
6. **@tool**: 데코레이터로 함수를 LLM 도구로 — 함수명 · docstring · 타입 힌트가 도구 정보가 된다
7. **tool_calls**: LLM 은 도구 이름 · 매개변수만 정하고, 실행은 `도구.invoke(args)` 로 우리 코드가 한다
8. **tool_map**: 도구가 여러 개면 이름 → 도구 dict 로 찾아 실행
9. **서버 옵션**: `--enable-auto-tool-choice` + `--tool-call-parser hermes`

---

[⬅️ Day53](Day53.md) · [🏠 전체 목차](../README.md) · [Day56 ➡️](Day56.md)
