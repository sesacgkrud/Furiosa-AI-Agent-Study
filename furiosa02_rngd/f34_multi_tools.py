# f34_multi_tools.py
# 도구 여러 개 중 LLM 이 골라 쓰기
# [목적] add · multiply · subtract 도구 3 개를 bind_tools 로 한 번에 등록하고, 질문(더하기 · 곱하기 · 빼기) 마다
#        LLM 이 알맞은 도구와 매개변수를 고르면 tool_map 으로 그 도구를 찾아 실행 (Day54)
# [흐름] @tool 3 개 → tools 리스트 → tool_map {이름: 도구} → bind_tools(tools)
#        → 질문 3 개 for 문 : invoke → tool_calls 의 name · args 출력 → tool_map[name].invoke(args) 실행 결과 출력
# ※ f33 은 도구가 add 하나라 add.invoke 로 바로 실행했지만, 도구가 여러 개면 LLM 이 고른 이름으로 찾아야 한다 → tool_map
# ※ docstring 앞뒤 공백(""" 두 정수를 더합니다. """) 은 description 에서 지워진다 → '두 정수를 더합니다.'

from kingkeras import get_llm
from langchain_core.tools import tool

#1. 함수 정의 (@tool 로 도구 3 개)
@tool
def add(a: int, b: int) -> int:
    """ 두 정수를 더합니다. """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """ 두 정수를 곱합니다. """
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    """ 두 정수를 뺍니다. """
    return a - b

#2. 도구 목록
# tool_map : 도구 이름(str) → 도구 객체 dict (딕셔너리 컴프리헨션)
#            LLM 이 돌려준 call["name"] 문자열로 실행할 도구를 바로 찾기 위해 만든다
tools = [add, multiply, subtract]
tool_map = {tool.name : tool for tool in tools}

# print("=========================")
# print(tool_map)
# ※ 아래는 구조를 알아보기 쉽게 정리한 모양 (실제 출력은 args_schema=<class '...add'>, func=<function add at 0x...> 형태)
# tools = {
#     "add": StructuredTool(
#         name="add",
#         description="두 정수를 더합니다.",
#         args_schema=add,
#         func=add
#     ),

#     "multiply": StructuredTool(
#         name="multiply",
#         description="두 정수를 곱합니다.",
#         args_schema=multiply,
#         func=multiply
#     ),

#     "subtract": StructuredTool(
#         name="subtract",
#         description="두 정수를 뺍니다.",
#         args_schema=subtract,
#         func=subtract
#     )
# }

# print("=========================")
# print(tool_map["add"])
# name="add",
# description="두 정수를 더합니다.",
# args_schema=add,
# func=add

#3. 도구 연결 : 도구 3 개를 한 번에 등록 → LLM 이 질문마다 알맞은 도구를 고른다
llm = get_llm(max_tokens=300)
llm_with_tools = llm.bind_tools(tools)

#4. 프롬프트
questions = [
    "15와 27을 더해줘",
    "12와 8을 곱해줘",
    "200에서 150을 빼줘",
]

# Tool Calling + 도구 실행
for question in questions:
    response = llm_with_tools.invoke(question)

    print("\n질문: ", question)

    for call in response.tool_calls:
        print("선택한 도구: ", call["name"])    # 선택된 툴(함수)
        print("매개 변수: ", call['args'])      # LLM 이 질문에서 뽑은 매개변수 값

        result = tool_map[call["name"]].invoke(call["args"])   # 이름으로 도구를 찾아 실제 실행
        print("실행 결과: ", result)

        # 질문:  15와 27을 더해줘
        # 선택한 도구:  add
        # 매개 변수:  {'a': 15, 'b': 27}
        # 실행 결과:  42

        # 질문:  12와 8을 곱해줘
        # 선택한 도구:  multiply
        # 매개 변수:  {'a': 12, 'b': 8}
        # 실행 결과:  96

        # 질문:  200에서 150을 빼줘
        # 선택한 도구:  subtract
        # 매개 변수:  {'a': 200, 'b': 150}
        # 실행 결과:  50