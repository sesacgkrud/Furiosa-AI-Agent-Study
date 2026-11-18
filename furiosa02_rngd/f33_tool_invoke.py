# f31_tool_test.py 베이스
# tool_calls 로 실제 도구 실행
# [목적] LLM 이 돌려준 tool_calls 의 args 를 add.invoke 에 넣어 실제로 함수를 실행하고 계산 결과를 얻는다 (Day54)
# [흐름] @tool add → bind_tools → invoke("21과 32를 add로 더해줘.") → for call in tool_calls → add.invoke(call["args"]) → 53
# ※ Tool Calling 의 역할 분담 : LLM = 어떤 도구를 어떤 값으로 부를지 결정 / 파이썬(우리 코드) = 도구를 실제로 실행

from kingkeras import get_llm
from langchain_core.tools import tool

#1. 도구 정의
@tool
def add(aaa: int, bbb: int) -> int:
    """두 정수를 더합니다."""
    return aaa + bbb

# print(add.name)         # 함수명
# print(add.description)  # add 함수의 주석
# print(add.args)         # 매개변수

#2. LLM 생성
llm = get_llm(max_tokens=200)

#3. 도구 등록
llm_with_tools = llm.bind_tools([add])

#4. 질문
response = llm_with_tools.invoke(
            "21과 32를 add로 더해줘."
)

# call["args"] = {'aaa': 21, 'bbb': 32} → add.invoke 에 dict 그대로 넣으면 add(aaa=21, bbb=32) 실행
for call in response.tool_calls:
    result = add.invoke(call["args"])
    print("계산 결과: ", result)        # 계산 결과:  53