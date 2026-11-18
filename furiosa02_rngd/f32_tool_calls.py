# f31_tool_test.py 베이스
# tool_calls 꺼내 보기
# [목적] f31 의 response.tool_calls (딕셔너리 리스트) 를 for 문으로 돌며 도구 이름 · 매개변수 · 호출 ID 를 하나씩 출력 (Day54)
# [흐름] @tool add → bind_tools → invoke → for call in response.tool_calls → call["name"] · call["args"] · call["id"]
# ※ tool_calls 는 리스트 : 한 질문에 도구를 여러 번 부를 수도 있어서 for 문으로 꺼낸다
# ※ 호출 ID 는 요청마다 새로 만들어진다 (f31 과 값이 다름)

from kingkeras import get_llm
from langchain_core.tools import tool

#1. 도구 정의
@tool
def add(aaa: int, bbb: int) -> int:
    """두 정수를 더합니다."""
    return aaa + bbb

print(add.name)         # 함수명
print(add.description)  # add 함수의 주석
print(add.args)         # 매개변수

#2. LLM 생성
llm = get_llm(max_tokens=200)

#3. 도구 등록
llm_with_tools = llm.bind_tools([add])

#4. 질문
response = llm_with_tools.invoke(
            "237341과 31321을 add로 더해줘."
)

#5. 결과 확인 : call = {'name': ..., 'args': {...}, 'id': ..., 'type': 'tool_call'}
for call in response.tool_calls:
    print("tool 이름: ", call["name"])      # LLM 이 고른 도구 이름
    print("매개변수: ", call["args"])       # 질문에서 뽑은 값 (dict)
    print("호출 ID: ", call["id"])          # 이 도구 호출을 구분하는 ID

# tool 이름:  add
# 매개변수:  {'aaa': 237341, 'bbb': 31321}
# 호출 ID:  call-4879eee9a593430682dcc2340efe641a