# f31_tool_test.py
# LangChain @tool + bind_tools 로 RNGD 서버(Qwen3-8B-FP8) 에 Tool Calling 시험
# [목적] 덧셈 함수 add 를 @tool 로 도구로 만들고, bind_tools 로 LLM 에 등록한 뒤 질문하면
#        LLM 이 답 대신 어떤 도구를 어떤 값으로 부를지(tool_calls) 를 돌려주는지 확인 (Day54)
# [흐름] @tool 정의 → name · description · args 확인 → get_llm → bind_tools([add]) → invoke
#        → content · tool_calls · response_metadata 출력
# [서버] Tool Calling 을 쓰려면 서버를 아래 옵션으로 띄운다 (kingkeras.py 서버 명령 참고)
#        --enable-auto-tool-choice  : 모델이 도구를 쓸지 스스로 고르게 허용
#        --tool-call-parser hermes  : 모델이 만든 도구 호출 문장을 OpenAI 형식 tool_calls 로 바꾸는 파서 (Qwen3 는 hermes 형식)
# ※ finish_reason 이 'tool_calls' = 답변이 아니라 도구 호출 요청으로 끝났다는 뜻
# ※ 이 단계에서는 LLM 이 tool_calls 만 만들고 add 함수는 실행되지 않는다

from kingkeras import get_llm
from langchain_core.tools import tool

#1. 도구 정의
# @tool 데코레이터 : 평범한 함수를 LLM 에 알려 줄 수 있는 도구(StructuredTool) 로 바꾼다
#                   타입 힌트(int) → 매개변수 형식, docstring → 도구 설명 으로 쓰인다
@tool
def add(aaa: int, bbb: int) -> int:
    """두 정수를 더합니다."""
    return aaa + bbb

print(add.name)         # 도구 이름 = 함수명 → add
print(add.description)  # 도구 설명 = docstring → 두 정수를 더합니다.
print(add.args)         # 매개변수 형식 → {'aaa': {'title': 'Aaa', 'type': 'integer'}, 'bbb': {...}}

#2. LLM 생성
llm = get_llm(max_tokens=200)

#3. 도구 등록 : 요청마다 도구 목록(이름 · 설명 · 매개변수 형식) 을 함께 보내는 LLM 을 만든다
llm_with_tools = llm.bind_tools([add])

#4. 질문
response = llm_with_tools.invoke(
            "237341과 31321을 add로 더해줘."
)

#5. 결과 확인
# 도구를 고르면 LLM 은 답을 직접 계산하지 않고 "add 를 이 값으로 불러 달라" 는 tool_calls 만 돌려준다
# → content 는 비어 있고(또는 짧은 문장), 실제 계산(함수 실행) 은 아직 하지 않은 상태 (f33 에서 실행)
print("일반 응답: ", response.content)
print("==============================")
print("Tool Calls:", response.tool_calls)
# Tool Calls: [{
#               'name': 'add',
#               'args': {'aaa': 237341, 'bbb': 31321},
#               'id': 'call-92cf06d6a19a4340be03a9d1453991e5', 'type': 'tool_call'
#               }
#             ]

print("메타 데이터: ", response.response_metadata)   # 토큰 사용량 · 모델 이름 · finish_reason
# 메타 데이터:
# {
#     'token_usage': {
#         'completion_tokens': 126,
#         'prompt_tokens': 177,
#         'total_tokens': 303,
#         'completion_tokens_details': {
#             'accepted_prediction_tokens': 0,
#             'audio_tokens': 0,
#             'reasoning_tokens': 0,
#             'rejected_prediction_tokens': 0
#         },
#         'prompt_tokens_details': {
#             'audio_tokens': 0,
#             'cache_write_tokens': None,
#             'cached_tokens': 176
#         }
#     },
#     'model_provider': 'openai',
#     'model_name': 'furiosa-ai/Qwen3-8B-FP8',
#     'system_fingerprint': None,
#     'id': 'chat-d9fb69089d594f008cd145c3ba4a1619',
#     'finish_reason': 'tool_calls',
#     'logprobs': None
# }
# → completion_tokens 126 : tool_calls 를 만드는 데 쓴 출력 토큰
# → cached_tokens 176 : 도구 설명이 들어간 같은 앞부분(prefix) 이 Prefix Cache 에 적중
