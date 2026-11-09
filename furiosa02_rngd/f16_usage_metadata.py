# [목적] 응답 객체의 usage_metadata 로 입력 · 출력 · 전체 토큰 수 확인 (Day47)
# [흐름] llm.invoke → response.content (답변) + response.usage_metadata (토큰 사용량 dict) 출력
# ※ 결과 dict 의 input_token_details.cache_read : 이전 요청과 같은 앞부분(프롬프트 캐시)을 다시 계산하지 않고 재사용한 토큰 수
#   output_token_details.reasoning : 추론(생각) 과정에 쓴 토큰 수

from kingkeras import get_llm   

llm = get_llm(max_tokens=500)

response = llm.invoke("NPU란 무엇인가?")

# 답변 텍스트
print(response.content)
print("==============================")
# 토큰 사용량 : input_tokens(질문) · output_tokens(답변) · total_tokens(합)
print(response.usage_metadata)

# 실행 결과 — output_tokens 500 = max_tokens 에 걸려 답이 중간에 잘렸다는 뜻
# {
    # 'input_tokens': 15,
    # 'output_tokens': 500,
    # 'total_tokens': 515,
    # 'input_token_details': {
    #   'audio': 0,
    #   'cache_read': 14
    # },
    # 'output_token_details': {
    #   'audio': 0,
    # 'reasoning': 0
    # }
# }