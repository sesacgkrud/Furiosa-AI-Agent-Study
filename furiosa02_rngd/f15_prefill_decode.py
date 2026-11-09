# [목적] 질문 길이 · 요구 분량이 다른 3 개 프롬프트의 응답 시간을 비교 (Day47)
#        LLM 추론 = prefill(입력 프롬프트 전체를 한 번에 처리) + decode(출력 토큰을 하나씩 생성) 두 단계
# [흐름] prompts 리스트 반복 → invoke 앞뒤 시간 측정 → 질문 · 시간 출력
# ※ 세 질문의 입력 길이 차이는 작아서(입력 12 ~ 28 토큰, f18 결과) 시간 차이는 대부분 출력 길이(decode) 에서 생긴다
#   → '자세히' 를 요구한 질문일수록 답이 길어 오래 걸린다

from kingkeras import get_llm
import time

# 최대 2000 토큰까지 답변 허용 (길게 답할 여지를 준다)
llm = get_llm(max_tokens=2000)

# 짧은 질문 → 긴 질문 순서
prompts = [
    "NPU란?",
    "NPU와 GPU의 차이를 5가지 설명해줘.",
    "NPU의 구조와 장점, 활용 분야를 자세히 설명해줘."
]

# 프롬프트마다 한 번씩 요청
for prompt in prompts:
    start = time.perf_counter()
    response = llm.invoke(prompt)
    # 걸린 초 (elasped 는 elapsed 의 오타, 코드 그대로 둔다)
    elasped = time.perf_counter() - start

    print("질문: ", prompt)
    print("시간: ", round(elasped, 2), "초")

# 실행 결과 — 요구 분량이 많을수록 시간이 길다
# 질문:  NPU란?
# 시간:  12.92 초
# 질문:  NPU와 GPU의 차이를 5가지 설명해줘.
# 시간:  14.73 초
# 질문:  NPU의 구조와 장점, 활용 분야를 자세히 설명해줘.
# 시간:  21.96 초