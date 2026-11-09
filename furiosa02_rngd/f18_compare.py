# [목적] f15 의 3 개 프롬프트로 입력 토큰 · 출력 토큰 · 응답 시간을 함께 비교 (Day47)
# [흐름] 프롬프트마다 invoke 시간 측정 + usage_metadata 의 input_tokens · output_tokens 출력
# ※ 결과를 보면 입력은 12 ~ 28 토큰으로 비슷하고, 출력 토큰 수(888 → 1353 → 2000)에 따라 응답 시간(9.67 → 14.68 → 21.81 초)이 늘어난다
#   → 응답 시간은 출력 길이(decode) 에 거의 비례 (초당 약 92 토큰)

from kingkeras import get_llm
import time

# 최대 2000 토큰 (뒤의 쉼표는 문법상 문제 없음)
llm = get_llm(max_tokens=2000, )

prompts = [
    "NPU란?",
    "NPU와 GPU의 차이를 5가지 설명해줘.",
    "NPU의 구조와 장점, 활용 분야를 자세히 설명해줘."
]

for prompt in prompts:
    start = time.perf_counter()
    response = llm.invoke(prompt)
    # 걸린 초 (elasped 는 elapsed 의 오타, 코드 그대로 둔다)
    elasped = time.perf_counter() - start

    # 질문 토큰 수
    input_tokens = response.usage_metadata["input_tokens"]
    # 답변 토큰 수
    output_tokens = response.usage_metadata['output_tokens']

    print("질문: ", prompt)
    print("입력 토큰: ", input_tokens)
    print("출력 토큰: ", output_tokens)
    print("응답 시간: ", round(elasped, 2), "초")
    print("==============================")

    # 실행 결과 — 마지막 질문은 출력 2000 = max_tokens 에 걸려 답이 잘렸다
    # 질문:  NPU란?
    # 입력 토큰:  12
    # 출력 토큰:  888
    # 응답 시간:  9.67 초
    # ==============================
    # 질문:  NPU와 GPU의 차이를 5가지 설명해줘.
    # 입력 토큰:  24
    # 출력 토큰:  1353
    # 응답 시간:  14.68 초
    # ==============================
    # 질문:  NPU의 구조와 장점, 활용 분야를 자세히 설명해줘.
    # 입력 토큰:  28
    # 출력 토큰:  2000
    # 응답 시간:  21.81 초
    # ==============================