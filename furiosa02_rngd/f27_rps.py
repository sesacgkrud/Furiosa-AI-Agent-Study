# f26_concurrent_request.py 베이스
# RPS : Requests Per Second (1 초에 처리한 요청 수)
# [목적] 질문을 16 개로 늘려 동시 4 개로 보내고, 출력 토큰 기준 처리량(Output TPS) 과 요청 기준 처리량(RPS) 을 함께 계산 (Day49)
# [흐름] ask 가 응답 대신 output_tokens 를 돌려준다 → 합계 / 전체 시간 = Output TPS, 질문 수 / 전체 시간 = RPS
# ※ Output TPS ≈ RPS × 요청당 출력 토큰 (3.28 × 100 = 328)

import time
from kingkeras import get_llm
from concurrent.futures import ThreadPoolExecutor

llm = get_llm(max_tokens=100)

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘."
    for i in range(16)
]

# 질문 1 개 → 그 답변의 출력 토큰 수
def ask(prompt):
    response = llm.invoke(prompt)
    usage = response.usage_metadata or {}
    return usage.get("output_tokens", 0)

start = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as executor:
    tokens = list(executor.map(ask, prompts))   # 질문별 출력 토큰 수 16 개

elapsed = time.perf_counter() - start

total_tokens = sum(tokens)

print("전체 출력 토큰: ", total_tokens)
print("전체 시간: ", round(elapsed, 2), "초")
print("Output TPS: ", round(total_tokens / elapsed, 2))   # 초당 출력 토큰 수
print("RPS: ", round(len(prompts) / elapsed, 2))          # 초당 처리한 요청 수

# 전체 출력 토큰:  1600
# 전체 시간:  4.88 초
# Output TPS:  328.2
# RPS:  3.28
