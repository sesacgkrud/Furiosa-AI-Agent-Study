# f24_single_request.py 베이스
# [목적] 질문 8 개를 for 문으로 하나씩 차례대로(순차) 보내 전체 시간과 요청 처리량(req/s) 측정 (Day49)
# [흐름] 질문 8 개 → for 문으로 invoke (앞 요청이 끝나야 다음 요청) → 전체 시간 · 요청 수 / 전체 시간
# ※ 요청 처리량(req/s) = 1 초에 끝내는 요청 수 → 순차라 요청 1 개 지연(약 1.1 초) 의 역수에 가깝다

import time
from kingkeras import get_llm

llm = get_llm(max_tokens=100)

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘."
    for i in range(8)
]

start = time.perf_counter()
for prompt in prompts:
    llm.invoke(prompt)
elapsed = time.perf_counter() - start

print("전체 요청: ", len(prompts))
print("전체 소요 시간: ", round(elapsed, 3), "초")
print("요청 처리량", round(len(prompts) / elapsed, 3), "req/s")

# 전체 요청:  8
# 전체 소요 시간:  8.704 초
# 요청 처리량 0.919 req/s
