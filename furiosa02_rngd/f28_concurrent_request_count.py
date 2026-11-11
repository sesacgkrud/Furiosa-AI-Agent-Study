# f27_rps.py 베이스
# 동시 요청 수별 성능 비교
# [목적] 같은 질문 16 개를 동시 요청 수(workers) 1 · 2 · 3 · 4 로 바꿔 보내며 전체 시간 · 평균 지연 · RPS · Output TPS 비교 (Day49)
# [흐름] ask 가 (요청별 지연 시간, 출력 토큰 수) 튜플을 돌려준다 → workers 마다 전체 시간 · 평균 지연 · req/s · TPS 출력
# ※ 동시 요청 수를 늘리면 평균 지연은 1.08 → 1.21 초로 조금만 늘고, 처리량(req/s · TPS) 은 약 3.6 배

import time
import statistics
from kingkeras import get_llm
from concurrent.futures import ThreadPoolExecutor

llm = get_llm(max_tokens=100)

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘."
    for i in range(16)
]

# 질문 1 개 → (이 요청의 지연 시간, 출력 토큰 수)
def ask(prompt):
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata or {}
    tokens = usage.get("output_tokens", 0)

    return elapsed, tokens

for workers in [1, 2, 3, 4]:

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(ask, prompts))
    total_time = time.perf_counter() - start

    # results = [(지연, 토큰), ...] → r[0] 지연 시간 리스트, r[1] 토큰 합
    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    print(f"\n===== Workers: {workers} =====")

    print("전체 시간: ", round(total_time, 2), "초")
    print("평균 지연: ", round(statistics.mean(latencies), 2), "초")
    print("요청 처리량: ", round(len(results) / total_time, 2), "req/s")
    print("Output TPS: ", round(total_tokens / total_time, 2))

print()

# ===== Workers: 1 =====
# 전체 시간:  17.36 초
# 평균 지연:  1.08 초
# 요청 처리량:  0.92 req/s
# Output TPS:  92.17

# ===== Workers: 2 =====
# 전체 시간:  9.3 초
# 평균 지연:  1.16 초
# 요청 처리량:  1.72 req/s
# Output TPS:  172.09

# ===== Workers: 3 =====
# 전체 시간:  7.11 초
# 평균 지연:  1.2 초
# 요청 처리량:  2.25 req/s
# Output TPS:  224.92

# ===== Workers: 4 =====
# 전체 시간:  4.86 초
# 평균 지연:  1.21 초
# 요청 처리량:  3.29 req/s
# Output TPS:  329.02

# → Workers 3 은 16 개가 3 개씩 나눠떨어지지 않아(3 × 5 + 1) 마지막 1 개를 혼자 처리하는 시간이 더해져 처리량이 덜 늘었다
