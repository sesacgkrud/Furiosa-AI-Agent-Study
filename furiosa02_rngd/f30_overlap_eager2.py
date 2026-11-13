# f28_concurrent_request_count.py 베이스
# eager 모드 vs overlap 모드 (동시 요청)
# [목적] f30_overlap_eager1 의 순차 요청과 달리 질문 16 개를 동시 요청 수(workers) 1 · 2 · 3 · 4 로 보내고,
#        서버를 eager 모드(기본) · overlap 모드로 각각 띄워 전체 시간 · 평균 지연 · req/s · Output TPS 를 비교 (Day51)
# [흐름] ask 가 (요청별 지연 시간, 출력 토큰 수) 튜플을 돌려준다 → ThreadPoolExecutor 로 workers 개씩 동시에 보냄
#        → workers 마다 전체 시간 · 평균 지연 · req/s · TPS 출력 (아래 결과는 workers 4 만 기록)
# [서버] eager 모드   : 아래 serve 명령 그대로 (--enable-overlap-scheduling 기본값 False)
#        overlap 모드 : 같은 명령 끝에 --enable-overlap-scheduling 추가
#        overlap scheduling = 지금 배치를 NPU 가 계산하는 동안 CPU 가 다음 배치를 미리 스케줄링 → 배치 사이 CPU 대기 시간이 줄어든다
# ※ 동시 요청이면 continuous batching 으로 배치가 계속 이어져 overlap 효과가 보인다
#   workers 4 : 전체 시간 4.89 → 4.81 초, Output TPS 327.2 → 332.76 (약 1.7 % 향상), 평균 지연 1.22 → 1.2 초

# 서버 실행 명령 (overlap 모드는 끝에 --enable-overlap-scheduling 추가)
"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:0 \
    --host 0.0.0.0 \
    --port 8000
"""

import time
import statistics
from kingkeras import get_llm
from concurrent.futures import ThreadPoolExecutor

llm = get_llm(max_tokens=100)

# 번호만 다른 질문 16 개
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

# 동시 요청 수 1 ~ 4 로 바꿔 가며 16 개 질문을 모두 처리
for workers in [1, 2, 3, 4]:

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(ask, prompts))   # 질문 순서대로 (지연, 토큰) 16 개
    total_time = time.perf_counter() - start

    # results = [(지연, 토큰), ...] → r[0] 지연 시간 리스트, r[1] 토큰 합
    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    print(f"\n===== Workers: {workers} =====")

    print("전체 시간: ", round(total_time, 2), "초")
    print("평균 지연: ", round(statistics.mean(latencies), 2), "초")
    print("요청 처리량: ", round(len(results) / total_time, 2), "req/s")   # RPS
    print("Output TPS: ", round(total_tokens / total_time, 2))           # 초당 출력 토큰 수

print()

# ===== Workers: 4 (eager, 기본) =====
# 전체 시간:  4.89 초
# 평균 지연:  1.22 초
# 요청 처리량:  3.27 req/s
# Output TPS:  327.2

# ===== Workers: 4 (overlap, --enable-overlap-scheduling) =====
# 전체 시간:  4.81 초
# 평균 지연:  1.2 초
# 요청 처리량:  3.33 req/s
# Output TPS:  332.76

# → 동시 요청에서는 overlap 모드가 전체 시간 · 지연 · 처리량 모두 조금 낫다 (순차 요청 f30_overlap_eager1 에서는 차이 없음)
