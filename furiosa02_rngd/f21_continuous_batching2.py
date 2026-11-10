# f21_continuous_batching1.py 베이스
# [목적] pool.map 에 ask 함수 대신 lambda 를 넘기고 max_tokens 를 50 으로 줄여 동시 요청 비교 (Day48)
# [흐름] f21_continuous_batching1 과 같음 (동시 요청 수 1 · 2 · 4 · 8 → 전체 시간 · 처리량 · 평균 지연)
# ※ ask 함수는 정의만 남아 있고 이 파일에서는 쓰지 않는다 (pool.map 은 아래 lambda 를 쓴다)

import time
import statistics
from concurrent.futures import ThreadPoolExecutor

from kingkeras import run_once

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 자세히 설명해줘."
    for i in range(8)
]

def ask(prompt):
    return run_once(prompt, 100)

for workers in [1,2,4,8]:
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                # lambda 매개변수 : 이름 없는 한 줄 함수 → 질문 1 개를 받아 run_once(질문, max_tokens=50)
                # 매개변수 이름이 prompts 지만 리스트가 아니라 리스트에서 하나씩 꺼낸 질문 1 개가 들어온다
                lambda prompts: run_once(prompts, 50),
                prompts
            )
        )

    elasped = time.perf_counter() - start

    total_tokens = sum(
        r["output_tokens"] for r in results

    )

    avg_latency = statistics.mean(
        r["elapsed"] for r in results
    )

    print(f"\n동시 요청 수: {workers}")
    print(f"전체 시간: {elasped:.2f}초")
    print(f"전체 출력 토큰: {total_tokens}")
    print(f"처리량: {total_tokens / elasped:.2f} tokens/s")
    print(f"평균 요청 지연: {avg_latency:.2f}초")

    # 동시 요청 수: 1
    # 전체 시간: 4.67초
    # 전체 출력 토큰: 400
    # 처리량: 85.56 tokens/s
    # 평균 요청 지연: 0.55초

    # 동시 요청 수: 2
    # 전체 시간: 2.41초
    # 전체 출력 토큰: 400
    # 처리량: 165.85 tokens/s
    # 평균 요청 지연: 0.60초

    # 동시 요청 수: 4
    # 전체 시간: 1.31초
    # 전체 출력 토큰: 400
    # 처리량: 304.73 tokens/s
    # 평균 요청 지연: 0.65초

    # 동시 요청 수: 8
    # 전체 시간: 0.71초
    # 전체 출력 토큰: 400
    # 처리량: 563.98 tokens/s
    # 평균 요청 지연: 0.70초

    # → max_tokens 를 100 → 50 으로 줄이니 출력 토큰 합 800 → 400, 시간 · 평균 지연도 약 절반
    #   처리량(초당 토큰 수) 은 max_tokens 100 일 때와 비슷하다
