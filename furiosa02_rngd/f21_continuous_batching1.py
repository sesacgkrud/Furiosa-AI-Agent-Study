# [목적] 같은 개수(8 개)의 요청을 동시 요청 수(1 · 2 · 4 · 8) 를 바꿔 보내며 전체 시간 · 처리량 · 평균 지연 비교 (Day48)
#        furiosa-llm 서버의 continuous batching(연속 배칭) 효과 확인
# [흐름] 질문 8 개 → ThreadPoolExecutor(max_workers=동시 요청 수) 로 run_once 를 동시에 실행
#        → 전체 시간 · 출력 토큰 합 · 처리량(출력 토큰 / 전체 시간) · 요청별 평균 지연 출력
# ※ continuous batching : 서버가 동시에 들어온 요청들을 한 배치로 묶어 decode 를 함께 진행하고,
#   끝난 요청 자리에 새 요청을 바로 채워 넣는다 → 요청 1 개 지연은 거의 그대로, 전체 처리량은 늘어난다
# ※ 맨 아래 주석은 실제 실행 결과 (max_tokens=100)

import time
# statistics : 파이썬 기본 라이브러리, mean() = 평균
import statistics
# ThreadPoolExecutor : 스레드 여러 개로 함수를 동시에 실행 (요청을 보내고 기다리는 동안 다른 요청도 보낸다)
from concurrent.futures import ThreadPoolExecutor

from kingkeras import run_once

# 리스트 컴프리헨션 : i = 0 ~ 7 → 번호만 다른 질문 8 개
prompts = [
    f"NPU 추론 최적화 방법 {i}번을 자세히 설명해줘."
    for i in range(8)
]

# 질문 리스트 확인용 (변수 이름은 prompts)
# print(prompt)
# print(len(prompt))

# run_once 단독 실행 확인용
# aaa = run_once("NPU가 뭐게", 1000)
# print(aaa["text"])

# pool.map 에 넘길 함수 : 질문 1 개 → run_once(질문, max_tokens=100) 결과 dict
def ask(prompt):
    return run_once(prompt, 100)

# 동시 요청 수를 1 → 2 → 4 → 8 로 늘려 같은 8 개 질문을 처리
for workers in [1,2,4,8]:
    start = time.perf_counter()

    # max_workers=workers : 동시에 실행하는 스레드 수 = 서버에 동시에 들어가는 요청 수
    # pool.map(함수, 리스트) : 리스트의 각 질문에 ask 를 적용, 결과는 질문 순서대로 돌아온다
    # with 블록이 끝날 때 모든 요청이 끝날 때까지 기다린다
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(ask, prompts)
        )

    # 8 개 질문을 모두 처리하는 데 걸린 전체 시간
    elasped = time.perf_counter() - start

    # 8 개 답변의 출력 토큰 합 (max_tokens=100 에 모두 걸려 100 × 8 = 800)
    total_tokens = sum(
        r["output_tokens"] for r in results

    )
    # print(len(results))
    # print(total_tokens)

    # 요청 1 개가 걸린 시간(run_once 의 elapsed) 의 평균
    avg_latency = statistics.mean(
        r["elapsed"] for r in results
    )

    print(f"\n동시 요청 수: {workers}")
    print(f"전체 시간: {elasped:.2f}초")
    print(f"전체 출력 토큰: {total_tokens}")
    # 처리량(throughput) = 전체 출력 토큰 / 전체 시간 → 단위는 초당 토큰 수
    print(f"처리량: {total_tokens / elasped:.2f} tokens/s")
    print(f"평균 요청 지연: {avg_latency:.2f}초")

    # 동시 요청 수: 1
    # 전체 시간: 9.03초
    # 전체 출력 토큰: 800
    # 처리량: 88.61 tokens/s
    # 평균 요청 지연: 1.09초

    # 동시 요청 수: 2
    # 전체 시간: 4.63초
    # 전체 출력 토큰: 800
    # 처리량: 172.85 tokens/s
    # 평균 요청 지연: 1.15초

    # 동시 요청 수: 4
    # 전체 시간: 2.44초
    # 전체 출력 토큰: 800
    # 처리량: 327.30 tokens/s
    # 평균 요청 지연: 1.22초

    # 동시 요청 수: 8
    # 전체 시간: 1.31초
    # 전체 출력 토큰: 800
    # 처리량: 612.23 tokens/s
    # 평균 요청 지연: 1.30초

    # → 동시 요청 수를 2 배로 늘릴 때마다 전체 시간은 약 1/2, 처리량은 약 2 배
    #   요청 1 개의 평균 지연은 1.09 → 1.30 초로 조금만 늘어난다 (continuous batching)
