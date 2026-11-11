# [목적] 시작 시각 · 출력 길이(max_tokens) 가 다른 요청 5 개를 동시에 보내, 짧은 요청이 긴 요청을 기다리지 않고
#        먼저 끝나는지 확인 (continuous batching) (Day49)
# [흐름] ask(이름, 대기 초, max_tokens) : 대기 → 요청 시작 출력 → run_once → 완료 시간 출력
#        ThreadPoolExecutor(max_workers=4) 에 pool.submit 으로 A ~ E 5 개 작업 등록 → f.result() 로 결과 모으기
# ※ continuous batching : 서버는 요청을 한 배치로 함께 decode 하다가, 끝난 요청은 바로 내보내고 새 요청을 바로 끼워 넣는다
#   → 늦게 들어온 짧은 요청(B · D) 이 먼저 들어온 긴 요청(A · C) 보다 먼저 끝난다

import time
from concurrent.futures import ThreadPoolExecutor
from kingkeras import run_once

def ask(name, delay, max_tokens):
    # time.sleep(초) : 그만큼 멈춘다 → 요청마다 시작 시각을 delay 초씩 늦춘다
    time.sleep(delay)

    # flush=True : 버퍼에 모아 두지 않고 바로 화면에 출력 (여러 스레드의 출력 순서를 실제 시각대로 보기 위해)
    print(f"{name} 요청 시작", flush=True)

    start = time.perf_counter()
    result = run_once(
        f"NPU와 LLM 추론 최적화를 자세히 설명해줘. 요청 {name}",
        max_tokens
    )
    elapsed = time.perf_counter() - start
    print(f"{name} 요청 완료: {elapsed:.2f}초", flush=True)

    return result

# 방법1 : with 문 → 블록이 끝날 때 모든 작업이 끝날 때까지 기다리고 pool 을 자동으로 닫는다
with ThreadPoolExecutor(max_workers=4) as pool:
    # pool.submit(함수, 인자들...) : 작업 1 개를 등록하고 바로 Future(나중에 결과를 받을 객체) 를 돌려준다
    # pool.map 과 달리 작업마다 인자를 여러 개 · 다르게 줄 수 있다
    futures = [
        pool.submit(ask, "A", 0, 800),     # 0 초에 시작, 긴 출력
        pool.submit(ask, "B", 1, 200),     # 1 초 뒤, 짧은 출력
        pool.submit(ask, "C", 2, 1000),    # 2 초 뒤, 가장 긴 출력
        pool.submit(ask, "D", 3, 200),     # 3 초 뒤, 짧은 출력
        pool.submit(ask, "E", 4, 300),     # 스레드가 4 개라 E 는 앞 작업 하나가 끝나 스레드가 비어야 시작된다
    ]
    # f.result() : 그 작업이 끝날 때까지 기다렸다가 반환값(run_once 의 dict) 을 꺼낸다
    results = [f.result() for f in futures]

# 방법2 : with 없이 직접 만들고, 다 쓴 뒤 pool.shutdown() 으로 닫는다 (방법1 과 같은 동작)
# pool = ThreadPoolExecutor(max_workers=4)
# futures = [
#     pool.submit(ask, "A", 0, 800),
#     pool.submit(ask, "B", 1, 200),
#     pool.submit(ask, "C", 2, 1000),
#     pool.submit(ask, "D", 3, 200),
#     pool.submit(ask, "E", 4, 300),
# ]
# results = [f.result() for f in futures]
# pool.shutdown()

print("========== 끝 ==========")

# A 요청 시작
# B 요청 시작
# C 요청 시작
# D 요청 시작
# B 요청 완료: 2.39초
# D 요청 완료: 2.37초
# E 요청 시작
# A 요청 완료: 9.58초
# E 요청 완료: 3.48초
# C 요청 완료: 11.41초

# → 늦게 시작한 짧은 요청 B · D (200 토큰) 가 먼저 시작한 긴 요청 A (800) · C (1000) 보다 먼저 끝났다
#   (긴 요청이 끝날 때까지 기다리지 않는다 = continuous batching)
# → E 는 스레드 4 개가 A ~ D 로 모두 차 있어서, B 가 끝나 스레드가 빈 뒤에야 sleep(4) 를 시작했다
#   (time.sleep 이 스레드 안에 있어서 delay 는 프로그램 시작이 아니라 스레드를 받은 시점부터 센다)
