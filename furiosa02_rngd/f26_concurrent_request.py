# f25_sequential_request.py 베이스
# [목적] 같은 질문 8 개를 ThreadPoolExecutor 로 4 개씩 동시에 보내 순차(f25) 와 전체 시간 · 요청 처리량 비교 (Day49)
# [흐름] ask(질문) = llm.invoke → executor.map 으로 4 개 스레드에서 동시 실행 → 전체 시간 · req/s
# ※ 결과 : 순차 8.704 초 · 0.919 req/s → 동시 4 개 2.494 초 · 3.208 req/s (약 3.5 배)

import time
from kingkeras import get_llm
from concurrent.futures import ThreadPoolExecutor

llm = get_llm(max_tokens=100)

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘."
    for i in range(8)
]

# executor.map 에 넘길 함수 : 질문 1 개 → 응답
def ask(prompt):
    return llm.invoke(prompt)

start = time.perf_counter()
# max_workers=4 : 서버에 동시에 들어가는 요청 4 개, 결과는 질문 순서대로 리스트로
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(ask, prompts))

elapsed = time.perf_counter() - start

print("전체 요청: ", len(prompts))
print("전체 소요 시간: ", round(elapsed, 3), "초")
print("요청 처리량", round(len(prompts) / elapsed, 3), "req/s")

# 전체 요청:  8
# 전체 소요 시간:  2.494 초
# 요청 처리량 3.208 req/s
