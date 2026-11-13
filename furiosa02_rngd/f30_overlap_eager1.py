# f25_sequential_request.py 베이스
# eager 모드 vs overlap 모드 (순차 요청)
# [목적] 같은 질문 8 개를 for 문으로 하나씩(순차) 보내고, 서버를 eager 모드(기본) · overlap 모드로 각각 띄워 전체 시간 · RPS · TPS 를 비교 (Day51)
# [흐름] 질문 8 개 → for 문으로 invoke (앞 요청이 끝나야 다음 요청) → usage_metadata 의 output_tokens 누적
#        → 전체 요청 수 · 출력 토큰 합 · 전체 시간 · RPS(요청 수 / 시간) · TPS(토큰 수 / 시간) 출력
# [서버] eager 모드   : 아래 serve 명령 그대로 (--enable-overlap-scheduling 기본값 False)
#        overlap 모드 : 같은 명령 끝에 --enable-overlap-scheduling 추가
#        overlap scheduling = 지금 배치를 NPU 가 계산하는 동안 CPU 가 다음 배치를 미리 스케줄링해 CPU 스케줄링 시간을 NPU 계산 뒤에 숨긴다
#        (전체 처리량 · TPOT 개선, 대신 TTFT 가 조금 늘 수 있음 — furiosa-llm serve --help)
# ※ 순차 요청은 배치에 요청이 1 개뿐이라 미리 스케줄링할 다음 배치가 거의 없다 → 두 모드 결과가 거의 같다 (8.747 초 vs 8.782 초)
# ※ max_tokens=100 이라 질문마다 100 토큰을 다 채워 출력 토큰 합 = 8 × 100 = 800

# 서버 실행 명령 (overlap 모드는 끝에 --enable-overlap-scheduling 추가)
"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:0 \
    --host 0.0.0.0 \
    --port 8000
"""

import time
from kingkeras import get_llm

llm = get_llm(max_tokens=100)

# 번호만 다른 질문 8 개
prompts= [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘."
    for i in range(8)
]

start = time.perf_counter()
total_tokens = 0                                # 8 개 답변의 출력 토큰 합

# 순차 요청 : 한 번에 1 개씩 보내고 답이 와야 다음 질문
for prompt in prompts:
    response = llm.invoke(prompt)
    
    usage = response.usage_metadata or {}       # 토큰 사용량 (없으면 빈 dict)
    tokens = usage.get("output_tokens")         # 이 답변의 출력 토큰 수
                                                # ※ 기본값이 없어 usage 가 비면 None → 아래 += 에서 TypeError (f28 처럼 get("output_tokens", 0) 이면 안전)
    
    total_tokens += tokens

elapsed = time.perf_counter() - start
print("전체 요청: ", len(prompts))
print("전체 출력 토큰: ", total_tokens)
print("전체 소요 시간: ", round(elapsed, 3), "초")
print("RPS: ", round(len(prompts) / elapsed, 3))         # 1 초에 끝낸 요청 수
print("TPS: ", round(total_tokens / elapsed, 3))         # 1 초에 만든 출력 토큰 수 (≈ RPS × 100)

########## eager 모드 (기본, overlap scheduling 끔) ##########
# 전체 요청:  8
# 전체 출력 토큰:  800
# 전체 소요 시간:  8.747 초
# RPS:  0.915
# TPS:  91.458

######### overlap 모드 (--enable-overlap-scheduling) #########
# 전체 요청:  8
# 전체 출력 토큰:  800
# 전체 소요 시간:  8.782 초
# RPS:  0.911
# TPS:  91.095

# → 순차 요청에서는 두 모드 차이가 0.4 % 안쪽 (측정 오차 수준)
