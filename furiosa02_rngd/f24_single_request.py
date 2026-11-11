# [목적] 요청 1 개만 보내 지연 시간(latency) 과 입력 / 출력 토큰 수를 측정 → f25 (순차) · f26 (동시) 비교의 기준 (Day49)
# [흐름] get_llm(max_tokens=100) → invoke 앞뒤 perf_counter → usage_metadata 로 토큰 수 출력

import time
from kingkeras import get_llm

llm = get_llm(max_tokens=100)

start = time.perf_counter()
response = llm.invoke("NPU의 장점을 세 가지 설명해줘.")
elapsed = time.perf_counter() - start

# usage_metadata 가 None 이면 빈 dict → .get 으로 기본값 0
usage = response.usage_metadata or {}

# response.content = 답변 내용
print("응답 내용: ", response.content)
print("지연 시간: ", round(elapsed, 3), "초")
print("입력 토큰: ", usage.get("input_tokens", 0))
print("출력 토큰: ", usage.get("output_tokens", 0))

# 지연 시간:  1.129 초
# 입력 토큰:  19
# 출력 토큰:  100
