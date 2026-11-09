# [목적] usage_metadata 의 실제 출력 토큰 수로 TPS(초당 토큰 수) 계산 (Day47)
#        Day45 f06_tps 는 max_tokens 를 출력 토큰으로 가정했지만, 여기서는 실제로 만든 토큰 수를 쓴다
# [흐름] invoke 시간 측정 → output_tokens / 걸린 시간 = TPS
# ※ 걸린 시간에 prefill(첫 토큰까지) 시간도 포함되므로 순수 생성 속도보다 약간 낮게 나온다

from kingkeras import get_llm
import time
llm = get_llm(max_tokens=1000)

prompt = "NPU의 특징을 자세히 설명해줘."

start = time.perf_counter()
response = llm.invoke(prompt)
# 요청 ~ 답변 완료까지 걸린 초
elapsed = time.perf_counter() - start

# 답변으로 실제 생성된 토큰 수
tokens = response.usage_metadata["output_tokens"]
# 초당 생성 토큰 수
tps = tokens / elapsed

print("출력 토큰: ", tokens)
print("전체 시간: ", round(elapsed, 2), "초")
print("TPS: ", round(tps, 2))

# 실행 결과 (출력 1000 = max_tokens 에 걸림, 약 91 tokens/s)
# 출력 토큰:  1000
# 전체 시간:  10.95 초
# TPS:  91.34