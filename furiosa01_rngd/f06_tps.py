# [목적] 한 번 답변하는 데 걸린 시간을 재고, 초당 토큰 수(TPS, tokens per second)를 계산 (Day45)
# [흐름] perf_counter 로 invoke 앞뒤 시간 기록 → 걸린 시간 → max_tokens / 걸린 시간
# ※ 출력 토큰 수를 실제 값이 아니라 max_tokens(1000) 로 가정한다 → 답이 일찍 끝나면 TPS 가 실제보다 크게 나온다
#   (실제 출력 토큰 수로 계산하는 방법은 furiosa02_rngd/f17_tps.py)

from langchain_openai import ChatOpenAI
import time

# 최대 출력 토큰 수 (TPS 계산에도 이 값을 쓴다)
max_tokens = 1000

llm = ChatOpenAI(                            # Request, Response 역할
    base_url = "http://127.0.0.1:8000/v1",
    api_key = "EMPTY",
    max_tokens = max_tokens,
    # 0 → 가장 확률 높은 단어만 선택
    temperature = 0,
    # 누적 확률 상위 90% 후보에서 선택
    top_p = 0.9,
)

# time.perf_counter() : 시간 차이를 재는 정밀한 시계 (초 단위 실수)
start = time.perf_counter()
response = llm.invoke("NPU와 GPU의 차이점을 설명해줘")
end = time.perf_counter()

# 끝 - 시작 = 걸린 초, 소수 둘째 자리까지
elapsed = round(end - start, 2)

# print(response.content)
print("추론 시간 : ", elapsed, "초")
print("출력 토큰 : ", max_tokens)
# TPS = 토큰 수 / 걸린 시간
print("평균 처리량 : ", round(max_tokens/elapsed, 2), "tokens/s")