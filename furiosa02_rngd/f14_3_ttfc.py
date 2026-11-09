# [목적] for 문 없이 next(iter(...)) 로 스트림의 첫 chunk 하나만 꺼내 TTFC 측정 (Day47)
# [흐름] iter(llm.stream(prompt)) → next() 로 첫 chunk → 그때까지 걸린 시간 출력
# ※ f14_1 · f14_2 와 달리 content 검사가 없어서, 첫 chunk 가 내용 없는 chunk(역할 정보 등)여도 그 시간을 잰다
#   → '첫 글자' 시간이 아니라 '첫 응답 조각' 시간

# time to first chunk : 사용자가 프롬프트를 입력한 후 첫 번째 토큰이 생성될 때까지 걸리는 시간

from kingkeras import get_llm
import time

llm = get_llm(max_tokens=500)
prompt = "NPU의 장점을 자세히 설명해줘."

start = time.perf_counter()

# iter() : 반복 가능한 객체 → 반복자, next() : 반복자에서 다음(여기서는 첫 번째) 값 하나만 꺼낸다
chunk = next(iter(llm.stream(prompt)))

# 첫 chunk 를 받기까지 걸린 초
first_time = time.perf_counter() - start

print("==============================")
print("첫 컨텐츠 응답 시간: ", round(first_time, 2), "초")

# 실행 결과
# ==============================
# 첫 컨텐츠 응답 시간:  0.03 초