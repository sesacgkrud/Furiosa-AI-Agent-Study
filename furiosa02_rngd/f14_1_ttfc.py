# [목적] TTFC(Time To First Chunk) 측정 — 요청 후 내용이 있는 첫 조각이 도착할 때까지 걸린 시간 (Day47)
#        응답 전체 시간과 달리 '사용자가 첫 글자를 보기까지 기다리는 시간' 이다
# [흐름] 시작 시간 기록 → llm.stream 반복 → content 가 있는 첫 chunk 에서만 시간 기록 → 나머지는 계속 출력
# ※ 아래 원래 주석의 '첫 번째 토큰' 은 정확히는 '내용이 있는 첫 chunk' (chunk 에 토큰이 여러 개 들어 있을 수도 있다)

# time to first chunk : 사용자가 프롬프트를 입력한 후 첫 번째 토큰이 생성될 때까지 걸리는 시간

from kingkeras import get_llm
import time

llm = get_llm(max_tokens=500)
prompt = "NPU의 장점을 자세히 설명해줘."
# 아직 첫 chunk 시간을 재지 않았다는 표시
first_time = None


# 요청 시작 시간
start = time.perf_counter()
for chunk in llm.stream(prompt):
    # 역할 정보만 있는 빈 chunk 는 건너뛰고 글자가 있는 chunk 만 본다
    if chunk.content:
        # None 일 때(= 첫 번째 내용 chunk) 한 번만 기록 → 이후 chunk 는 시간을 덮어쓰지 않는다
        if first_time is None:
            first_time = time.perf_counter() - start
        # 답변은 끝까지 스트리밍 출력
        print(chunk.content, end="", flush=True)

print("==============================")
# 첫 내용 chunk 까지 걸린 초
print("첫 컨텐츠 응답 시간: ", round(first_time, 2), "초")

# 실행 결과 (0.03 초 → 첫 글자는 거의 바로 나온다)
# ==============================
# 첫 컨텐츠 응답 시간:  0.03 초