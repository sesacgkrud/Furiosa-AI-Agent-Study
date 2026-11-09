# [목적] f14_1 을 고쳐서 첫 내용 chunk 의 시간만 재고 바로 멈추는 TTFC 측정 (Day47)
# [흐름] llm.stream 반복 → content 가 있는 첫 chunk 에서 시간 기록 → break 로 반복 종료 (답변 출력 없음)
# ※ 첫 내용 chunk 에서 바로 break 하므로 'if first_time is None' 검사가 필요 없어 주석 처리했다
#   (주석 처리 후 남은 두 줄의 들여쓰기가 깊지만 if chunk.content 블록 안이라 정상 동작)

# time to first chunk : 사용자가 프롬프트를 입력한 후 첫 번째 토큰이 생성될 때까지 걸리는 시간

from kingkeras import get_llm
import time

llm = get_llm(max_tokens=500)
prompt = "NPU의 장점을 자세히 설명해줘."
first_time = None


start = time.perf_counter()
# 응답을 조각 단위로 받는다
for chunk in llm.stream(prompt):

    # 글자가 있는 첫 chunk
    if chunk.content:
        # if first_time is None:
            # 첫 내용 chunk 까지 걸린 초
            first_time = time.perf_counter() - start
            # 시간을 쟀으니 나머지 답변은 받지 않고 반복 종료
            break
        # print(chunk.content, end="", flush=True)

print("==============================")
print("첫 컨텐츠 응답 시간: ", round(first_time, 2), "초")

# ==============================
# 실행 결과
# 첫 컨텐츠 응답 시간:  0.03 초