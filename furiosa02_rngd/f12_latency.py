# [목적] 같은 질문을 5 번 보내 매번의 응답 시간(latency)과 평균을 측정 (Day47)
# [흐름] for 5 회 : perf_counter 로 invoke 앞뒤 시간 → 리스트에 저장 → 합 / 개수 = 평균
# ※ 맨 아래 주석은 실제 실행 결과 (max_tokens=500 → 매번 약 5.4 초로 거의 일정)

from kingkeras import get_llm
import time

llm = get_llm(max_tokens=500)
prompt = "NPU의 특징을 말해줘."

# 회차별 응답 시간을 모을 리스트
times = []

# 같은 요청을 5 번 반복 → 한 번만 재면 생기는 우연한 차이를 줄인다
for i in range(5):
    # 요청 직전 시간
    start = time.perf_counter()
    response = llm.invoke(prompt)
    # 답변을 다 받은 직후 시간
    end = time.perf_counter()

    # 걸린 초 (변수 이름 elasped 는 elapsed 의 오타지만 코드 그대로 둔다)
    elasped = end - start
    # 리스트에 회차 시간 추가
    times.append(elasped)
    # i 는 0 부터라 +1 해서 1 회 ~ 5 회로 출력
    print(i + 1, "회: ", round(elasped, 2), "초")

# 평균 응답 시간 = 전체 합 / 횟수
average = sum(times) / len(times)
print("평균: ", round(average, 2), "초")

# 실행 결과
# 1 회:  5.37 초
# 2 회:  5.4 초
# 3 회:  5.35 초
# 4 회:  5.42 초
# 5 회:  5.36 초
# 평균:  5.38 초