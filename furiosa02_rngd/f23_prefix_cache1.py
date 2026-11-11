# [목적] 앞부분(prefix) 이 같고 끝의 질문만 다른 요청을 4 번 보내, Prefix Cache 가 앞부분을 재사용하는지 확인 (Day49)
# [흐름] 같은 긴 문장 × 150 (prefix) + "질문 i" → run_once(max_tokens=100) → 응답 시간 · 입력 토큰 출력
#        → get_cache_metrics 로 서버의 Prefix Cache 조회 · 적중 횟수(누적) 출력
# ※ Prefix Cache : 1 회 요청에서 계산한 prefix 부분의 KV 를 저장해 두었다가, 2 회부터는 다시 계산하지 않고 재사용
#   → 2 회부터 적중 횟수가 prefix 토큰 수만큼 늘어난다
# ※ get_cache_metrics 값은 서버 시작 후 누적값 → 요청마다 늘어난 양은 f23_prefix_cache2 에서 전후 차이로 계산

from kingkeras import run_once, get_cache_metrics

# 모든 요청이 공유하는 긴 앞부분
prefix = "NPU 추론 최적화에 대한 설명입니다." * 150

# i = 1 ~ 4
for i in range(1, 5):
    # 앞부분은 같고, 줄바꿈 뒤 질문 번호만 다르다
    prompt = prefix + f"\n질문 {i}: KV Cache란?"
    print(f"\n===== {i}회 요청 =====")
    result = run_once(prompt, 100)

    print("응답 시간: ", round(result["elapsed"], 3))
    print("입력 토큰: ", result["input_tokens"])

    print(("\nPrefix Cache 메트릭"))
    get_cache_metrics()
