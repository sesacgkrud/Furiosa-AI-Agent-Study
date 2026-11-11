# f23_prefix_cache1.py 베이스
# [목적] 요청 전후로 get_cache_metrics 를 불러 차이를 계산 → 누적값이 아니라 '이번 요청' 의 조회 · 적중 횟수와 적중률을 출력 (Day49)
# [흐름] before = 요청 전 캐시 상태 → run_once → after = 요청 후 캐시 상태 → after - before = 이번 요청 증가량
# ※ 1 회는 prefix 가 아직 캐시에 없어 적중이 거의 없고, 2 회부터는 prefix 를 재사용해 적중률이 높아진다
#   (같은 prefix 를 이전 실행에서 이미 보낸 서버라면 1 회부터 적중할 수 있다)

from kingkeras import run_once, get_cache_metrics

prefix = "NPU 추론 최적화에 대한 설명입니다." * 150

for i in range(1, 5):
    print(f"\n===== {i}회 요청 시작 =====")

    prompt = prefix + f"\n질문 {i}: KV Cache란?"

    # get_cache_metrics 는 출력과 함께 {queries, hits, hit_rate, kv_usage} dict 를 돌려준다
    before = get_cache_metrics()    # 요청 전 캐시 상태
    result = run_once(prompt, 100)
    after = get_cache_metrics()     # 요청 후 캐시 상태

    # 이번 요청에서 증가한 횟수 (누적값의 차이)
    hits = after["hits"] - before["hits"]
    queries = after["queries"] - before["queries"]

    # 조회가 0 이면 0 으로 나누기 오류가 나지 않게 0
    if queries > 0:
        hit_rate = hits / queries * 100
    else:
        hit_rate = 0

    # print(hit_rate)
    print(f"\n ===== {i}회 요청 결과 =====")

    print("응답 시간: ", round(result["elapsed"], 3))
    print("입력 토큰: ", result["input_tokens"])
    # int() : 메트릭 값이 float(예: 2109.0) 이라 정수로 바꿔 출력
    print("이번 조회 횟수: ", int(queries))
    print("이번 적중 횟수: ", int(hits))
    print("이번 적중률: ", round(hit_rate, 2), "%")
