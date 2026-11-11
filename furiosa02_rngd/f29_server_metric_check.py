# [목적] furiosa-llm 서버의 /metrics 에서 실행 중 · 대기 중 요청 수, KV Cache 사용률, 요청 처리 시간 지표만 골라 출력 (Day49)
# [흐름] requests.get 으로 /metrics 읽기 → '#' 설명 줄 제외 → keywords 중 하나라도 들어 있는 줄만 출력
# ※ 지표 의미
#   num_requests_running        : 지금 NPU 에서 처리 중인 요청 수
#   num_requests_waiting        : 처리 자리가 없어 대기 중인 요청 수
#   kv_cache_usage_percent      : KV Cache 사용률 (0 ~ 1)
#   e2e_request_latency_seconds : 요청 시작 ~ 끝(end-to-end) 시간 분포 → _bucket(구간별 개수) · _sum(합) · _count(개수) 줄이 여러 개 나온다
# ※ 서버를 켠 뒤의 값이라 다른 실습(f26 · f28 등) 을 돌리는 중에 실행하면 running · waiting 이 0 이 아니게 나온다

# requests : HTTP 요청 라이브러리 (Day48 kingkeras 의 urllib.request 와 같은 일을 더 짧게)
import requests

url = "http://127.0.0.1:8000/metrics"

response = requests.get(url, timeout=5)
# 응답 코드가 4xx · 5xx (오류) 이면 예외를 일으킨다
response.raise_for_status()

# 찾을 지표 이름의 일부
keywords = [
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_percent",
    "e2e_request_latency_seconds",
]

for line in response.text.splitlines():
    if line.startswith("#"):
        continue
    # any(...) : keywords 중 하나라도 줄 안에 있으면 True
    if any(word in line for word in keywords):
        print(line)
