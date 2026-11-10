# [목적] furiosa02_rngd 실습 파일들이 공통으로 import 하는 LLM 모듈 (Day47 · Day48)
#        Day47 : get_llm (Day45 의 a_kingkeras.py 와 같은 역할, 기본값을 None 으로 바꿨다)
#        Day48 : run_once (요청 1 번 + 시간 · 토큰 수 측정), get_cache_metrics (서버 /metrics 로 캐시 상태 확인) 추가
# [흐름] get_llm(max_tokens, temperature, top_p) → furiosa-llm 서버(127.0.0.1:8000) 에 붙는 ChatOpenAI 반환
#        run_once(prompt, max_tokens) → invoke 앞뒤 시간 + usage_metadata 토큰 수를 dict 로 반환
#        get_cache_metrics() → Prefix Cache 조회 · 적중 횟수 · 적중률, KV Cache 사용률 출력 후 dict 로 반환
# ※ None = 값을 정하지 않음 → 요청에 넣지 않아 서버(모델) 기본 설정을 따른다

# 서버 실행 명령 (NPU 0 번에 Qwen3-8B-FP8 을 띄운다)
"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:0 \
    --host 0.0.0.0 \
    --port 8000
"""

import time
from  langchain_openai import ChatOpenAI

# 세 옵션 모두 기본값 None → 필요한 것만 골라서 넘긴다 (예: get_llm(max_tokens=500))
def get_llm(max_tokens=None, temperature=None, top_p=None):
    llm = ChatOpenAI(
        # furiosa-llm serve 로 띄운 로컬 OpenAI 호환 서버
        base_url = "http://127.0.0.1:8000/v1",
        # 로컬 서버라 아무 값
        api_key = "EMPTY",
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
    )
    return llm

# 질문 1 개를 보내고 걸린 시간 · 입력 / 출력 토큰 수 · 답변을 dict 로 돌려준다 (f21 · f22 에서 사용)
# get_llm(max_tokens) → 첫 번째 인자 자리라 max_tokens 로 들어간다
def run_once(prompt, max_tokens=1000):
    llm = get_llm(max_tokens)
    # invoke 앞뒤 시간 차 = 요청 ~ 답변 완료까지의 응답 시간(latency)
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    # usage_metadata 가 None 이면 빈 dict 로 바꿔, 아래 .get() 이 오류 없이 0 을 돌려주게 한다
    usage = response.usage_metadata or {}
    return {
        "elapsed" : elapsed,
        # usage["input_tokens"] 처럼 [] 로 꺼내면 키가 없을 때 KeyError → .get(키, 기본값) 을 쓴다
        # "input_tokens" : usage["input_tokens"],
        "input_tokens" : usage.get("input_tokens", 0),
        "output_tokens" : usage.get("output_tokens", 0),
        "text" : response.content
    }

# urllib.request : 파이썬 기본 라이브러리, URL 에 요청을 보내 응답을 읽는다 (설치 필요 없음)
import urllib.request

# furiosa-llm 서버의 /metrics 를 읽어 Prefix Cache · KV Cache 상태를 출력하고 dict 로 반환
# /metrics 응답 (한 줄에 지표 하나) :
#   # TYPE furiosa_llm_prefix_cache_queries counter
#   furiosa_llm_prefix_cache_queries{model_name="furiosa-ai/Qwen3-8B-FP8",engine="0"} 19015
#   → 지표이름{라벨들} 값
# ※ counter(누적값) 라서 서버를 켠 뒤부터 계속 쌓이고, 서버를 다시 띄우면 0 부터 센다
#   → 같은 코드여도 서버를 언제 켰는지 · 그동안 어떤 요청이 들어왔는지에 따라 값이 다르다
def get_cache_metrics():
    url = "http://127.0.0.1:8000/metrics"

    # timeout=5 : 5 초 안에 응답이 없으면 오류 (서버가 꺼져 있으면 연결 오류)
    with urllib.request.urlopen(url, timeout=5) as response:
        data = response.read().decode("utf-8")   # bytes → 문자열

    hits = 0       # Prefix Cache 적중 : 이전 요청에서 계산해 둔 KV 를 재사용한 입력 토큰 수
    queries = 0    # Prefix Cache 조회 : 캐시에 있는지 찾아본 입력 토큰 수 (요청 수가 아니라 토큰 수)
    kv_usage = 0   # KV Cache 사용률 : KV Cache 공간 중 지금 사용 중인 비율

    for line in data.splitlines():
        # '# HELP' · '# TYPE' 설명 줄은 건너뛴다
        if line.startswith("#"):
            continue
        # 줄 끝의 값 : split() 으로 공백 기준으로 나눠 마지막 [-1]
        # '지표이름{' 까지 넣어 찾아야 이름이 비슷한 다른 지표와 섞이지 않는다
        # engine 이 여러 개면 줄도 여러 개 → += 로 합친다
        if "furiosa_llm_prefix_cache_hits{" in line:
            hits += float(line.split()[-1])
        elif "furiosa_llm_prefix_cache_queries{" in line:
            queries += float(line.split()[-1])
        elif "furiosa_llm_kv_cache_usage_percent{" in line:
            # 서버 값은 0 ~ 1 비율 → × 100 해서 % 로
            kv_usage = float(line.split()[-1]) * 100

    # 적중률(%) = 적중 / 조회, 조회가 0 이면 0 으로 나누기 오류가 나지 않게 0
    hit_rate = hits / queries * 100 if queries > 0 else 0

    print("\n========== Furiosa NPU Cache 상태 ==========")
    print(f"Prefix Cache 조회 횟수:  {queries:.0f}")
    print(f"Prefix Cache 적중 횟수:  {hits:.0f}")
    print(f"Prefix Cache 적중률:  {hit_rate:.2f}")
    print(f"KV Cache 사용률:  {kv_usage:.2f}")

    # 출력만 하지 않고 값도 돌려줘서, 호출한 쪽에서 요청 전후 차이를 계산할 수 있다
    return {
        "queries" : queries,
        "hits" : hits,
        "hit_rate" : hit_rate,
        "kv_usage" : kv_usage,
    }

# 모듈 맨 바깥에서 호출하면 import 하는 순간(요청을 보내기 전) 실행되므로 주석 처리
# → 요청을 보낸 뒤 f22_kv_cache2.py 에서 호출한다
# get_cache_metrics()
