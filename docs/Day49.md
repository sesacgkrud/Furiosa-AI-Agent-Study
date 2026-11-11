# Day49 - RNGD 순차 · 동시 요청 처리량 비교 (RPS · Output TPS) & submit 으로 요청별 완료 순서 확인 & Prefix Cache 적중 측정 & 서버 /metrics 지표 확인

**학습 기간:** 2026-11-11

> Day48 에 이어 furiosa-llm 서버(Qwen3-8B-FP8) 에 여러 요청을 보내는 방법을 실습했다.
>
> 시작 시각 · 출력 길이가 다른 요청 5 개를 `pool.submit` 으로 동시에 보내, 늦게 들어온 짧은 요청이 먼저 들어온 긴 요청을 기다리지 않고 먼저 끝나는 것을 확인했다(f21_3, continuous batching).
>
> 앞부분(prefix) 이 같은 긴 질문을 4 번 보내며 Prefix Cache 조회 · 적중 횟수를 보고, 요청 전후 `get_cache_metrics()` 값의 차이로 **이번 요청의** 조회 · 적중 횟수와 적중률을 계산했다(f23).
>
> 요청 1 개(f24) → 8 개 순차(f25) → 8 개 동시(f26) 로 전체 시간과 **요청 처리량(RPS, req/s)** 을 비교하고, 16 개 요청에서 RPS 와 **Output TPS** 를 함께 계산한 뒤(f27) 동시 요청 수 1 ~ 4 별로 비교했다(f28). 마지막으로 `requests` 로 서버 `/metrics` 에서 실행 중 · 대기 중 요청 수, KV Cache 사용률, 요청 처리 시간 지표를 골라 출력했다(f29).

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| pool.submit | 작업 1 개를 등록하고 Future 를 바로 돌려준다 (작업마다 인자를 다르게) |
| Future.result() | 그 작업이 끝날 때까지 기다렸다가 반환값을 꺼낸다 |
| time.sleep | 지정한 초만큼 멈춘다 → 요청 시작 시각을 늦춘다 |
| continuous batching | 늦게 온 짧은 요청이 긴 요청을 기다리지 않고 먼저 끝난다 |
| Prefix Cache 증가량 | 요청 전후 `get_cache_metrics()` 차이 = 이번 요청의 조회 · 적중 횟수 |
| RPS (req/s) | 요청 수 / 전체 시간 = 1 초에 끝낸 요청 수 |
| Output TPS | 전체 출력 토큰 / 전체 시간 ≈ RPS × 요청당 출력 토큰 |
| 순차 vs 동시 | 8 개 순차 8.70 초 → 동시 4 개 2.49 초 (약 3.5 배) |
| requests.get | HTTP 요청 → `.text` 로 본문, `raise_for_status()` 로 오류 확인 |
| any() | 하나라도 True 면 True → 키워드 중 하나가 들어 있는 줄만 출력 |

---

## 📖 핵심 학습 내용

### 1. submit 으로 시작 시각 · 길이가 다른 요청 보내기 (f21_continuous_batching3)

```python
def ask(name, delay, max_tokens):
    time.sleep(delay)
    print(f"{name} 요청 시작", flush=True)
    start = time.perf_counter()
    result = run_once(f"NPU와 LLM 추론 최적화를 자세히 설명해줘. 요청 {name}", max_tokens)
    elapsed = time.perf_counter() - start
    print(f"{name} 요청 완료: {elapsed:.2f}초", flush=True)
    return result

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [
        pool.submit(ask, "A", 0, 800),
        pool.submit(ask, "B", 1, 200),
        pool.submit(ask, "C", 2, 1000),
        pool.submit(ask, "D", 3, 200),
        pool.submit(ask, "E", 4, 300),
    ]
    results = [f.result() for f in futures]
```

- `pool.map` 은 함수 하나에 리스트 값을 하나씩 넣지만, `pool.submit(함수, 인자들...)` 은 작업마다 인자를 여러 개 · 다르게 줄 수 있다
- `submit` 은 바로 Future 를 돌려주고, `f.result()` 에서 그 작업이 끝날 때까지 기다린다
- 방법2 : `with` 없이 `pool = ThreadPoolExecutor(...)` 로 만들고 끝에 `pool.shutdown()` 으로 닫는다 (같은 동작)
- `flush=True` : 여러 스레드의 출력이 실제 시각 순서대로 바로 보이게 한다

```
A 요청 시작
B 요청 시작
C 요청 시작
D 요청 시작
B 요청 완료: 2.39초
D 요청 완료: 2.37초
E 요청 시작
A 요청 완료: 9.58초
E 요청 완료: 3.48초
C 요청 완료: 11.41초
```

| 요청 | 시작 대기 | max_tokens | 걸린 시간 |
|:---:|:---:|:---:|:---:|
| A | 0 초 | 800 | 9.58 초 |
| B | 1 초 | 200 | 2.39 초 |
| C | 2 초 | 1000 | 11.41 초 |
| D | 3 초 | 200 | 2.37 초 |
| E | 4 초 | 300 | 3.48 초 |

- 늦게 시작한 짧은 요청 B · D 가 먼저 시작한 긴 요청 A · C 보다 먼저 끝났다 → 서버가 요청들을 함께 처리하다가 끝난 요청은 바로 내보낸다 (continuous batching)
- E 는 스레드 4 개가 A ~ D 로 차 있어서, B 가 끝나 스레드가 빈 뒤에야 `sleep(4)` 를 시작했다 (`time.sleep` 이 스레드 안에 있어 delay 는 스레드를 받은 시점부터 센다)

### 2. Prefix Cache 적중 측정 (f23_prefix_cache1 · 2)

```python
prefix = "NPU 추론 최적화에 대한 설명입니다." * 150

for i in range(1, 5):
    prompt = prefix + f"\n질문 {i}: KV Cache란?"

    before = get_cache_metrics()    # 요청 전 캐시 상태
    result = run_once(prompt, 100)
    after = get_cache_metrics()     # 요청 후 캐시 상태

    hits = after["hits"] - before["hits"]
    queries = after["queries"] - before["queries"]
    if queries > 0:
        hit_rate = hits / queries * 100
    else:
        hit_rate = 0

    print("이번 조회 횟수: ", int(queries))
    print("이번 적중 횟수: ", int(hits))
    print("이번 적중률: ", round(hit_rate, 2), "%")
```

- 4 개 질문은 앞부분(prefix) 이 같고 끝의 질문 번호만 다르다
- f23_1 : 요청마다 `get_cache_metrics()` 로 서버 누적값을 출력
- f23_2 : Day48 에서 `get_cache_metrics()` 가 dict 를 돌려주도록 만든 것을 이용해, 요청 전후 차이 = **이번 요청의** 조회 · 적중 횟수를 계산
- 1 회 요청은 prefix 가 아직 캐시에 없어 적중이 적고, 2 회부터는 1 회에서 계산한 prefix 의 KV 를 재사용한다
- 메트릭 값은 float 이라 `int()` 로 바꿔 출력한다

### 3. 단일 · 순차 · 동시 요청 비교 (f24 · f25 · f26)

```python
# f25 : 순차 (앞 요청이 끝나야 다음 요청)
for prompt in prompts:
    llm.invoke(prompt)

# f26 : 동시 4 개
def ask(prompt):
    return llm.invoke(prompt)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(ask, prompts))

print("요청 처리량", round(len(prompts) / elapsed, 3), "req/s")
```

| 파일 | 방식 | 요청 수 | 전체 시간 | 요청 처리량 |
|---|---|:---:|:---:|:---:|
| f24 | 단일 | 1 | 1.129 초 (입력 19 · 출력 100 토큰) | - |
| f25 | 순차 | 8 | 8.704 초 | 0.919 req/s |
| f26 | 동시 4 개 | 8 | 2.494 초 | 3.208 req/s |

- 순차는 요청 1 개 지연(약 1.1 초) × 8 ≈ 8.7 초 → 처리량은 1 / 지연에 가깝다
- 동시 4 개는 같은 8 개를 2.494 초에 끝내 처리량이 약 3.5 배

### 4. RPS · Output TPS (f27_rps · f28_concurrent_request_count)

```python
# f27 : 16 개를 동시 4 개로
def ask(prompt):
    response = llm.invoke(prompt)
    usage = response.usage_metadata or {}
    return usage.get("output_tokens", 0)

print("Output TPS: ", round(total_tokens / elapsed, 2))
print("RPS: ", round(len(prompts) / elapsed, 2))

# f28 : 동시 요청 수 1 ~ 4 비교, ask 가 (지연 시간, 출력 토큰) 튜플을 돌려준다
for workers in [1, 2, 3, 4]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(ask, prompts))
    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)
```

- f27 (16 개 · 동시 4 개) : 출력 1600 토큰 / 4.88 초 → Output TPS 328.2, RPS 3.28 (= 328.2 / 100 토큰)

**f28 (16 개, max_tokens 100)**

| Workers | 전체 시간 | 평균 지연 | 요청 처리량 | Output TPS |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 17.36 초 | 1.08 초 | 0.92 req/s | 92.17 |
| 2 | 9.3 초 | 1.16 초 | 1.72 req/s | 172.09 |
| 3 | 7.11 초 | 1.2 초 | 2.25 req/s | 224.92 |
| 4 | 4.86 초 | 1.21 초 | 3.29 req/s | 329.02 |

- 동시 요청 수를 1 → 4 로 늘리면 평균 지연은 1.08 → 1.21 초로 조금만 늘고, 처리량은 약 3.6 배
- Workers 3 은 16 개가 3 개씩 나눠떨어지지 않아(3 × 5 + 1) 마지막 1 개를 혼자 처리하는 시간이 더해져 처리량이 덜 늘었다

### 5. 서버 /metrics 지표 확인 (f29_server_metric_check)

```python
import requests

response = requests.get(url, timeout=5)
response.raise_for_status()

keywords = [
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_percent",
    "e2e_request_latency_seconds",
]

for line in response.text.splitlines():
    if line.startswith("#"):
        continue
    if any(word in line for word in keywords):
        print(line)
```

| 지표 | 의미 |
|---|---|
| `num_requests_running` | 지금 처리 중인 요청 수 |
| `num_requests_waiting` | 처리 자리가 없어 대기 중인 요청 수 |
| `kv_cache_usage_percent` | KV Cache 사용률 (0 ~ 1) |
| `e2e_request_latency_seconds` | 요청 시작 ~ 끝 시간 분포 (`_bucket` · `_sum` · `_count` 줄) |

- Day48 의 `urllib.request.urlopen` 대신 `requests.get` 으로 같은 `/metrics` 를 읽었다
- `raise_for_status()` : 응답 코드가 오류(4xx · 5xx) 면 예외를 일으킨다
- `any(조건 for ...)` : 키워드 중 하나라도 줄에 들어 있으면 출력

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `furiosa02_rngd/f21_continuous_batching3.py` | `pool.submit` 으로 시작 시각 · max_tokens 가 다른 요청 A ~ E 를 보내 완료 순서 확인 |
| `furiosa02_rngd/f23_prefix_cache1.py` | 같은 prefix + 다른 질문 4 번 → 요청마다 Prefix Cache 누적값 출력 |
| `furiosa02_rngd/f23_prefix_cache2.py` | 요청 전후 `get_cache_metrics()` 차이로 이번 요청의 조회 · 적중 횟수 · 적중률 |
| `furiosa02_rngd/f24_single_request.py` | 요청 1 개의 지연 시간 · 입력 / 출력 토큰 |
| `furiosa02_rngd/f25_sequential_request.py` | 8 개 순차 요청 → 전체 시간 · req/s |
| `furiosa02_rngd/f26_concurrent_request.py` | 8 개 동시 4 개 요청 → 전체 시간 · req/s |
| `furiosa02_rngd/f27_rps.py` | 16 개 동시 4 개 → Output TPS · RPS |
| `furiosa02_rngd/f28_concurrent_request_count.py` | 동시 요청 수 1 ~ 4 별 전체 시간 · 평균 지연 · req/s · Output TPS |
| `furiosa02_rngd/f29_server_metric_check.py` | `requests` 로 `/metrics` 에서 running · waiting · KV Cache · e2e latency 지표 출력 |
| `docs/Day49.md` | Day49 학습 기록 신규 작성 |
| `docs/Day48.md` | 하단 nav에 Day49 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(furiosa02_rngd/ Day49) / 진행도(49일, 61%) 갱신 (수정) |

---

## 📊 실행 결과

| 파일 | 설정 | 결과 |
|---|---|---|
| f21_3 | A ~ E (max_tokens 800 · 200 · 1000 · 200 · 300) | 완료 순서 B · D · A · E · C (2.39 · 2.37 · 9.58 · 3.48 · 11.41 초) |
| f24 | 요청 1 개 (max_tokens 100) | 1.129 초, 입력 19 · 출력 100 토큰 |
| f25 | 8 개 순차 | 8.704 초, 0.919 req/s |
| f26 | 8 개 동시 4 개 | 2.494 초, 3.208 req/s |
| f27 | 16 개 동시 4 개 | 1600 토큰 / 4.88 초, Output TPS 328.2, RPS 3.28 |
| f28 | 16 개, workers 1 → 4 | 17.36 → 4.86 초, 0.92 → 3.29 req/s, TPS 92.17 → 329.02, 평균 지연 1.08 → 1.21 초 |

---

## 💻 핵심 개념

### map vs submit

```
pool.map(함수, 리스트)          : 리스트 값을 하나씩 함수에 넣고, 결과를 입력 순서대로 돌려준다
pool.submit(함수, 인자1, 인자2) : 작업 1 개 등록 → Future 반환 (작업마다 인자를 다르게)
future.result()                 : 그 작업이 끝날 때까지 기다렸다가 반환값을 꺼낸다
with ThreadPoolExecutor(...)    : 블록 끝에서 모든 작업을 기다리고 자동으로 닫는다 (= pool.shutdown())
```

### 처리량 지표

```
지연 (latency)  : 요청 1 개의 시작 ~ 끝 시간
RPS (req/s)     : 요청 수 / 전체 시간
Output TPS      : 전체 출력 토큰 / 전체 시간 ≈ RPS × 요청당 출력 토큰
순차 요청        : 전체 시간 ≈ 요청 수 × 지연      → RPS ≈ 1 / 지연
동시 요청 N 개   : 전체 시간 ≈ (요청 수 / N, 올림) × 지연 → RPS 약 N 배, 지연은 거의 그대로
```

### 요청별 캐시 증가량

```
before = get_cache_metrics()   → 요청 전 누적값
run_once(...)
after  = get_cache_metrics()   → 요청 후 누적값
이번 조회 = after["queries"] - before["queries"]
이번 적중 = after["hits"]    - before["hits"]
```

---

## 성과

- `pool.submit` · `Future.result()` 로 작업마다 다른 인자의 요청을 동시에 보내고, 짧은 요청이 먼저 끝나는 continuous batching 동작을 확인했다
- 요청 전후 메트릭 차이로 요청 1 개의 Prefix Cache 조회 · 적중 횟수와 적중률을 계산했다
- 단일 · 순차 · 동시 요청을 비교해 RPS 와 Output TPS 를 계산했다
- 동시 요청 수를 늘려도 지연은 거의 그대로이고 처리량이 늘어나는 것을 수치로 확인했다
- `requests` 로 서버 `/metrics` 에서 필요한 지표만 골라 출력했다

---

## 💡 주요 학습 포인트

1. **pool.submit**: 작업마다 인자를 다르게 등록, Future 를 돌려준다
2. **Future.result()**: 작업이 끝날 때까지 기다렸다가 결과를 꺼낸다
3. **time.sleep 위치**: 스레드 안에서 sleep 하면 스레드를 받은 시점부터 센다
4. **continuous batching**: 늦게 온 짧은 요청이 긴 요청을 기다리지 않고 먼저 끝난다
5. **누적 메트릭은 전후 차이로**: before / after 의 차이 = 이번 요청의 값
6. **RPS**: 요청 수 / 전체 시간 (req/s)
7. **Output TPS ≈ RPS × 요청당 출력 토큰**
8. **순차 vs 동시**: 순차는 지연 × 요청 수, 동시 N 개는 약 1/N 시간
9. **나눠떨어지지 않는 동시 요청 수**: 남은 요청을 처리하는 시간이 더해져 처리량이 덜 는다
10. **requests.get · raise_for_status · any**: HTTP 요청, 오류 확인, 조건 중 하나라도 맞는지 검사

---

[⬅️ Day48](Day48.md) · [🏠 전체 목차](../README.md)
