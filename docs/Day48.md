# Day48 - RNGD 동시 요청 Continuous Batching 처리량 비교 & KV Cache · Prefix Cache 확인 (/metrics) & 파이썬 map · lambda

**학습 기간:** 2026-11-10

> Day47 의 `kingkeras.py` 에 요청 1 번의 응답 시간 · 입력 / 출력 토큰 수를 돌려주는 `run_once()` 를 추가하고, 이를 이용해 furiosa-llm 서버(Qwen3-8B-FP8) 의 동작을 두 가지로 확인했다.
>
> 첫째, 같은 질문 8 개를 `ThreadPoolExecutor` 로 **동시 요청 수 1 · 2 · 4 · 8** 로 바꿔 보내 전체 시간 · 처리량(초당 출력 토큰) · 요청별 평균 지연을 비교했다(f21). 서버의 **continuous batching** 덕분에 동시 요청 수를 2 배로 늘릴 때마다 전체 시간은 절반, 처리량은 2 배가 되고 요청 1 개의 지연은 거의 그대로였다.
>
> 둘째, 입력 길이 · 출력 길이를 바꿔 요청하며 토큰 수 · 응답 시간을 비교하고(f22_1), 서버의 `/metrics` 를 읽는 `get_cache_metrics()` 를 만들어 **Prefix Cache 조회 · 적중 토큰 수, 적중률, KV Cache 사용률**을 요청마다 출력했다(f22_2).
>
> 셋째, f21 에서 쓴 `pool.map` · `lambda` 의 바탕이 되는 파이썬 기본 문법 `map(함수, 리스트)` 와 `lambda` 를 따로 실습했다(python/). 기존 모듈 · 패키지 실습 폴더(Day45)는 `python_pkg/` 로 이름을 바꿨다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| run_once | 요청 1 번 → `{elapsed, input_tokens, output_tokens, text}` dict 반환 |
| ThreadPoolExecutor | `max_workers` 개 스레드로 함수를 동시에 실행 → 서버에 동시에 들어가는 요청 수 |
| pool.map | 리스트의 각 값에 함수를 적용, 결과는 입력 순서대로 |
| lambda | 이름 없는 한 줄 함수 (`lambda p: run_once(p, 50)`) |
| map | `map(함수, 리스트)` → 각 값에 함수 적용, map 객체라 `list()` 로 감싸서 확인 |
| map + lambda | `list(map(lambda x: "합격" if x >= 60 else "불합격", score))` |
| statistics.mean | 평균 (파이썬 기본 라이브러리) |
| continuous batching | 동시에 들어온 요청을 한 배치로 묶어 decode → 처리량은 늘고 요청별 지연은 거의 그대로 |
| 처리량 (throughput) | 전체 출력 토큰 / 전체 시간 (tokens/s) |
| KV Cache | decode 때 앞 토큰들의 Key · Value 를 저장해 두는 NPU 메모리 공간 |
| Prefix Cache | 이전 요청과 앞부분이 같은 입력이면 저장된 KV 를 재사용 (적중) |
| /metrics | 서버 상태 지표 페이지 → `urllib.request.urlopen` 으로 읽어 줄별로 파싱 |

---

## 📖 핵심 학습 내용

### 1. run_once · get_cache_metrics 추가 (kingkeras.py)

```python
def run_once(prompt, max_tokens=1000):
    llm = get_llm(max_tokens)
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata or {}
    return {
        "elapsed" : elapsed,
        "input_tokens" : usage.get("input_tokens", 0),
        "output_tokens" : usage.get("output_tokens", 0),
        "text" : response.content
    }
```

- `usage_metadata or {}` : 값이 None 이면 빈 dict 로 바꾼다
- `usage["input_tokens"]` 대신 `usage.get("input_tokens", 0)` → 키가 없어도 KeyError 없이 0

### 2. 동시 요청 · Continuous Batching (f21_continuous_batching1 · 2)

```python
from concurrent.futures import ThreadPoolExecutor
import statistics

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 자세히 설명해줘."
    for i in range(8)
]

def ask(prompt):
    return run_once(prompt, 100)

for workers in [1,2,4,8]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(ask, prompts))
    elasped = time.perf_counter() - start

    total_tokens = sum(r["output_tokens"] for r in results)
    avg_latency = statistics.mean(r["elapsed"] for r in results)
```

- 질문 8 개를 `max_workers` = 1 · 2 · 4 · 8 개 스레드로 보낸다 → 동시에 서버에 들어가는 요청 수
- `with` 블록이 끝날 때 모든 요청이 끝날 때까지 기다리므로, 블록 앞뒤 시간 = 8 개를 모두 처리한 전체 시간
- f21_2 는 `ask` 대신 `lambda prompts: run_once(prompts, 50)` 를 넘기고 max_tokens 를 50 으로 줄였다

**f21_1 (max_tokens 100)**

| 동시 요청 수 | 전체 시간 | 전체 출력 토큰 | 처리량 | 평균 요청 지연 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 9.03 초 | 800 | 88.61 tokens/s | 1.09 초 |
| 2 | 4.63 초 | 800 | 172.85 tokens/s | 1.15 초 |
| 4 | 2.44 초 | 800 | 327.30 tokens/s | 1.22 초 |
| 8 | 1.31 초 | 800 | 612.23 tokens/s | 1.30 초 |

**f21_2 (max_tokens 50)**

| 동시 요청 수 | 전체 시간 | 전체 출력 토큰 | 처리량 | 평균 요청 지연 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 4.67 초 | 400 | 85.56 tokens/s | 0.55 초 |
| 2 | 2.41 초 | 400 | 165.85 tokens/s | 0.60 초 |
| 4 | 1.31 초 | 400 | 304.73 tokens/s | 0.65 초 |
| 8 | 0.71 초 | 400 | 563.98 tokens/s | 0.70 초 |

- 동시 요청 수를 2 배로 늘릴 때마다 전체 시간은 약 1/2, 처리량은 약 2 배
- 요청 1 개의 평균 지연은 1.09 → 1.30 초로 조금만 늘어난다 → 서버가 동시에 들어온 요청들을 한 배치로 묶어 decode 를 함께 진행하기 때문 (continuous batching)
- max_tokens 를 절반으로 줄이면 출력 토큰 · 시간 · 지연도 약 절반이고, 처리량(tokens/s) 은 비슷하다

### 3. 입력 · 출력 길이와 응답 시간 (f22_kv_cache1)

```python
long_prompt = (
    "NPU는 인공지능 연산을 가속하는 프로세서다." * 100
) + "위 내용을 요약해줘."

tests = [
    ("짧은 입력", short_prompt, 80),
    ("긴 입력", long_prompt, 80),
    ("긴 입력과 긴 출력", long_prompt, 400),
]

for name, prompt, max_tokens in tests:
    result = run_once(prompt, max_tokens)
    ...
    input("서버 KV Cache 사용률을 확인하고 싶다 Enter...")
```

| 실험 | 입력 토큰 | 출력 토큰 | 소요 시간 |
|---|:---:|:---:|:---:|
| 짧은 입력 | 15 | 80 | 0.89 초 |
| 긴 입력 | 1817 | 80 | 0.88 초 |
| 긴 입력과 긴 출력 | 1817 | 193 | 2.14 초 |

- 입력이 15 → 1817 토큰으로 120 배 길어져도 응답 시간은 거의 같다 (prefill 은 입력 전체를 한 번에 처리)
- 출력 토큰이 늘어야 시간이 늘어난다 (decode 는 토큰을 하나씩 생성)
- `input()` 으로 요청 사이에 멈춰 서버의 KV Cache 사용률을 확인했다

### 4. /metrics 로 Prefix Cache · KV Cache 확인 (kingkeras.get_cache_metrics · f22_kv_cache2)

```python
import urllib.request

def get_cache_metrics():
    url = "http://127.0.0.1:8000/metrics"
    with urllib.request.urlopen(url, timeout=5) as response:
        data = response.read().decode("utf-8")

    for line in data.splitlines():
        if line.startswith("#"):
            continue
        if "furiosa_llm_prefix_cache_hits{" in line:
            hits += float(line.split()[-1])
        elif "furiosa_llm_prefix_cache_queries{" in line:
            queries += float(line.split()[-1])
        elif "furiosa_llm_kv_cache_usage_percent{" in line:
            kv_usage = float(line.split()[-1]) * 100

    hit_rate = hits / queries * 100 if queries > 0 else 0
```

```python
# f22_kv_cache2.py
prompts = [
    "NPU란 무엇인가?",
    "NPU의 특징을 설명해줘." * 100,
    "NPU의 특징을 설명해줘." * 300,
]

for i, prompt in enumerate(prompts, 1):
    result = run_once(prompt, 300)
    ...
    get_cache_metrics()
```

| 지표 | 의미 |
|---|---|
| `furiosa_llm_prefix_cache_queries` | Prefix Cache 에서 찾아본 입력 토큰 수 (누적) |
| `furiosa_llm_prefix_cache_hits` | 그중 캐시에 있어서 재사용한 토큰 수 (누적) |
| 적중률 | hits / queries × 100 |
| `furiosa_llm_kv_cache_usage_percent` | KV Cache 공간 중 사용 중인 비율 (0 ~ 1 → × 100 해서 %) |

- `/metrics` 는 `# HELP` · `# TYPE` 설명 줄과 `지표이름{라벨} 값` 줄로 되어 있다 → `#` 줄은 건너뛰고 `split()[-1]` 로 값만 꺼낸다
- 조회 · 적중은 **요청 수가 아니라 토큰 수**이고, 서버를 켠 뒤부터 쌓이는 **누적값**이다 → 서버를 다시 띄우면 0 부터 센다
  - 같은 코드여도 서버를 언제 켰는지, 그동안 어떤 요청이 들어왔는지에 따라 기기마다 값이 다르다
  - 이번 요청의 값은 요청 전후 **차이**로 본다
- `get_cache_metrics()` 는 요청을 보낸 뒤 호출해야 한다 → 모듈 맨 아래에서 바로 호출하면 import 하는 순간(요청 전) 실행된다

| 실험 | 입력 토큰 | 출력 토큰 | 응답 시간 | 조회 증가 | 적중 증가 |
|---|:---:|:---:|:---:|:---:|:---:|
| 1. 짧은 질문 | 15 | 300 | 3.25 초 | 14 | 14 |
| 2. 문장 × 100 | 909 | 300 | 3.28 초 | 908 | 908 |
| 3. 문장 × 300 | 2709 | 300 | 3.38 초 | 2708 | 2708 |

- 같은 질문을 이미 보낸 적 있는 서버에서 실행해, 입력 거의 전부가 캐시에서 재사용(적중) 되었다
- 질문 3 은 질문 2 와 같은 문장으로 시작하므로, 처음 보내더라도 질문 2 에서 계산한 앞부분을 재사용할 수 있다 (Prefix Cache)
- KV Cache 사용률은 요청이 쌓이면서 3.10 → 3.21 → 3.32 % 로 늘었다

### 5. 파이썬 map · lambda (python/p01 ~ p03)

f21_2 의 `pool.map(lambda p: run_once(p, 50), prompts)` 에 쓰인 `map` · `lambda` 를 파이썬 기본 문법으로 따로 실습했다.

```python
# p01_map.py
def multiply(x):
    return x * 10

numbers = [1,2,3,4]
result = map(multiply, numbers)

print(result)          # <map object at 0x000001F6794BACB0>
print(list(result))    # [10, 20, 30, 40]
```

- `map(함수, 리스트)` : 리스트의 각 값에 함수를 적용한다 (함수는 괄호 없이 이름만 넘긴다)
- 결과는 바로 리스트가 아니라 map 객체 → `print` 하면 `<map object at 0x...>`, `list()` 로 감싸야 값이 보인다

```python
# p02_lambda1.py
def multiply(x):
    return x * 2
print(multiply(10))              # 20

aaa = lambda x : x * 2
print(aaa(10))                   # 20

# p02_lambda2.py
aaa = lambda x, y : x + y
print(aaa(1, 2)) # 3
print(aaa(3, 4)) # 7
```

- `lambda 매개변수 : 돌려줄 값` → `def` · `return` 없이 한 줄로 만드는 이름 없는 함수
- 변수에 담으면 일반 함수처럼 호출할 수 있고, 매개변수가 여러 개면 `,` 로 구분한다

```python
# p03_lambda3.py
score = [85, 45, 70, 55, 95]
results = list(map(lambda x: "합격" if x >=60 else "불합격", score))
print(results) # ['합격', '불합격', '합격', '불합격', '합격']
```

- `값1 if 조건 else 값2` 를 lambda 안에 넣어 60 점 이상이면 "합격", 아니면 "불합격"
- 한 번만 쓸 함수는 `def` 로 따로 만들지 않고 `map` 안에 lambda 로 바로 넣는다

### 6. python/ → python_pkg/ 폴더 이름 변경

- Day45 의 모듈 · 패키지 실습 파일(p01_car ~ p51_from_module, machine/ · machine/test/)을 내용 변경 없이 `python_pkg/` 로 옮기고, `python/` 은 오늘의 map · lambda 실습 폴더로 썼다
- Day45 문서의 파일 경로도 `python_pkg/` 로 고쳤다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `furiosa02_rngd/kingkeras.py` | `run_once()` (응답 시간 · 입력 / 출력 토큰 수 dict) · `get_cache_metrics()` (`/metrics` 로 Prefix Cache · KV Cache 상태) 추가 (수정) |
| `furiosa02_rngd/f21_continuous_batching1.py` | 질문 8 개를 `ThreadPoolExecutor` 동시 요청 수 1 · 2 · 4 · 8 로 → 전체 시간 · 처리량 · 평균 지연 (max_tokens 100) |
| `furiosa02_rngd/f21_continuous_batching2.py` | `pool.map` 에 `lambda` 를 넘기고 max_tokens 50 으로 같은 비교 |
| `furiosa02_rngd/f22_kv_cache1.py` | 짧은 입력 · 긴 입력 · 긴 입력 + 긴 출력의 토큰 수 · 응답 시간, `input()` 으로 멈춰 KV Cache 확인 |
| `furiosa02_rngd/f22_kv_cache2.py` | 입력 길이를 늘려 가며 요청마다 `get_cache_metrics()` 로 Prefix Cache · KV Cache 상태 출력 |
| `python/p01_map.py` | `map(multiply, numbers)` → map 객체 · `list()` 로 결과 확인 |
| `python/p02_lambda1.py` · `python/p02_lambda2.py` | `def` 함수와 같은 `lambda` · 매개변수 2 개 lambda |
| `python/p03_lambda3.py` | `list(map(lambda x: "합격" if x >= 60 else "불합격", score))` |
| `python_pkg/` | Day45 의 `python/` 모듈 · 패키지 실습 파일 폴더 이름 변경 (내용 같음) |
| `furiosa01_rngd/f03_tool_calling.py` | 서버 실행 명령 주석 `--devices npu:0` (수정) |
| `docs/Day43.md` | furiosa-llm serve 명령 `--devices npu:0` (수정) |
| `docs/Day45.md` | 모듈 · 패키지 실습 파일 경로 `python/` → `python_pkg/` (수정) |
| `docs/Day48.md` | Day48 학습 기록 신규 작성 |
| `docs/Day47.md` | 하단 nav에 Day48 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(furiosa02_rngd/ · python/ Day48, python_pkg/ Day45) / 진행도(48일, 60%) 갱신 (수정) |

---

## 📊 실행 결과

| 파일 | 설정 | 결과 |
|---|---|---|
| f21_1 (max_tokens 100, 질문 8 개) | 동시 요청 수 1 → 8 | 전체 9.03 → 1.31 초 · 처리량 88.61 → 612.23 tokens/s · 평균 지연 1.09 → 1.30 초 |
| f21_2 (max_tokens 50, 질문 8 개) | 동시 요청 수 1 → 8 | 전체 4.67 → 0.71 초 · 처리량 85.56 → 563.98 tokens/s · 평균 지연 0.55 → 0.70 초 |
| f22_1 | 짧은 입력 · 긴 입력 · 긴 입력 + 긴 출력 | 입력 15 · 1817 · 1817 / 출력 80 · 80 · 193 / 0.89 · 0.88 · 2.14 초 |
| f22_2 (max_tokens 300) | 입력 15 · 909 · 2709 토큰 | 3.25 · 3.28 · 3.38 초, 조회 · 적중 증가량이 입력 토큰 수와 거의 같음 (캐시 재사용) |
| p01 ~ p03 (python/) | map · lambda | `[10, 20, 30, 40]` · `20` · `3` / `7` · `['합격', '불합격', '합격', '불합격', '합격']` |

---

## 💻 핵심 개념

### 지연(latency) vs 처리량(throughput)

```
지연 (latency)      : 요청 1 개가 보내서 답을 받을 때까지 걸린 시간   → 사용자 1 명이 느끼는 속도
처리량 (throughput) : 전체 출력 토큰 / 전체 시간 (tokens/s)          → 서버가 1 초에 만들어 내는 토큰 수
continuous batching : 동시에 들어온 요청을 한 배치로 묶어 decode 를 함께 진행,
                      끝난 요청 자리에 새 요청을 바로 채운다 → 지연은 거의 그대로, 처리량은 동시 요청 수만큼 증가
```

### 동시 요청 코드 패턴

```
with ThreadPoolExecutor(max_workers=N) as pool:     # 스레드 N 개 = 동시 요청 N 개
    results = list(pool.map(함수, 리스트))           # 리스트 각 값에 함수 적용, 입력 순서대로 결과
lambda x: run_once(x, 50)                            # 이름 없는 한 줄 함수
statistics.mean(값들)                                # 평균
```

### KV Cache · Prefix Cache

```
KV Cache     : 앞 토큰들의 Key · Value 를 NPU 메모리에 저장 → decode 때 다시 계산하지 않음
               입력 + 출력 토큰이 많을수록 더 많이 쓴다
Prefix Cache : 새 요청의 앞부분이 이전 요청과 같으면 저장된 KV 를 그대로 재사용 → prefill 계산 생략
/metrics     : furiosa_llm_prefix_cache_queries (조회 토큰 수) · _hits (적중 토큰 수) · kv_cache_usage_percent
               counter 는 서버 시작 후 누적 → 요청 전후 차이로 본다
```

### map · lambda

```
map(함수, 리스트)             → 각 값에 함수 적용, map 객체 → list() 로 감싸서 확인 (한 번 꺼내면 비워진다)
lambda x : x * 2              → def f(x): return x * 2 와 같은 이름 없는 한 줄 함수
lambda x : 값1 if 조건 else 값2 → 조건에 따라 다른 값
list(map(lambda x: ..., 리스트)) → 한 번 쓸 함수를 바로 넣어 리스트 변환
```

---

## 성과

- `run_once()` 로 요청 1 번의 응답 시간 · 토큰 수를 재는 코드를 모듈에 정리했다
- `ThreadPoolExecutor` 로 동시 요청 수를 바꿔 보내 continuous batching 으로 처리량이 늘어나는 것을 확인했다
- 입력 길이보다 출력 길이가 응답 시간을 좌우한다는 것을 긴 입력 실험으로 다시 확인했다
- 서버 `/metrics` 를 직접 읽어 Prefix Cache 조회 · 적중 토큰 수와 KV Cache 사용률을 출력했다
- f21 에 쓴 `map` · `lambda` 를 파이썬 기본 문법으로 따로 실습했다 (map 객체 · 매개변수 2 개 lambda · 조건 lambda)

---

## 💡 주요 학습 포인트

1. **ThreadPoolExecutor**: `max_workers` 개 스레드로 함수를 동시에 실행 → LLM 서버에 동시 요청
2. **pool.map**: 리스트의 각 값에 함수를 적용하고 결과를 입력 순서대로 돌려준다
3. **lambda**: `def` 없이 한 줄로 만드는 이름 없는 함수
4. **continuous batching**: 동시 요청을 묶어 처리 → 동시 요청 수 2 배마다 처리량 약 2 배, 지연은 거의 그대로
5. **지연과 처리량은 다른 지표**: 사용자 1 명의 체감 속도 vs 서버 전체의 초당 토큰 수
6. **입력보다 출력이 시간을 결정**: 입력 120 배에도 응답 시간은 거의 같고, 출력 토큰이 늘면 시간이 는다
7. **KV Cache**: 앞 토큰들의 Key · Value 를 저장해 두는 공간, 토큰이 많을수록 사용률 증가
8. **Prefix Cache**: 앞부분이 같은 입력은 저장된 KV 를 재사용 (적중)
9. **/metrics 값은 누적 · 토큰 단위**: 서버 재시작 시 0, 기기 · 실행 순서마다 다르므로 요청 전후 차이로 본다
10. **측정 함수는 요청 뒤에 호출**: 모듈 맨 아래에서 호출하면 import 순간(요청 전) 실행된다
11. **map 은 map 객체를 돌려준다**: `list()` 로 감싸야 값이 보이고, 한 번 꺼내 쓰면 비워진다
12. **lambda + 조건 표현식**: `lambda x: "합격" if x >= 60 else "불합격"` 처럼 한 줄로 분기

---

[⬅️ Day47](Day47.md) · [🏠 전체 목차](../README.md)
