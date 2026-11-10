# Day47 - HalvingGridSearchCV 단계별 후보 축소 하이퍼파라미터 탐색 & RNGD 응답 시간 · TTFC · TPS 측정 & 전체 실습 코드 주석 정리

**학습 기간:** 2026-11-09

> Day45 의 `GridSearchCV`, Day46 의 `RandomizedSearchCV` 에 이어 `HalvingGridSearchCV` 로 하이퍼파라미터를 탐색했다. 42 개 조합을 **적은 샘플로 먼저 모두 교차 검증 → 점수 상위 1/factor 만 남기고 → 샘플 수를 factor 배로 늘려 다시 교차 검증** 하는 과정을 단계(iter)마다 반복한다.
>
> digits + XGBClassifier 에서 `min_resources` · `max_resources` · `factor` 를 바꿔 가며 단계 수 · 후보 수 · 샘플 수 · 점수 · 시간이 어떻게 달라지는지 비교했다(m10).
>
> RNGD 에서는 Day45 의 `get_llm()` 모듈을 `kingkeras.py` 로 다시 만들어(furiosa02_rngd/), furiosa-llm 서버(Qwen3-8B-FP8) 의 **응답 시간(latency)** 을 5 번 재서 평균을 내고(f12), 스트리밍에서 **첫 chunk 가 오기까지의 시간(TTFC)** 을 세 가지 방법으로 측정했다(f14). 질문 길이 · 요구 분량에 따른 응답 시간을 비교하고(f15), `usage_metadata` 로 입력 · 출력 토큰 수를 확인한 뒤(f16) 실제 출력 토큰으로 **TPS** 를 계산하고(f17), 입력 · 출력 토큰과 응답 시간을 한 표로 비교했다(f18). Day42 ~ 45 의 `rngd/` 폴더는 `furiosa01_rngd/` 로 이름을 바꿨다.
>
> 지금까지 올린 모든 실습 파일(keras1 · keras2 · RAG · ml · rngd 등)에 `[목적]` 머리 주석과 코드 해석 주석을 달고, 베이스 파일에서 복사되어 실제 코드와 맞지 않던 주석(파일 이름 · 출력 shape · 파라미터 수 · 결과 설명 등)을 바로잡았다. 코드는 바꾸지 않았다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| HalvingGridSearchCV | 모든 조합에서 출발해 단계마다 후보를 줄이고 샘플 수를 늘리는 GridSearchCV |
| import 방법 | 실험적 기능이라 `sklearn.model_selection` 에서 바로 import 하면 ImportError → 정의된 모듈에서 직접 import |
| param_grid | GridSearchCV 계열이라 인자 이름이 `param_grid` (`n_iter` 없음) |
| min_resources | 첫 단계 샘플 수 (기본값 → 2 × cv 개수 × 클래스 개수 = 100) |
| max_resources | 단계에서 쓸 수 있는 최대 샘플 수 (기본값 → x_train 행 수 1437) |
| factor | 단계마다 후보 1/factor, 샘플 수 factor 배 (기본값 3) |
| 단계 수 | `min_resources × factor^iter` 가 `max_resources` 를 넘지 않을 때까지만 진행 |
| latency | 같은 질문 5 번 → `time.perf_counter()` 로 회차별 시간 → 평균 (약 5.4 초) |
| TTFC | `llm.stream` 에서 내용이 있는 첫 chunk 까지 걸린 시간 (약 0.03 초) |
| prefill · decode | 입력 전체를 한 번에 처리(prefill) + 출력 토큰을 하나씩 생성(decode) → 응답 시간은 출력 길이에 비례 |
| usage_metadata | `response.usage_metadata` → `input_tokens` · `output_tokens` · `total_tokens` |
| TPS | 실제 `output_tokens` / 걸린 시간 (약 91 tokens/s) |

---

## 📖 핵심 학습 내용

### 1. HalvingGridSearchCV (m10_HalvingGridSearchCV)

Day46 의 m09_00 (RandomizedSearchCV) 을 베이스로, 데이터를 digits 로 하고 탐색 방법을 HalvingGridSearchCV 로 바꿨다.

```python
from sklearn.model_selection._search_successive_halving import HalvingGridSearchCV
# from sklearn.experimental import enable_halving_search_cv

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

# print(x_train.shape) # (1437, 64)

#2. 모델 구성
xgb = XGBClassifier()

model = HalvingGridSearchCV(
    estimator=xgb,
    param_grid=parameters,
    cv=kfold,
    verbose=1,
    refit=True,
    min_resources=50,                   # 첫 단계 샘플 수 (기본값으로 두면 100)
    max_resources=1000,                 # 단계에서 쓸 수 있는 최대 샘플 수 (기본값으로 두면 1437)
    factor=2,                           # 단계마다 후보 1/2 · 샘플 수 2 배 (기본값 3)
)
```

- HalvingGridSearchCV 는 실험적(experimental) 기능이라 `from sklearn.model_selection import HalvingGridSearchCV` 로 바로 가져오면 ImportError 가 난다 → 정의된 모듈 `_search_successive_halving` 에서 직접 import 했다
  - 공식 방법은 `from sklearn.experimental import enable_halving_search_cv` 를 먼저 import 하는 것
- GridSearchCV 계열이라 조합은 `param_grid` 로 넘긴다 (RandomizedSearchCV 의 `param_distributions` · `n_iter` 는 쓰지 않는다)
- parameters 는 Day45 · Day46 과 같은 42 개 조합

### 2. 단계(iter) 진행 방식

verbose=1 로 실행하면 단계마다 남은 후보 수(`n_candidates`)와 쓰는 샘플 수(`n_resources`)가 출력된다.

| 출력 | 의미 |
|---|---|
| `min_resources_` | 첫 단계 샘플 수 |
| `max_resources_` | 쓸 수 있는 최대 샘플 수 |
| `n_required_iterations` | 후보가 1 개로 줄어들 때까지 필요한 단계 수 |
| `n_possible_iterations` | 샘플 수 범위(min ~ max) 안에서 실제로 가능한 단계 수 |
| `n_iterations` | 실제로 진행한 단계 수 |
| `n_candidates` · `n_resources` | 그 단계의 후보 수 · 샘플 수 |

- 다음 단계 샘플 수가 `max_resources` 를 넘으면 거기서 멈추고, 마지막 단계에 남은 후보 중 최고를 고른다
- `min_resources` 기본값으로 두면 이 데이터에서는 2 × cv 개수(5) × 클래스 개수(10) = **100** 으로 정해진다

### 3. 설정별 비교

**결과 1 : min_resources=100 · max_resources=1437 · factor=3 (기본값과 같은 값)**

```
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
factor: 3
iter: 0  n_candidates: 42  n_resources: 100  → 210 fits
iter: 1  n_candidates: 14  n_resources: 300  →  70 fits
iter: 2  n_candidates: 5   n_resources: 900  →  25 fits
model.score :  0.9444444444444444
best_score :  0.9453755431409062
걸린시간 :  210.39 초
```

- 후보 42 → 14 → 5 (1/3 씩, 올림), 샘플 수 100 → 300 → 900 (3 배씩)
- 다음 단계 2700 은 1437 을 넘어서 3 단계에서 멈춘다 → 후보를 1 개까지 줄이려면 4 단계가 필요하지만 남은 5 개 중 최고를 고른다

**결과 2 : min_resources=50 · max_resources=1000 · factor=2 (현재 코드 설정)**

```
n_iterations: 5
n_required_iterations: 6
n_possible_iterations: 5
min_resources_: 50
max_resources_: 1000
factor: 2
iter: 0  n_candidates: 42  n_resources: 50   → 210 fits
iter: 1  n_candidates: 21  n_resources: 100  → 105 fits
iter: 2  n_candidates: 11  n_resources: 200  →  55 fits
iter: 3  n_candidates: 6   n_resources: 400  →  30 fits
iter: 4  n_candidates: 3   n_resources: 800  →  15 fits
model.score :  0.9416666666666667
best_score :  0.933490566037736
걸린시간 :  142.75 초
```

- 후보 42 → 21 → 11 → 6 → 3 (1/2 씩, 올림), 샘플 수 50 → 100 → 200 → 400 → 800 (2 배씩)
- 다음 단계 1600 은 1000 을 넘어서 5 단계에서 멈춘다
- 같은 설정으로 다시 실행하니 단계 · 후보 수 · 샘플 수는 똑같고 점수 · 시간은 달랐다 (model.score 0.9444 · best_score 0.9473 · 219.19 초)
  - 단계마다 x_train 에서 `n_resources` 개 샘플을 무작위로 뽑는데, `random_state` 를 주지 않아 실행마다 뽑히는 샘플이 달라진다

| 설정 | 단계 수 | 마지막 단계 (후보 · 샘플) | 총 훈련 | best_score | model.score | 걸린 시간 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 · 1437 · factor 3 | 3 | 5 개 · 900 | 305 | 0.9454 | 0.9444 | 210.39 초 |
| 50 · 1000 · factor 2 | 5 | 3 개 · 800 | 415 | 0.9335 | 0.9417 | 142.75 초 |
| 50 · 1000 · factor 2 (재실행) | 5 | 3 개 · 800 | 415 | 0.9473 | 0.9444 | 219.19 초 |

- factor 를 2 로 줄이면 단계가 3 → 5 개로 늘고, 총 훈련 횟수도 305 → 415 번으로 많아진다
- 같은 설정이라도 실행마다 점수 · 시간이 달라서, 한 번 실행한 결과만으로 두 설정의 점수 · 시간 우열을 정하기는 어렵다

### 4. LLM 모듈 · 기본 호출 (kingkeras · f11_basic)

```python
# kingkeras.py
def get_llm(max_tokens=None, temperature=None, top_p=None):
    llm = ChatOpenAI(
        base_url = "http://127.0.0.1:8000/v1",
        api_key = "EMPTY",
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
    )
    return llm
```

```python
# f11_basic.py  (서버 : furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:0 --host 0.0.0.0 --port 8000)
from kingkeras import get_llm

llm = get_llm(max_tokens=1000)
response = llm.invoke("NPU와 GPU의 차이를 설명해줘.")
print(response.content)
```

- Day45 의 `a_kingkeras.py` 는 기본값이 (1000, 0, 0.9) 였고, `kingkeras.py` 는 기본값을 `None` 으로 두어 필요한 값만 넘긴다
- `None` 이면 요청에 넣지 않아 서버(모델) 기본 설정을 따른다

### 5. 응답 시간 5 번 측정 (f12_latency)

```python
times = []
for i in range(5):
    start = time.perf_counter()
    response = llm.invoke(prompt)
    end = time.perf_counter()
    elasped = end - start
    times.append(elasped)
    print(i + 1, "회: ", round(elasped, 2), "초")

average = sum(times) / len(times)
```

- max_tokens=500 → 5.37 · 5.4 · 5.35 · 5.42 · 5.36 초, 평균 5.38 초 (회차마다 거의 일정)

### 6. 스트리밍 · TTFC (f13_stream · f14_1 ~ f14_3)

TTFC(Time To First Chunk) : 요청 후 내용이 있는 첫 조각이 도착할 때까지의 시간 → 사용자가 첫 글자를 보기까지 기다리는 시간

```python
# f14_1 : 끝까지 출력하면서 첫 내용 chunk 시간만 한 번 기록
start = time.perf_counter()
for chunk in llm.stream(prompt):
    if chunk.content:
        if first_time is None:
            first_time = time.perf_counter() - start
        print(chunk.content, end="", flush=True)

# f14_2 : 첫 내용 chunk 에서 시간을 재고 break
# f14_3 : for 문 없이 첫 chunk 하나만 꺼낸다
chunk = next(iter(llm.stream(prompt)))
first_time = time.perf_counter() - start
```

| 파일 | 방법 | 결과 |
|---|---|---|
| f14_1 | content 가 있는 첫 chunk 시간 기록 + 나머지 답변 출력 | 0.03 초 |
| f14_2 | content 가 있는 첫 chunk 에서 `break` | 0.03 초 |
| f14_3 | `next(iter(...))` 로 첫 chunk (content 검사 없음) | 0.03 초 |

- 전체 답변(500 토큰)은 약 5.4 초인데 첫 글자는 0.03 초 만에 나온다 → 스트리밍이 체감 속도를 크게 줄인다
- `iter()` 는 반복 가능한 객체를 반복자로, `next()` 는 반복자에서 값 하나를 꺼낸다
- f14_3 은 content 를 검사하지 않아 첫 chunk 가 빈 chunk(역할 정보) 여도 그 시간을 잰다

### 7. 질문별 응답 시간 · 토큰 수 (f15_prefill_decode · f16_usage_metadata · f17_tps · f18_compare)

```python
response = llm.invoke("NPU란 무엇인가?")
print(response.usage_metadata)
# {'input_tokens': 15, 'output_tokens': 500, 'total_tokens': 515,
#  'input_token_details': {'audio': 0, 'cache_read': 14}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

tokens = response.usage_metadata["output_tokens"]
tps = tokens / elapsed
```

| 질문 | 입력 토큰 | 출력 토큰 | 응답 시간 |
|---|:---:|:---:|:---:|
| NPU란? | 12 | 888 | 9.67 초 |
| NPU와 GPU의 차이를 5가지 설명해줘. | 24 | 1353 | 14.68 초 |
| NPU의 구조와 장점, 활용 분야를 자세히 설명해줘. | 28 | 2000 | 21.81 초 |

- LLM 추론 = prefill(입력 프롬프트를 한 번에 처리) + decode(출력 토큰을 하나씩 생성)
- 입력은 12 ~ 28 토큰으로 비슷하고 출력 토큰 수에 따라 시간이 늘어난다 → 응답 시간은 대부분 decode(출력 길이) 가 결정한다 (초당 약 92 토큰)
- 마지막 질문은 출력 2000 = max_tokens 에 걸려 답이 잘렸다
- f17 : 출력 1000 토큰 / 10.95 초 = TPS 91.34 (Day45 f06 은 max_tokens 를 출력 토큰으로 가정했고, 여기서는 실제 `output_tokens` 를 쓴다)
- `cache_read` : 이전 요청과 같은 앞부분(프롬프트 캐시) 을 다시 계산하지 않고 재사용한 토큰 수

### 8. 실습 폴더 정리 · 전체 코드 주석 정리

- `rngd/` → `furiosa01_rngd/` (Day42 ~ 45 f01 ~ f08), 오늘 실습은 `furiosa02_rngd/` (f11 ~ f18)
- `furiosa01_rngd/f08_pkg01_invoke.py` · `f08_pkg02_stream.py` 의 import 를 `from furiosa01.a_kingkeras import get_llm` (패키지.모듈) 형식으로 바꿨다
- 지금까지 올린 실습 파일 전체에 `[목적]` · `[흐름]` 머리 주석과 코드 해석 주석을 달았다 (코드는 그대로, 주석만 추가 · 수정)
- 베이스 파일에서 복사되어 실제 코드와 맞지 않던 주석을 바로잡았다
  - 파일 1 행 이름 · 설명이 베이스 파일 것으로 남아 있던 keras39 · 40 · 41 계열
  - train_size 와 맞지 않던 shape 주석 (digits 0.6 → (1078, 64) (719, 64), iris 0.7 → (105, 4) (45, 4), santander 0.7 → (140000, 200) (60000, 200))
  - validation_split 으로 떼어지는 실제 개수 (int(n × (1 - 비율)) 에서 자른다)
  - MaxPooling · GlobalAveragePooling 을 넣은 뒤에도 남아 있던 Total params (층 구성대로 다시 계산)
  - Jena 실습의 y 가 기온으로 바뀐 뒤에도 남아 있던 '풍향' 주석

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `ml/m10_HalvingGridSearchCV.py` | digits `HalvingGridSearchCV` + `XGBClassifier` → `min_resources` · `max_resources` · `factor` 설정별 단계 진행 비교 |
| `furiosa02_rngd/kingkeras.py` | `get_llm(max_tokens, temperature, top_p)` — 기본값 `None` 인 ChatOpenAI 생성 모듈 |
| `furiosa02_rngd/f11_basic.py` | `get_llm` 으로 Qwen3-8B-FP8 서버에 질문 하나 `invoke` |
| `furiosa02_rngd/f12_latency.py` | 같은 질문 5 번 → 회차별 응답 시간 · 평균 |
| `furiosa02_rngd/f13_stream.py` | `llm.stream` 스트리밍 출력 |
| `furiosa02_rngd/f14_1_ttfc.py` ~ `f14_3_ttfc.py` | 첫 chunk 까지 걸린 시간(TTFC) 측정 3 가지 방법 |
| `furiosa02_rngd/f15_prefill_decode.py` | 질문 3 개의 응답 시간 비교 |
| `furiosa02_rngd/f16_usage_metadata.py` | `usage_metadata` 로 입력 · 출력 · 전체 토큰 수 확인 |
| `furiosa02_rngd/f17_tps.py` | 실제 `output_tokens` / 걸린 시간 = TPS |
| `furiosa02_rngd/f18_compare.py` | 질문별 입력 토큰 · 출력 토큰 · 응답 시간 비교 |
| `furiosa01_rngd/` (f01 ~ f08, a_kingkeras.py) | `rngd/` 에서 폴더 이름 변경, f08 두 파일은 import 를 `furiosa01.a_kingkeras` 로 변경 (수정) |
| `keras1/` · `keras2/` · `RAG/` · `ml/` (전체) | `[목적]` 머리 주석 · 코드 해석 주석 추가, 실제 코드와 맞지 않던 주석 바로잡기 (수정) |
| `docs/Day47.md` | Day47 학습 기록 신규 작성 |
| `docs/Day46.md` | 하단 nav에 Day47 링크 추가 (수정) |
| `docs/Day42.md` · `Day43.md` · `Day45.md` | 학습 파일 경로 `rngd/` → `furiosa01_rngd/` (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(ml/ · furiosa01_rngd/ · furiosa02_rngd/ Day47) / 진행도(47일, 59%) 갱신 (수정) |

---

## 📊 실행 결과

| 파일 | 설정 | 결과 |
|---|---|---|
| m10 (digits) | min 100 · max 1437 · factor 3 | 3 단계 · 305 fits · best_score 0.9454 · test 0.9444 (210.39 초) |
| m10 (digits) | min 50 · max 1000 · factor 2 | 5 단계 · 415 fits · best_score 0.9335 · test 0.9417 (142.75 초) |
| m10 (digits) | min 50 · max 1000 · factor 2 (재실행) | 5 단계 · 415 fits · best_score 0.9473 · test 0.9444 (219.19 초) |
| f12 (Qwen3-8B, max_tokens 500) | 같은 질문 5 번 | 5.35 ~ 5.42 초, 평균 5.38 초 |
| f14_1 ~ f14_3 | TTFC | 0.03 초 |
| f15 (max_tokens 2000) | 질문 3 개 | 12.92 · 14.73 · 21.96 초 |
| f16 (max_tokens 500) | usage_metadata | 입력 15 · 출력 500 · 전체 515 (cache_read 14) |
| f17 (max_tokens 1000) | TPS | 출력 1000 토큰 / 10.95 초 = 91.34 tokens/s |
| f18 (max_tokens 2000) | 입력 · 출력 토큰 · 시간 | 12 · 888 · 9.67 초 / 24 · 1353 · 14.68 초 / 28 · 2000 · 21.81 초 |

---

## 💻 핵심 개념

### Grid · Random · Halving 탐색 비교

```
GridSearchCV(estimator, param_grid=parameters, cv=kfold)
  → 모든 조합 (42) × fold (5), 매번 x_train 전체로 훈련
RandomizedSearchCV(estimator, param_distributions=parameters, cv=kfold, n_iter=10)
  → 무작위 n_iter 조합 (10) × fold (5), 매번 x_train 전체로 훈련
HalvingGridSearchCV(estimator, param_grid=parameters, cv=kfold, min_resources, max_resources, factor)
  → 모든 조합으로 시작하되 적은 샘플로 걸러내고, 살아남은 후보만 샘플을 늘려 다시 훈련
```

### 단계 진행 공식

```
iter 0 : 후보 = 전체 조합 수           샘플 수 = min_resources
iter k : 후보 = 이전 후보 / factor (올림)  샘플 수 = min_resources × factor^k
멈춤   : 다음 샘플 수 > max_resources
```

### LLM 응답 시간 지표

```
latency (응답 시간) : 요청 ~ 답변 완료        → time.perf_counter() 를 invoke 앞뒤로
TTFC (첫 chunk 시간) : 요청 ~ 첫 내용 chunk   → llm.stream 에서 첫 chunk.content 가 올 때
TPS (초당 토큰 수)   : output_tokens / latency → response.usage_metadata["output_tokens"]
응답 시간 ≈ prefill(입력 처리, 짧음) + decode(출력 토큰 수 × 토큰당 시간)
```

---

## 성과

- HalvingGridSearchCV 의 import 방법(실험적 기능)과 `param_grid` 인자를 확인하고 digits 에 적용했다
- verbose 출력으로 단계마다 후보 수가 줄고 샘플 수가 늘어나는 과정을 확인했다
- `min_resources` · `max_resources` · `factor` 를 바꿔 단계 수 · 후보 수 · 샘플 수 · 총 훈련 횟수가 달라지는 것을 비교했다
- RNGD furiosa-llm 서버의 응답 시간 · TTFC · TPS 를 직접 재고, 출력 토큰 수가 응답 시간을 좌우하는 것을 확인했다
- 지금까지의 실습 파일 전체에 목적 · 코드 해석 주석을 달고, 실제 코드와 맞지 않던 주석을 바로잡았다

---

## 💡 주요 학습 포인트

1. **HalvingGridSearchCV 는 단계별 토너먼트**: 적은 샘플로 전체 후보를 거르고, 상위 후보만 샘플을 늘려 다시 평가
2. **실험적 기능**: `sklearn.model_selection` 에서 바로 import 불가 → `enable_halving_search_cv` 또는 정의된 모듈에서 import
3. **GridSearchCV 계열 인자**: `param_grid` 사용, `n_iter` 없음
4. **min_resources 기본값**: 분류에서는 2 × cv 개수 × 클래스 개수 (digits 는 100)
5. **max_resources 기본값**: x_train 행 수 (digits 는 1437)
6. **factor**: 후보는 1/factor, 샘플 수는 factor 배
7. **단계 수는 샘플 범위가 결정**: 다음 샘플 수가 max_resources 를 넘으면 멈추고 남은 후보 중 최고를 고른다
8. **실행마다 점수가 달라진다**: 단계별 샘플을 무작위로 뽑기 때문 (`random_state` 미지정)
9. **TTFC 와 latency 는 다르다**: 첫 글자는 0.03 초, 전체 답변은 수 초 → 스트리밍이 체감 속도를 줄인다
10. **응답 시간은 출력 토큰 수가 결정**: prefill 은 짧고 decode 가 출력 토큰 수만큼 걸린다
11. **TPS 는 실제 출력 토큰으로**: `usage_metadata["output_tokens"]` / 걸린 시간
12. **`next(iter(...))`**: 반복자에서 첫 값 하나만 꺼낸다

---

[⬅️ Day46](Day46.md) · [🏠 전체 목차](../README.md) · [Day48 ➡️](Day48.md)
