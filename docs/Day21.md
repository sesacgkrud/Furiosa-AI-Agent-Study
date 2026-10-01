# Day21 - Bidirectional RNN, Jena 기온 예측 비교, LangChain 입문 (API 키 관리 · PromptTemplate · LCEL · OutputParser)

**학습 기간:** 2026-09-30

> Day20에서는 return_sequences와 Flatten으로 RNN 출력을 다루고, Jena 기후 데이터로 다음 하루 144개를 예측했다. Day21은 RNN을 **정방향 + 역방향** 두 번 읽게 만드는 **Bidirectional**을 배웠다. 작은 시계열 두 개에 적용해 보고, 결과가 좋지 않았던 모델은 구조를 바꿔 80에 가깝게 개선했다. 그다음 Jena 데이터의 y를 **기온(`T (degC)`)**으로 바꿔 LSTM과 Bidirectional(LSTM)을 비교했다. 오후에는 **LangChain**으로 넘어가 OpenAI 모델을 코드에서 호출하고, **API 키를 안전하게 다루는 방법**(코드 직접 입력 → 환경 변수 → `.env`)과 **PromptTemplate → LCEL 체인 → StrOutputParser**까지 익혔다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| Bidirectional | RNN 층을 감싸서 정방향 + 역방향으로 두 번 읽고 출력을 이어 붙인다 |
| 출력 크기 | `units × 2` → `Bidirectional(SimpleRNN(10))` = `(None, 20)` |
| 파라미터 | RNN 파라미터 × 2 → SimpleRNN(10) 120 → **240** |
| 사용법 | `input_shape`는 바깥 Bidirectional에, `return_sequences`는 안쪽 RNN에 |
| model.add | `model.add()` 없이 만든 층은 모델에 들어가지 않는다 |
| 모델 개선 | Bi-SimpleRNN 2층 + Dropout → **Bi-LSTM(128) + Dense 5층** : 72 → 79.84 |
| Jena 기온 | y = `T (degC)`, LSTM RMSE 4.80 / Bidirectional RMSE **2.10** (시간 약 2배) |
| 두 지표 | 마지막 하루 RMSE와 test 전체 loss를 같이 본다 |
| ChatOpenAI | `llm.invoke('질문')` → `response.content`가 답변 글 |
| API 키 관리 | 코드 직접 입력 ✗ → 환경 변수 → **`.env` + `load_dotenv()`** |
| 키 확인 | `os.getenv()`로 읽고 앞 8자 + 뒤 4자만 출력 |
| 다른 API 서버 | `api_key` + **`base_url`**을 함께 넣는다, `.strip()`으로 공백 제거 |
| PromptTemplate | `'{country}의 수도는?'` → `{변수}` 자리에 값을 넣는 틀 |
| LCEL | `chain = prompt \| model \| output_parser` |
| StrOutputParser | 응답 객체 → 문자열. `.content` 없이 바로 출력 |

---

## 📖 핵심 학습 내용

### 1. Bidirectional 입문 (keras59_Bidirectional1)

keras54_RNN1 코드를 베이스로 `SimpleRNN`을 `Bidirectional`로 감쌌다.

```python
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])
x = x.reshape(x.shape[0], x.shape[1], 1)                      # (7, 3, 1)

model.add(Bidirectional(SimpleRNN(10), input_shape=(3, 1)))   # (None, 3, 1) → (None, 20)
model.add(Dense(512, activation='relu'))
...
model.add(Dense(1))
```

- **정방향** : 1 → 2 → 3 순서로 읽는다 (기존 RNN)
- **역방향** : 3 → 2 → 1 순서로 거꾸로 읽는다
- 두 방향의 마지막 출력을 **이어 붙인다(concat)** → 출력 크기 `units × 2`

| 층 | Output Shape | Param |
|---|:---:|:---:|
| `Bidirectional(SimpleRNN(10))` | `(None, 20)` | **240** |
| `Dense(512)` | `(None, 512)` | 10752 (`20 × 512 + 512`) |
| ... | | |
| **Total** | | **191,405** |

- SimpleRNN(10) 하나 = `10 × (1 + 10 + 1)` = 120 → 정방향 / 역방향이 가중치를 따로 가져서 **× 2 = 240**
- 뒤 Dense는 입력이 20으로 늘어서 파라미터도 늘어난다
- 결과 (정답 11) : SimpleRNN **10.846822** / Bidirectional **10.700574**
  - 훈련 y가 4 ~ 10뿐이라 11은 본 적 없는 범위 밖 값이다

### 2. Bidirectional 모델 구성 개선 (keras59_Bidirectional2)

간격이 큰 시계열(keras55_LSTM2_scale 데이터)로 `[50, 60, 70]` → **80**을 맞힌다 (목표 79.5).

**수정 전 모델**

```python
model.add(Bidirectional(SimpleRNN(128, return_sequences=True), input_shape=(3, 1)))

Bidirectional(SimpleRNN(64)),
model.add(Dense(32, activation='relu'))
Dropout(0.3),
model.add(Dense(1))
```

| 문제 | 설명 |
|---|---|
| `model.add()` 없음 | `Bidirectional(SimpleRNN(64)),` / `Dropout(0.3),`는 층만 만들고 모델에 들어가지 않았다 |
| 출력 모양 | `return_sequences=True`의 3차원 `(None, 3, 256)`이 그대로 Dense로 → 최종 출력 **`(None, 3, 1)`** |
| 예측 | 80 하나가 아니라 `[[[72.0] ...]]`처럼 시점마다 3개가 나왔다 |
| SimpleRNN 한계 | `model.add`로 넣어도 SimpleRNN 2층은 약 70에서 멈췄다. 게이트가 없어서 +10씩 크게 뛰는 규칙을 잘 못 잡는다 |
| Dropout | 데이터가 13개뿐이라 오히려 배울 것을 버린다 |

기존 코드는 지우지 않고 주석 처리한 뒤 **`[주석 이유]`**를 적었다.

**수정 후 모델**

```python
model.add(Bidirectional(LSTM(128), input_shape=(3, 1)))  # (None, 3, 1) → (None, 256)
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
# Bidirectional 파라미터 = 4 × 128 × (1 + 128 + 1) × 2 = 133120
```

- **SimpleRNN → LSTM** : 게이트가 +1 규칙과 +10 규칙을 같이 잡는다
- RNN은 1층, `return_sequences`는 기본값(False) → `(None, 256)` 2차원이 바로 Dense로
- 정방향 128 + 역방향 128 = 256개 특징을 Dense로 점점 줄여 값 1개로 모은다

- 결과 : 수정 전 loss 0.00237 / **[[[72.0553] ...]]** → 수정 후 loss 0.00082 / **[[79.8367]]**
  - 수정 후 3번 실행 : 79.84 / 79.40 / 78.98

### 3. Jena Climate 기온 예측 - LSTM vs Bidirectional (keras59_Bidirectional3_jena)

keras58_kaggle_jena2를 베이스로, y를 **`T (degC)`(기온)**로 저장한 npy를 불러와 첫 층만 바꿨다.

```python
# model.add(LSTM(64, input_shape=(144, 13)))                 # (None, 144, 13) → (None, 64)
model.add(Bidirectional(LSTM(64), input_shape=(size, 13)))   # (None, 144, 13) → (None, 128)
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(size))                                       # 다음 하루 144개 기온
```

| 층 | Output Shape | Param |
|---|:---:|:---:|
| `Bidirectional(LSTM(64))` | `(None, 128)` | 39936 |
| `Dense(128)` | `(None, 128)` | 16512 |
| `Dense(64)` | `(None, 64)` | 8256 |
| `Dense(144)` | `(None, 144)` | 9360 |
| **Total** | | **74,064** |

- LSTM(64) 하나 = `4 × 64 × (13 + 64 + 1)` = 19968 → **× 2 = 39936**
- 뒤 `Dense(128)`도 입력이 128이라 `128 × 128 + 128` = 16512 (LSTM만 쓸 때는 `64 × 128 + 128` = 8320)
- y(기온)는 스케일링하지 않아서 RMSE가 바로 **섭씨(℃)** 단위다

| 모델 | test loss (mse) | 마지막 하루 기온 RMSE | 걸린 시간 |
|---|:---:|:---:|:---:|
| `LSTM(64)` | **3.2113** | 4.8036 | 484.4초 |
| `Bidirectional(LSTM(64))` | 4.1010 | **2.1046** | 919.26초 |

- 마지막 하루(144개) 기온 RMSE는 Bidirectional이 더 좋았다 (4.80 → 2.10도)
- test 전체 loss는 LSTM이 더 낮았다
  - 144개 한 구간 점수는 그날 날씨에 따라 흔들리고, test loss는 8만 개 묶음 전체의 평균이다 → **두 지표를 같이 본다**
- 걸린 시간은 약 **2배** → Bidirectional은 LSTM을 두 개 쓰는 것과 같다
- Bidirectional은 문장처럼 앞뒤 문맥이 모두 중요한 데이터에서 효과가 크다. 시간 대비 성능을 같이 보고 고른다

### 4. LangChain으로 OpenAI 모델 호출 - API 키 직접 입력 (rag01_key_insert)

```python
from langchain_openai import ChatOpenAI

openai_api_key = "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"   # 실제 키 대신 예시 값
llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    openai_api_key=openai_api_key,
)

response = llm.invoke('안녕하세요.')
print(response)           # 응답 객체 전체
print(response.content)   # 답변 글만
```

- `ChatOpenAI` : LangChain에서 OpenAI 채팅 모델을 쓰는 클래스
- `temperature=0` : 답변을 가장 일정하게 (값이 클수록 다양하게)
- `invoke()` : 모델에 질문을 보내고 응답을 받는다
- 응답 객체 구조

```
content='안녕하세요! 무엇을 도와드릴까요?'                     ← 답변 글
response_metadata={'token_usage': {'completion_tokens': 14,    ← 답변 토큰
                                   'prompt_tokens': 9,          ← 질문 토큰
                                   'total_tokens': 23}, ...
                   'model_name': 'gpt-5.6-terra', 'finish_reason': 'stop'}
usage_metadata={'input_tokens': 9, 'output_tokens': 14, 'total_tokens': 23, ...}
```

- `'안녕 나는 하경이야'` → `'내 이름이 뭐게'`를 이어서 `invoke` 해 봤다
- 키를 코드에 직접 적으면 GitHub에 올릴 때 키가 그대로 노출된다 → **업로드한 파일의 키는 예시 값으로 바꿨다**

### 5. 환경 변수로 API 키 넣기 (rag02_environ01, rag03_env_check, rag04_key_check)

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"   # 예시 값

llm = ChatOpenAI(model_name='gpt-5.6-terra', temperature=0)   # openai_api_key 인자 없이
```

- `ChatOpenAI`는 `openai_api_key`를 안 주면 환경 변수 **`OPENAI_API_KEY`**를 알아서 찾는다
- rag03 : 코드의 `os.environ` 줄도 주석 처리 → 컴퓨터(시스템)에 등록된 환경 변수만으로 호출
- rag04 : 환경 변수에 키가 들어 있는지 확인

```python
key = os.getenv('OPENAI_API_KEY')
if key is None:
    print('OPENAI_API_KEY 없음')
else:
    print('키 길이 :', len(key))
    print('키 확인 :', key[:8] + "..." + key[-4:])   # 앞 8자 + 뒤 4자만 보여준다
```

- `os.getenv()`는 없으면 `None`을 돌려준다 (`os.environ['...']`은 없으면 에러)
- 키 전체를 출력하지 않고 **앞뒤 일부만** 보여줘서 확인한다

### 6. .env 파일로 API 키 관리 (rag05_env)

```python
from dotenv import load_dotenv
load_dotenv()                    # 같은 폴더(상위 폴더)의 .env 파일을 읽어 환경 변수로 등록

llm = ChatOpenAI(model_name='gpt-5.6-terra', temperature=0)
response = llm.invoke('openai api key 안전하게 보관하는 방법 알려줘')
print(response.content)
```

```
# .env  (키=값 형식, 따옴표 / 공백 없이)
OPENAI_API_KEY=sk-proj-...
```

| 방법 | 키 위치 | GitHub 노출 |
|---|---|:---:|
| 코드에 직접 입력 (rag01) | `.py` 파일 안 | ⚠️ 그대로 노출 |
| `os.environ[...] = ...` (rag02) | `.py` 파일 안 | ⚠️ 그대로 노출 |
| 시스템 환경 변수 (rag03) | 컴퓨터 설정 | 안전 |
| **`.env` + `load_dotenv()`** (rag05) | `.env` 파일 | **`.gitignore`에 넣으면 안전** |

- `.env`는 `.gitignore`에 추가해서 GitHub에 올리지 않는다

### 7. OpenAI 호환 API 서버 사용 - MonoRouter (rag06_monogpt)

```python
api_key = os.environ['MONOROUTER_API_KEY'].strip()   # strip() : 앞뒤 공백 / 줄바꿈 제거
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url,    # OpenAI 가 아닌 다른 서버의 키는 key 와 URL 을 함께 넣어야 한다
)
```

- OpenAI 공식 서버가 아닌 **OpenAI 호환 서버**의 키를 쓸 때는 `base_url`로 주소를 바꿔 준다
  - `base_url` 없이 넣으면 OpenAI 서버로 가서 `invalid_api_key` 에러가 난다
- `.env`에서 키를 복사할 때 섞인 공백 / 줄바꿈 때문에 생기는 에러를 `.strip()`으로 막는다

### 8. PromptTemplate (rag07_prompt)

```python
from langchain_core.prompts import PromptTemplate

template = '{country}의 수도는 어디인가요?'
prompt_template = PromptTemplate.from_template(template)
print(prompt_template)
# input_variables=['country'] input_types={} partial_variables={} template='{country}의 수도는 어디인가요?'
```

- `{변수}` 자리를 비워 둔 **질문 틀**. `from_template`이 `{}` 안의 이름을 찾아 `input_variables`로 만든다
- 흐름 : **pmo** (prompt - model - output) / **plp** (prompt - llm - parser)

### 9. LCEL 체인 - prompt | model (rag08_LCEL01, rag08_LCEL02)

```python
# LCEL = LangChain Expression Language
# Chain = prompt | model | output_parser

prompt = PromptTemplate.from_template('{topic}에 대해 {how} 설명해주세요.')
model = ChatOpenAI(model_name='gpt-5.6-terra', temperature=0, api_key=api_key, base_url=base_url)

chain = prompt | model                    # 파이프(|) 로 단계를 이어 붙인다

input = {'topic' : '양자 컴퓨터 학습 원리', 'how' : '초등학생도 이해하기 쉽게'}
response = chain.invoke(input)
print(response.content)
```

- `|` : 앞 단계의 출력이 다음 단계의 입력으로 넘어간다 → 프롬프트 완성 → 모델 호출
- 변수가 여러 개면 **딕셔너리 `{'변수명': 값}`**으로 한 번에 넘긴다
- LCEL01 : 변수 1개(`{topic}`) / LCEL02 : 변수 2개(`{topic}`, `{how}`)

### 10. StrOutputParser (rag09_out_parser01, rag09_out_parser02)

```python
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

chain = prompt | model | output_parser

response = chain.invoke(input)
# print(response.content)   # output_parser 가 있으면 필요 없음. '.content' 를 하면 에러 발생
print(response)
```

- parser : 모델의 응답을 분석해서 원하는 형태로 다듬는다
- `StrOutputParser` : 응답 객체 → **문자열(str)**. 결과가 이미 문자열이라 `.content`를 붙이면 에러
- rag09_out_parser02 : 여러 줄 템플릿으로 **역할 + 상황 + 출력 양식**을 지정

```python
template = '''
당신의 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화에 맞는 영어 회화를 작성해 주세요.
양식은 [FORMAT]을 참고하여 작성해 주세요.

# 상황:
{question}

# FORMAT:
- 영어회화 :
- 한글번역 :
'''
prompt = PromptTemplate.from_template(template=template)
```

- 상황 : `'저는 부산에서 물밀면을 먹고 싶어요.'`

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras1/keras59_Bidirectional1.py` | `Bidirectional(SimpleRNN(10))` 파라미터 240, `(None, 20)`, 11 예측 10.85 → 10.70 |
| `keras1/keras59_Bidirectional2.py` | 간격이 큰 시계열에 Bidirectional 적용, 모델 구성 개선 (`[주석 이유]`), 80 예측 72.06 → 79.84 |
| `keras1/keras59_Bidirectional3_jena.py` | Jena 기온 예측, `Bidirectional(LSTM(64))` 파라미터 39936, LSTM과 test loss / RMSE / 시간 비교 |
| `RAG/rag01_key_insert.py` | `ChatOpenAI`에 API 키 직접 입력, `invoke`, 응답 객체 구조 (키는 예시 값으로 변경) |
| `RAG/rag02_environ01.py` | `os.environ["OPENAI_API_KEY"]`로 키 등록 후 호출 (키는 예시 값으로 변경) |
| `RAG/rag03_env_check.py` | 코드의 키 줄을 주석 처리하고 시스템 환경 변수로만 호출 |
| `RAG/rag04_key_check.py` | `os.getenv`로 키 존재 확인, 앞 8자 + 뒤 4자만 출력 |
| `RAG/rag05_env.py` | `.env` + `load_dotenv()`로 키 불러와 호출 |
| `RAG/rag06_monogpt.py` | MonoRouter 키 + `base_url`, `.strip()` |
| `RAG/rag07_prompt.py` | `PromptTemplate.from_template`, `input_variables` 확인 |
| `RAG/rag08_LCEL01.py` | `chain = prompt \| model`, 변수 1개 |
| `RAG/rag08_LCEL02.py` | 변수 2개(`topic`, `how`)를 딕셔너리로 넘기기 |
| `RAG/rag09_out_parser01.py` | `prompt \| model \| StrOutputParser()`, `.content` 없이 출력 |
| `RAG/rag09_out_parser02.py` | 역할 + 상황 + FORMAT 여러 줄 템플릿 (영어 회화 선생님) |
| `docs/Day21.md` | Day21 학습 기록 신규 작성 |
| `docs/Day20.md` | 하단 nav에 Day21 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(RAG 폴더 추가) / 진행도(21일, 26%) 갱신 (수정) |
| `.gitignore` | `.env` 추가 (수정) |

---

## 📊 실행 결과

### Bidirectional 적용

| 파일 | 입력 | 정답 | 적용 전 | 적용 후 |
|---|---|:---:|:---:|:---:|
| `keras59_Bidirectional1.py` | `[8, 9, 10]` | 11 | 10.846822 (SimpleRNN) | 10.700574 (Bidirectional) |
| `keras59_Bidirectional2.py` | `[50, 60, 70]` | 80 | 72.0553 (수정 전, 출력 3개) | **79.8367** (Bi-LSTM(128) + Dense) |

### Jena Climate 기온 (y = `T (degC)`)

| 모델 | Param (첫 층) | test loss | 기온 RMSE | 걸린 시간 |
|---|:---:|:---:|:---:|:---:|
| `LSTM(64)` | 19968 | **3.2113** | 4.8036 | 484.4초 |
| `Bidirectional(LSTM(64))` | 39936 | 4.1010 | **2.1046** | 919.26초 |

**읽는 법**
- 파라미터와 시간이 약 2배 → Bidirectional = RNN 두 개
- 한 구간 RMSE와 전체 test loss의 승자가 달랐다 → 한 지표만 보고 결론 내리지 않는다

### LangChain 응답 (rag01, rag02)

| 질문 | content | 토큰 (질문 / 답변 / 전체) |
|---|---|:---:|
| `'안녕하세요.'` | `'안녕하세요! 무엇을 도와드릴까요?'` | 9 / 14 / 23 |

---

## 💻 핵심 개념

### Bidirectional

```python
model.add(Bidirectional(SimpleRNN(10), input_shape=(3, 1)))   # (None, 20)
model.add(Bidirectional(LSTM(64), input_shape=(144, 13)))      # (None, 128)
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(144, 13)))   # (None, 144, 128)
```

```
출력 크기 = units × 2               (정방향 + 역방향 이어 붙이기)
파라미터  = 안쪽 RNN 파라미터 × 2    (방향마다 가중치를 따로 가진다)

SimpleRNN(10), feature=1  : 10 × (1 + 10 + 1) × 2           = 240
LSTM(64),      feature=13 : 4 × 64 × (13 + 64 + 1) × 2      = 39936
LSTM(128),     feature=1  : 4 × 128 × (1 + 128 + 1) × 2     = 133120
```

### model.add 는 꼭 붙인다

```python
Bidirectional(SimpleRNN(64)),               # 층만 만들고 버려진다 → 모델에 없음
model.add(Bidirectional(SimpleRNN(64)))     # 모델에 들어간다
```

### API 키 관리 순서

```python
# 1. 코드에 직접 (✗ GitHub 노출)
ChatOpenAI(openai_api_key="sk-...")

# 2. 코드에서 환경 변수 등록 (✗ 여전히 코드에 키가 있다)
os.environ["OPENAI_API_KEY"] = "sk-..."

# 3. .env 파일 + load_dotenv (✓ .env 는 .gitignore)
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()

# 확인할 때도 전체 출력 금지
print(key[:8] + "..." + key[-4:])
```

### LangChain 체인

```python
prompt = PromptTemplate.from_template('{topic}에 대해 {how} 설명해주세요.')
model = ChatOpenAI(model_name='gpt-5.6-terra', temperature=0, api_key=api_key, base_url=base_url)
output_parser = StrOutputParser()

chain = prompt | model | output_parser
response = chain.invoke({'topic': '...', 'how': '...'})   # 문자열
print(response)
```

```
PromptTemplate  →  ChatOpenAI  →  StrOutputParser
 질문 틀 완성       모델 호출        응답 객체 → 문자열
```

---

## 성과

- `Bidirectional`로 SimpleRNN을 감싸 출력 `(None, 20)`, 파라미터 240(= 120 × 2)을 summary로 확인
- Bidirectional을 적용한 간격이 큰 시계열 모델에서 `model.add()` 누락과 `(None, 3, 1)` 출력 문제를 찾고, 기존 코드는 `[주석 이유]`와 함께 주석으로 남긴 채 Bi-LSTM(128) + Dense 5층으로 개선 : 72.06 → 79.84
- Jena 기후 데이터 y를 기온으로 바꿔 LSTM과 Bidirectional(LSTM)을 비교 : RMSE 4.80 → 2.10, 시간 484 → 919초, test loss 3.21 vs 4.10
- LangChain `ChatOpenAI`로 모델을 호출하고 응답 객체(`content`, 토큰 사용량) 구조 확인
- API 키를 코드 직접 입력 → 환경 변수 → `.env` + `load_dotenv()` 순서로 옮기며 안전한 관리 방법 실습, 키 확인도 앞뒤 일부만 출력
- `base_url`로 OpenAI 호환 서버(MonoRouter) 연결
- PromptTemplate → LCEL 체인(`prompt | model`) → StrOutputParser까지 이어 붙이고, 역할 + 상황 + 양식이 들어간 여러 줄 템플릿 작성
- GitHub에 올리는 파일의 실제 API 키를 예시 값으로 바꾸고 `.env`를 `.gitignore`에 추가

---

## 💡 주요 학습 포인트

1. **Bidirectional은 RNN을 감싸는 층이다**: 혼자 쓰지 않고 SimpleRNN / LSTM / GRU를 안에 넣는다
2. **출력은 units × 2**: 정방향과 역방향 출력을 이어 붙인다
3. **파라미터도 × 2**: 두 방향이 가중치를 따로 가진다 → 120 → 240
4. **input_shape는 바깥, return_sequences는 안쪽**: `Bidirectional(LSTM(64, return_sequences=True), input_shape=...)`
5. **`model.add()` 없이 만든 층은 모델에 없다**: 끝에 쉼표를 붙여도 에러 없이 조용히 버려진다
6. **return_sequences=True 뒤에 바로 Dense를 두면 출력이 3차원**: `(None, 3, 1)` → 시점마다 예측이 나온다
7. **작은 데이터에 Dropout은 손해**: 13개 데이터에서는 배울 것을 버린다
8. **SimpleRNN보다 LSTM**: 게이트가 크기가 다른 규칙(+1, +10)을 같이 잡는다
9. **결과는 여러 번 돌려 확인한다**: 수정 후 3번 실행 79.84 / 79.40 / 78.98
10. **Bidirectional은 시간이 약 2배**: LSTM을 두 개 돌리는 것과 같다
11. **지표 하나만 보지 않는다**: 마지막 하루 RMSE는 Bidirectional, test loss는 LSTM이 좋았다
12. **Bidirectional은 앞뒤 문맥이 중요한 데이터에 강하다**: 과거로 미래를 맞추는 시계열에서는 항상 좋아지지 않는다
13. **`invoke()`의 결과는 응답 객체**: 답변 글은 `.content`, 토큰 사용량은 `response_metadata`
14. **temperature=0은 가장 일정한 답변**: 값이 클수록 다양해진다
15. **API 키는 코드에 적지 않는다**: GitHub에 올리면 그대로 노출된다
16. **ChatOpenAI는 `OPENAI_API_KEY` 환경 변수를 자동으로 찾는다**: 인자로 안 넘겨도 된다
17. **`.env` + `load_dotenv()`**: 키는 `.env`에, `.env`는 `.gitignore`에
18. **`os.getenv`는 없으면 None, `os.environ[]`은 없으면 에러**
19. **키 확인은 앞뒤 일부만**: `key[:8] + "..." + key[-4:]`
20. **다른 서버의 키는 `base_url`과 함께**: 없으면 OpenAI 서버로 가서 `invalid_api_key`
21. **`.strip()`으로 키의 공백 / 줄바꿈 제거**: 복사하다 섞인 공백 에러를 막는다
22. **PromptTemplate은 `{변수}`가 있는 질문 틀**: `input_variables`로 변수 이름을 확인한다
23. **LCEL은 `|`로 단계를 잇는다**: `prompt | model | output_parser`
24. **변수가 여러 개면 딕셔너리로 넘긴다**: `{'topic': ..., 'how': ...}`
25. **StrOutputParser를 쓰면 결과가 문자열**: `.content`를 붙이면 에러
26. **템플릿에 역할 · 상황 · 양식을 넣으면 답변 형태를 정할 수 있다**

---

[⬅️ Day20](Day20.md) · [🏠 전체 목차](../README.md) · [Day22 ➡️](Day22.md)
