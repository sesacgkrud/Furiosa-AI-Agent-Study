# Day22 - Tokenizer와 One-Hot, 텍스트 감정 분류 (DNN · LSTM · One-Hot LSTM), Embedding 층, OpenAI 임베딩

**학습 기간:** 2026-10-01

> Day21에서는 Bidirectional RNN과 LangChain(PromptTemplate · LCEL · OutputParser)을 배웠다. Day22는 **자연어(글자)를 숫자로 바꾸는 방법**을 배웠다. 먼저 **Tokenizer**로 문장을 단어 번호로 바꾸고, 그 번호를 **One-Hot 3가지**(to_categorical · get_dummies · OneHotEncoder)로 바꿔 봤다. 이어서 문장 15개로 긍정 / 부정을 분류하면서 단어 표현 방법을 한 단계씩 바꿨다. **① 번호 그대로 DNN → ② 번호 그대로 LSTM → ③ One-Hot + LSTM → ④ Embedding 층** 순서로 진행하며 Embedding이 왜 필요한지 확인했다. 오후에는 LangChain의 **OpenAIEmbeddings**로 문장을 1536 / 3072 차원 벡터로 바꾸고, `dimensions`로 차원을 줄여 봤다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| Tokenizer | `fit_on_texts`로 단어 사전을 만들고 `texts_to_sequences`로 문장을 번호 리스트로 바꾼다 |
| 번호 규칙 | 많이 나온 단어부터 1번, 횟수가 같으면 먼저 나온 단어가 앞 번호, **0번은 비워 둠 (padding용)** |
| One-Hot 3가지 | `to_categorical`(0번 열까지 생김 → `[:, 1:]`) / `get_dummies`(1차원) / `OneHotEncoder`(2차원 `(n, 1)`) |
| 문장 2개 | 공통 단어 사전, 길이가 달라서 `np.concatenate`로 이어 붙임 → (24, 16) |
| pad_sequences | 문장 길이를 `maxlen=5`로 맞춤, `padding='post'` = 뒤를 0으로 채움 → (15, 5) |
| 01 DNN | 번호 그대로 `(15, 5)` + MinMaxScaler + Dense → 순서도, 단어 뜻도 반영 못 함 |
| 02 LSTM | `reshape(15, 5, 1)` → 순서는 보지만 단어가 여전히 숫자 1개 |
| 03 One-Hot LSTM | `(15, 5, 31)` → 단어끼리 크기 차이 없음, 대신 단어 수만큼 칸이 늘어남 |
| 04 Embedding | 번호를 그대로 받아 **학습되는 벡터**로 바꿈 → `(15, 5)` → `(15, 5, 100)` |
| Embedding 파라미터 | `input_dim × output_dim` = 32 × 100 = **3200** (bias 없음) |
| input_dim | 단어 수 + 1 (padding 0번) = **32** |
| OpenAIEmbeddings | `embed_query(문장)` → 벡터 (small **1536** / large **3072** 차원) |
| dimensions | `dimensions=5` → 벡터 길이 **5** |

---

## 📖 핵심 학습 내용

### 1. Tokenizer - 문장을 단어 번호로 (keras60_Tokenizer1)

```python
from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'

token = Tokenizer()              # 단어 사전을 만들고 문장을 번호로 바꿔 주는 도구
token.fit_on_texts([text])       # 단어 사전 만들기 (리스트로 넣는다)

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ..., ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text])
print(x)   # [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
```

- **번호 규칙** : 많이 나온 단어부터 1번 (마구 4번 → 1). 횟수가 같으면 먼저 나온 단어가 앞 번호. 0번은 비워 둔다
- 띄어쓰기로 자르고, 마침표 같은 문장부호는 자동으로 지운다 (`'먹었다.'` → `'먹었다'`)
- 번호는 **이름표**일 뿐 크기에 의미가 없다 (진짜 = 2가 마구 = 1의 2배가 아니다) → One-Hot이 필요하다

**One-Hot 3가지** (단어 14개, 단어 종류 9개)

```python
x = np.array(x).reshape(-1,)                 # [[...]] 2차원 → (14,)

# 1. tensorflow
x1 = to_categorical(x)                       # (14, 10) → 0번 열까지 생김
x1 = x1[:, 1:]                               # (14, 9)

# 2. pandas
x2 = pd.get_dummies(x, dtype=int)            # 1차원을 넣는다 → (14, 9)

# 3. sklearn
ohe = OneHotEncoder(sparse_output=False)
x3 = ohe.fit_transform(x.reshape(-1, 1))     # 2차원 (14, 1)을 넣는다 → (14, 9)
```

| 방법 | 입력 | 결과 | 특징 |
|---|---|:---:|---|
| `to_categorical` | 1차원 | (14, 10) → (14, 9) | 가장 큰 번호를 보고 0 ~ 9 = 10칸을 만든다 → 0번 열을 지운다 |
| `pd.get_dummies` | 1차원 | (14, 9) | 실제로 나온 값만 열로 만든다, `dtype=int` → 1 / 0 |
| `OneHotEncoder` | 2차원 `(n, 1)` | (14, 9) | `sparse_output=False` → 바로 볼 수 있는 배열 |

### 2. 문장 2개를 이어 붙여 One-Hot (keras60_Tokenizer2)

```python
text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '개똥이는 기관사를 좋아한다. 말똥이는 잘생겼다. 길동이는 마구 마구 더 잘생겼다.'

token.fit_on_texts([text1, text2])           # 두 문장으로 공통 단어 사전
# {'마구': 1, '진짜': 2, '매우': 3, '잘생겼다': 4, '나는': 5, ... '길동이는': 15, '더': 16}

x = token.texts_to_sequences([text1, text2])
# [[5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10], [11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]

x = np.concatenate(x)                        # 길이가 14, 10으로 달라서 이어 붙임 → (24,)
```

- 두 문장에 같이 나온 `'마구'`(합계 6번)는 1번, `'잘생겼다'`(2번)는 4번
- 세 방법 모두 **(24, 16)** (`to_categorical`은 (24, 17) → 0번 열 제거)

### 3. 텍스트 감정 분류 데이터 준비 - padding

```python
docs = [
    '너무 재미있다', '참 최고에요', '참 잘 만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '벌로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밌네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다',
]
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])   # 1 = 긍정 (7개), 0 = 부정 (8개)

token = Tokenizer()
token.fit_on_texts(docs)                              # 단어 31개 (번호 1 ~ 31)
x = token.texts_to_sequences(docs)
# [[2, 3], [1, 4], [1, 5, 6, 7], ..., [25, 26], [27, 28], [29, 30, 31]]  ← 길이가 1 ~ 5개로 제각각

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x, padding='post', maxlen=5, truncating='post')
print(padded_x.shape)                                 # (15, 5)
# [[ 2  3  0  0  0] [ 1  4  0  0  0] ...]
```

| 옵션 | 값 | 의미 |
|---|---|---|
| `padding` | `'post'` | 뒤를 0으로 채운다 (기본값 `'pre'` = 앞을 0으로) |
| `maxlen` | `5` | 문장 길이를 5로 맞춘다, 더 긴 문장은 잘린다 |
| `truncating` | `'post'` | 자를 때 뒤쪽을 자른다 (기본값 `'pre'` = 앞쪽) |

- 새 문장 **`'개똥이 잘생겼다'`**도 같은 `token`으로 `texts_to_sequences` → `pad_sequences` → `[[25 28 0 0 0]]`
  - `fit_on_texts`는 다시 하지 않는다 (다시 하면 단어 번호가 바뀐다)

### 4. 번호 그대로 DNN으로 분류 (keras61_Embedding01_DNN)

```python
x_train, x_test, y_train, y_test = train_test_split(
    padded_x, labels, train_size=0.7, random_state=11, stratify=labels)

print(np.unique(y_train, return_counts=True))   # (array([0, 1]), array([5, 5]))
print(np.unique(y_test, return_counts=True))    # (array([0, 1]), array([3, 2]))

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))   # 단어 5개
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))               # 이진 분류

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=2, callbacks=[es])
```

- 15개 중 train 10개 / test 5개
- `np.unique(..., return_counts=True)` : `stratify`로 나눈 뒤 긍정 / 부정이 몇 개씩 들어갔는지 확인
  - 전체 부정 8 : 긍정 7 → train 5 : 5, test 3 : 2
  - stratify가 없으면 test가 한 라벨만 될 수 있다 → 무조건 0만 내도 acc 1.0
- 훈련 데이터가 10개뿐이라 validation을 떼지 않고 **`monitor='loss'`**로 감시
- `MinMaxScaler`는 **열(단어 위치)마다** min / max를 따로 구한다
  - 같은 `'개똥이'`(25)도 첫 번째 자리 0.857, 두 번째 자리 0.833처럼 위치마다 다른 값이 된다
- 결과 : loss 9.06e-09, acc **1.0**, 10.79초, `'개똥이 잘생겼다'` **0.9935**
  - 실행할 때마다 `'개똥이 잘생겼다'` 값이 크게 바뀐다 (예: 0.0126이 나온 적도 있다)
  - `[25, 28]`은 부정 `'개똥이 바보'` `[25, 26]`과 긍정 `'말똥이 잘생겼다'` `[27, 28]` 사이의 숫자 → 초기값에 따라 어느 쪽으로든 기운다
  - 이 문장은 정답 라벨이 없다 → 값은 "얼마나 긍정이라고 보는지"일 뿐 맞고 틀림을 판단할 수 없다

### 5. 번호 그대로 LSTM으로 분류 (keras61_Embedding02_LSTM)

```python
padded_x = padded_x.reshape(15, 5, 1)        # (데이터 수, timesteps, feature)

x_train, x_test, y_train, y_test = train_test_split(
    padded_x, labels, train_size=0.7, random_state=14, stratify=labels)   # (10, 5, 1) (5, 5, 1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, 5)).reshape(-1, 5, 1)
x_test = scaler.transform(x_test.reshape(-1, 5)).reshape(-1, 5, 1)

model.add(LSTM(64, input_shape=(5, 1)))      # 단어를 하나씩 5번 읽는다
```

- MinMaxScaler는 2차원만 받는다 → **`(10, 5, 1)` → `(10, 5)` → 스케일링 → `(10, 5, 1)`** 을 한 줄로
- x_train으로 구한 위치별 min / max

| 위치 | 1번째 | 2번째 | 3번째 | 4번째 | 5번째 |
|---|:---:|:---:|:---:|:---:|:---:|
| min | 1 | 0 | 0 | 0 | 0 |
| max | 29 | 30 | 31 | 14 | 15 |

- 예) train `'개똥이 바보'` `[25 26 0 0 0]` → `[0.857 0.867 0 0 0]`
- 예) test `'참 잘 만든 영화에요'` `[1 5 6 7 0]` → `[0 0.167 0.194 0.5 0]` (train 기준으로 `transform`만)
- LSTM은 **순서**를 반영하지만 단어가 여전히 **숫자 1개(번호)** → 번호 크기 문제는 그대로
- 결과 : acc_score **1.0**, 8.35초, `'개똥이 잘생겼다'` **0.7037** (실행마다 0.0x ~ 0.9x로 흔들림)

### 6. One-Hot + LSTM - 3가지 방법 (keras61_Embedding03)

단어 번호를 One-Hot으로 바꿔 `(15, 5, 31)` → 단어마다 자기 칸이 생겨서 번호 크기 문제가 사라진다.

**to_categorical** (keras61_Embedding03_ohe_to_categorical_LSTM)

```python
all_x = np.concatenate([padded_x, x_predict])   # (16, 5)  15문장 + 예측 문장
all_x = to_categorical(all_x)                   # (16, 5, 32)
all_x = all_x[:, :, 1:]                         # (16, 5, 31)  0번(padding) 칸 제거

padded_x = all_x[:15]                           # (15, 5, 31)
x_predict = all_x[15:]                          # (1, 5, 31)

model.add(LSTM(64, input_shape=(5, 31)))
```

- 예측 문장만 따로 `to_categorical`하면 가장 큰 번호가 28이라 **29칸**만 생긴다 (훈련 데이터는 32칸)
  - → 훈련 데이터와 **이어 붙여 한 번에** 바꾼 뒤 슬라이싱으로 다시 나눈다
- 0번 칸을 지우면 padding 자리는 **전부 0인 벡터**가 된다
- One-Hot은 이미 0과 1뿐이라 스케일링하지 않는다

**OneHotEncoder** (keras61_Embedding03_ohe_sklearn_LSTM)

```python
ohe = OneHotEncoder(sparse_output=False)
padded_x = padded_x.reshape(-1, 1)              # (75, 1)   15문장 × 5단어를 한 줄로
padded_x = ohe.fit_transform(padded_x)          # (75, 32)
padded_x = padded_x[:, 1:]                      # (75, 31)
padded_x = padded_x.reshape(15, 5, 31)          # (15, 5, 31)

x_predict = ohe.transform(x_predict.reshape(-1, 1))[:, 1:].reshape(1, 5, 31)   # transform 만
```

- fit 때 본 값으로 칸을 정해 두므로 예측 문장은 **`transform`만** 하면 같은 32칸 → 이어 붙일 필요가 없다

**get_dummies** (keras61_Embedding03_ohe_pandas_LSTM)

```python
all_x = np.concatenate([padded_x, x_predict])   # (16, 5)
all_x = all_x.reshape(-1)                       # (80,)     1차원만 받는다
all_x = pd.get_dummies(all_x, dtype=int)        # (80, 32)  열 이름 = 단어 번호 0 ~ 31
all_x = all_x.drop([0], axis=1)                 # (80, 31)  0번 열 제거
all_x = all_x.values                            # DataFrame → numpy
all_x = all_x.reshape(16, 5, 31)
```

- 실제로 나온 값만 열로 만든다 → 예측 문장만 넣으면 0, 25, 28 세 열만 생긴다 → 이어 붙여서 바꾼다

**결과** : test acc 0.4 ~ 0.8, `'개똥이 잘생겼다'` 대부분 0에 가까움 (부정)
- test 문장의 단어 대부분이 훈련 때 한 번도 안 나온 단어라 맞힐 근거가 없다
- `'개똥이'`가 훈련 데이터에서 부정 문장(`'개똥이 바보'`)에만 나와서 부정 신호로 배웠다

### 7. Embedding 층 (keras61_Embedding04_important)

```python
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(input_dim=32, output_dim=100, input_length=5))
        # 단어 사전 크기 (단어 31개 + padding 1), 출력 차원, 입력 시퀀스 길이
model.add(SimpleRNN(10))
model.add(Dense(1))
model.summary()
```

| 층 | Output Shape | Param |
|---|:---:|:---:|
| `Embedding(32, 100)` | `(None, 5, 100)` | **3200** (`32 × 100`) |
| `SimpleRNN(10)` | `(None, 10)` | 1110 (`10 × (10 + 100 + 1)`) |
| `Dense(1)` | `(None, 1)` | 11 |
| **Total** | | **4,321** |

- 입력은 **단어 번호 `(15, 5)` 그대로** → 스케일링, reshape, One-Hot이 필요 없다
- 출력 `(None, 5, 100)` : 단어 하나가 100칸짜리 벡터, 3차원이라 RNN에 바로 넣는다
- Embedding 파라미터 = `input_dim × output_dim` (bias 없음) → 32줄짜리 단어 벡터 표
- `input_dim` = **단어 수 + 1** (`len(token.word_index) + 1`)
  - 더 작으면 큰 번호의 단어가 사전 범위를 벗어난다 → CPU는 에러, GPU는 에러 없이 0 벡터로 처리된다
- SimpleRNN 없이 `Embedding → Dense(1)`이면 `(None, 5, 1)` / 101 → 단어마다 답이 나온다 → 문장 하나에 답 하나를 내려면 RNN이 필요

**Embedding 쓰는 방법**

```python
model.add(Embedding(input_dim=32, output_dim=100, input_length=5))   # 기본
model.add(Embedding(input_dim=32, output_dim=100))                   # input_length 생략 → (None, None, 100)
model.add(Embedding(32, 100))                                        # 이름 생략 (앞 input_dim, 뒤 output_dim)
model.add(Embedding(32, 100, input_length=5))                        # input_length 는 이름을 써야 한다
# model.add(Embedding(32, 100, 5))   # ValueError: Could not interpret initializer identifier: 5
```

- `input_length`를 생략하면 문장 길이를 "아직 모름"으로 두고 데이터가 들어올 때 맞춘다. 파라미터 수는 같다
- 세 번째 자리는 `input_length`가 아니라 **`embeddings_initializer`** 자리라서 5를 넣으면 에러

**01 ~ 04를 이 순서로 진행한 이유**

| 파일 | 입력 | 해결 | 남은 문제 |
|---|:---:|---|---|
| 01 DNN | `(15, 5)` | - | 번호 크기에 의미가 없다 (바보 26 ≈ 잘생겼다 28), 순서를 못 본다 |
| 02 LSTM | `(15, 5, 1)` | 순서 반영 | 단어가 여전히 숫자 1개 |
| 03 One-Hot | `(15, 5, 31)` | 단어끼리 크기 차이 없음 | 단어 수만큼 칸이 늘어난다 (실제 데이터는 수만 개), 비슷한 단어도 서로 남남 |
| **04 Embedding** | `(15, 5)` → `(15, 5, 100)` | 크기 고정 + **학습되는 벡터** | - |

- Embedding = **One-Hot 다음에 bias 없는 Dense**를 붙인 것과 같은 계산
  - One-Hot 32칸 × Dense 100 = 3,200 = `Embedding(32, 100)`의 Param
  - One-Hot을 실제로 만들지 않고 번호로 해당 줄만 꺼내 써서 가볍고 빠르다
- 벡터 값은 가중치라서 훈련하면서 학습된다 → 비슷하게 쓰이는 단어는 비슷한 벡터가 된다

### 8. OpenAI 임베딩 - 문장을 벡터로 (rag10_Embedding01)

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-large',     # small → 1536 차원 / large → 3072 차원
    api_key=api_key,
    base_url=base_url,
)

prompt = '삼성전자의 창업주는 누구인가요?'
vector = embeddings.embed_query(prompt)   # 문장 1개 → 벡터 1개 (float 리스트)
print(len(vector))                        # 3072
```

- 이전 rag 파일 : `ChatOpenAI` → **답변 문장** / 이번 : `OpenAIEmbeddings` → **숫자 리스트(벡터)**
  - 답변을 만들지 않아서 PromptTemplate, chain을 쓰지 않는다
- keras Embedding은 우리 데이터 15문장으로 처음부터 학습 / OpenAI 임베딩은 이미 학습된 모델을 API로 빌려 쓴다
- RAG의 첫 단계 : 문서와 질문을 벡터로 바꿔 두면 **뜻이 비슷한 문장 = 벡터가 가까운 문장**으로 찾을 수 있다

| model | 차원 |
|---|:---:|
| `text-embedding-3-small` | **1536** |
| `text-embedding-3-large` | **3072** |

### 9. 임베딩 차원 줄이기 (rag10_Embedding02)

```python
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    dimensions=5,                         # 출력 벡터 길이를 5 로
)
vector = embeddings.embed_query(prompt)
print(len(vector))                        # 5
```

- 차원이 작을수록 저장 공간과 검색 속도에 유리 / 너무 작으면 문장 의미를 다 못 담는다
- keras `Embedding(output_dim=100)`으로 벡터 크기를 정한 것과 같은 개념

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras2/keras60_Tokenizer1.py` | 문장 1개 `Tokenizer`, `word_index` / `word_counts`, One-Hot 3가지 (14, 9) |
| `keras2/keras60_Tokenizer2.py` | 문장 2개 공통 단어 사전, `np.concatenate`, One-Hot 3가지 (24, 16) |
| `keras2/keras61_Embedding01_DNN.py` | 15문장 긍정 / 부정, `pad_sequences` (15, 5), 번호 그대로 MinMaxScaler + DNN sigmoid, `np.unique` 라벨 개수, `'개똥이 잘생겼다'` 예측 |
| `keras2/keras61_Embedding02_LSTM.py` | `reshape(15, 5, 1)` + LSTM, 3차원 MinMaxScaler (`reshape(-1, 5)` → 스케일링 → `reshape(-1, 5, 1)`) |
| `keras2/keras61_Embedding03_ohe_to_categorical_LSTM.py` | `to_categorical` (15, 5, 31) + LSTM, 예측 문장과 `np.concatenate` 후 분리 |
| `keras2/keras61_Embedding03_ohe_sklearn_LSTM.py` | `OneHotEncoder` (75, 1) → (15, 5, 31) + LSTM, 예측 문장은 `transform` |
| `keras2/keras61_Embedding03_ohe_pandas_LSTM.py` | `pd.get_dummies` (80,) → `drop([0], axis=1)` → `.values` → (16, 5, 31) + LSTM |
| `keras2/keras61_Embedding04_important.py` | `Embedding(32, 100, input_length=5)` + SimpleRNN, 파라미터 3200 / 4,321, 사용법 3가지, 01 ~ 04 진행 이유 |
| `RAG/rag10_Embedding01.py` | `OpenAIEmbeddings`, `embed_query`, small 1536 / large 3072 차원 |
| `RAG/rag10_Embedding02.py` | `dimensions=5`로 임베딩 차원 줄이기 |
| `keras1/` | 기존 `keras/` 폴더 이름 변경 (Day01 ~ Day21 파일, 내용 동일) (수정) |
| `docs/Day22.md` | Day22 학습 기록 신규 작성 |
| `docs/Day21.md` | 하단 nav에 Day22 링크 추가, 파일 경로 `keras/` → `keras1/` (수정) |
| `docs/Day11.md`, `docs/Day18.md` | 파일 경로 `keras/` → `keras1/` (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(`keras1/`, `keras2/`, rag10 추가, 설명 열 정렬) / 진행도(22일, 28%) 갱신 (수정) |

---

## 📊 실행 결과

### 텍스트 감정 분류 (15문장, train 10 / test 5)

| 파일 | 입력 shape | 단어 표현 | test acc | `'개똥이 잘생겼다'` |
|---|:---:|---|:---:|---|
| 01 DNN | (15, 5) | 번호 + MinMaxScaler | **1.0** | 0.9935 (실행마다 크게 바뀜) |
| 02 LSTM | (15, 5, 1) | 번호 + MinMaxScaler | **1.0** | 0.7037 (실행마다 0.0x ~ 0.9x) |
| 03 to_categorical | (15, 5, 31) | One-Hot | 0.4 ~ 0.8 | 0.0000 ~ 0.14 (부정) |
| 03 OneHotEncoder | (15, 5, 31) | One-Hot | 0.6 ~ 0.8 | 0.0000 ~ 0.004 (부정) |
| 03 get_dummies | (15, 5, 31) | One-Hot | 0.4 / 0.8 | 0.0015 / 0.3072 (부정) |

**읽는 법**
- test가 5문장뿐이라 acc는 0.2 단위로만 바뀌고 데이터 분할에 크게 좌우된다
- 01, 02의 acc 1.0은 번호가 비슷한 문장끼리 우연히 맞은 영향이 크다
- One-Hot은 처음 보는 단어에 정보가 없어서 test acc가 떨어지지만, 예측 문장은 부정 쪽으로 일정하게 나왔다

### Embedding summary (keras61_Embedding04)

| 구성 | Output Shape | Total params |
|---|:---:|:---:|
| `Embedding(32, 100, input_length=5)` → `Dense(1)` | `(None, 5, 1)` | 3,301 |
| `Embedding(32, 100, input_length=5)` → `SimpleRNN(10)` → `Dense(1)` | `(None, 1)` | **4,321** |

### OpenAI 임베딩 차원 (rag10)

| model | dimensions | 벡터 길이 |
|---|:---:|:---:|
| `text-embedding-3-small` | - | 1536 |
| `text-embedding-3-large` | - | 3072 |
| `text-embedding-3-small` | 5 | **5** |

---

## 💻 핵심 개념

### 자연어 전처리 흐름

```
문장 ──fit_on_texts──▶ 단어 사전 ──texts_to_sequences──▶ 번호 리스트 ──pad_sequences──▶ (문장 수, maxlen)
                                                                                         │
             ┌───────────────────────────┬───────────────────────────┬──────────────────┘
             ▼                           ▼                           ▼
   번호 그대로 (01, 02)          One-Hot (03)                Embedding 층 (04)
   크기에 의미 없음              (N, maxlen, 단어 수)         (N, maxlen, output_dim)
                                 단어 수만큼 칸이 늘어남       학습되는 벡터, 크기 고정
```

### One-Hot 3가지 비교

```python
to_categorical(x)[:, 1:]                       # 0번 열까지 만든다 → 직접 지운다
pd.get_dummies(x, dtype=int)                   # 1차원, 나온 값만 열로 (DataFrame)
OneHotEncoder(sparse_output=False).fit_transform(x.reshape(-1, 1))   # 2차원 (n, 1)
```

### 예측 문장은 훈련과 똑같은 순서로

```python
x_predict = token.texts_to_sequences(['개똥이 잘생겼다'])        # fit_on_texts 다시 하지 않음
x_predict = pad_sequences(x_predict, padding='post', maxlen=5)   # 같은 조건
x_predict = scaler.transform(x_predict)                          # fit 하지 않음
```

### Embedding

```python
Embedding(input_dim=단어 수 + 1, output_dim=벡터 크기, input_length=문장 길이)
```

```
입력  (None, 5)         단어 번호 그대로
출력  (None, 5, 100)    단어마다 100칸짜리 벡터
Param = input_dim × output_dim = 32 × 100 = 3200   (bias 없음)
      = One-Hot 32칸 → Dense(100, bias 없음) 과 같은 계산
```

### OpenAI 임베딩

```python
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=api_key, base_url=base_url, dimensions=5)
vector = embeddings.embed_query('문장')        # float 리스트
```

---

## 성과

- `Tokenizer`의 `fit_on_texts` / `word_index` / `word_counts` / `texts_to_sequences`로 문장을 단어 번호로 바꾸고 번호 규칙을 확인
- 문장 1개 (14, 9), 문장 2개 (24, 16)를 `to_categorical` / `get_dummies` / `OneHotEncoder` 3가지로 One-Hot
- `pad_sequences`로 15문장을 (15, 5)로 맞추고 sigmoid 이진 분류 모델 4단계 진행 : 번호 DNN → 번호 LSTM → One-Hot LSTM → Embedding
- `np.unique(..., return_counts=True)`로 stratify 분할 결과 (train 5 : 5, test 3 : 2) 확인
- 3차원 데이터를 `reshape(-1, 5)` → MinMaxScaler → `reshape(-1, 5, 1)`로 스케일링, 열(위치)별 스케일링 특성 확인
- One-Hot을 3가지 방법으로 각각 (15, 5, 31)로 만들어 LSTM에 적용, 예측 문장 칸 수를 `np.concatenate` / `transform`으로 맞춤
- `Embedding(32, 100, input_length=5)` 파라미터 3200 (= 32 × 100), SimpleRNN 1110, Total 4,321을 summary로 확인
- Embedding 사용법 3가지와 `Embedding(32, 100, 5)` 에러 원인(`embeddings_initializer` 자리) 확인
- 01 ~ 04를 비교해 Embedding이 필요한 이유(번호 크기 · 칸 수 · 단어 유사도)를 정리
- `OpenAIEmbeddings`로 문장을 small 1536 / large 3072 차원 벡터로 바꾸고, `dimensions=5`로 차원 조절

---

## 💡 주요 학습 포인트

1. **자연어는 숫자로 바꿔야 모델에 넣을 수 있다**: Tokenizer로 단어마다 번호를 붙이는 게 첫 단계
2. **`fit_on_texts`는 리스트로 넣는다**: 문장 여러 개를 넣는 게 기본 형태
3. **번호는 많이 나온 단어부터 1번**: 횟수가 같으면 먼저 나온 단어가 앞 번호, 0번은 padding용으로 비워 둔다
4. **문장부호는 자동으로 지워진다**: `'먹었다.'` → `'먹었다'`
5. **단어 번호는 이름표다**: 크기에 의미가 없어서 그대로 쓰면 26(바보)과 28(잘생겼다)이 비슷한 값이 된다
6. **`to_categorical`은 0번 열까지 만든다**: 단어 번호가 1부터라 `[:, 1:]`로 지운다
7. **`get_dummies`는 1차원, `OneHotEncoder`는 2차원 `(n, 1)`을 받는다**
8. **문장이 여러 개면 공통 단어 사전을 만든다**: `fit_on_texts([text1, text2])`
9. **문장 길이가 다르면 `pad_sequences`로 맞춘다**: `padding='post'`는 뒤를, 기본값 `'pre'`는 앞을 0으로 채운다
10. **예측 문장은 훈련과 같은 순서로 바꾼다**: `fit_on_texts`는 다시 하지 않고, 같은 `maxlen` / `padding`, scaler는 `transform`만
11. **`np.unique(y, return_counts=True)`로 라벨 개수를 확인한다**: stratify가 잘 됐는지 눈으로 본다
12. **MinMaxScaler는 열마다 따로 스케일링한다**: 같은 단어도 위치에 따라 다른 값이 된다
13. **MinMaxScaler는 2차원만 받는다**: 3차원은 `reshape(-1, 5)` → 스케일링 → `reshape(-1, 5, 1)`
14. **데이터가 아주 적으면 `monitor='loss'`**: validation을 떼면 훈련 데이터가 더 줄어든다
15. **test가 5개면 acc는 운이 크게 작용한다**: 분할에 따라 0 ~ 1.0까지 바뀐다
16. **정답 라벨이 없는 문장의 예측값은 맞고 틀림을 판단할 수 없다**: 모델이 얼마나 긍정이라고 보는지의 값
17. **One-Hot 칸 수는 데이터마다 달라질 수 있다**: 예측 문장만 따로 바꾸면 칸 수가 줄어든다 → 이어 붙여서 바꾸거나 `transform`
18. **One-Hot은 처음 보는 단어에 정보가 없다**: 단어마다 칸이 따로라 번호가 비슷해서 우연히 맞는 일도 없다
19. **One-Hot의 한계**: 단어 수만큼 칸이 늘어나고, 비슷한 단어도 서로 남남이다
20. **Embedding은 단어 번호를 그대로 받는다**: 스케일링, reshape, One-Hot이 필요 없다
21. **Embedding 출력은 3차원**: `(None, 문장 길이, output_dim)` → RNN에 바로 넣는다
22. **Embedding 파라미터 = input_dim × output_dim**: bias가 없다
23. **input_dim = 단어 수 + 1**: padding 0번까지 포함, 작으면 GPU에서는 에러 없이 틀린 결과가 나온다
24. **Embedding = One-Hot + bias 없는 Dense**: 계산은 같지만 One-Hot을 만들지 않아 가볍다
25. **`Embedding(32, 100, 5)`는 에러**: 세 번째 자리는 `embeddings_initializer`, `input_length`는 이름을 써야 한다
26. **OpenAIEmbeddings는 문장을 벡터로 바꾼다**: `ChatOpenAI`처럼 답변을 만들지 않는다
27. **임베딩 차원은 모델마다 다르다**: small 1536 / large 3072, `dimensions`로 줄일 수 있다
28. **RAG는 벡터로 비슷한 문서를 찾는다**: 뜻이 비슷한 문장 = 벡터가 가까운 문장

---

[⬅️ Day21](Day21.md) · [🏠 전체 목차](../README.md)
