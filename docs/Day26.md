# Day26 - 앙상블 모델 (다중 입력 · 다중 출력, Concatenate), PDF RAG 챗봇, HuggingFace 로컬 임베딩 (bge-m3 · Qwen3)

**학습 기간:** 2026-10-08

> 함수형 모델로 **서로 다른 데이터 여러 개를 각자의 모델에 넣고 출력을 `Concatenate`로 합치는 앙상블 모델**을 만들었다. 입력 2개 → 입력 3개 → 입력 3개 + **출력 2개**로 늘려 가며 `fit` · `evaluate` · `predict`에 x와 y를 **리스트**로 넣는 법을 익혔다. LangChain 쪽에서는 **`PyPDFLoader`로 PDF 논문(Attention Is All You Need)** 을 불러와 Day25의 FAISS 챗봇에 붙였고, OpenAI API 대신 **`HuggingFaceEmbeddings`로 bge-m3 · Qwen3-Embedding-0.6B를 내 PC(CPU)에서 돌려** 임베딩했다. 마지막으로 Qwen3 임베딩으로 같은 챗봇을 만들고, 임베딩 차원이 다른 DB(1536 vs 1024)를 섞으면 검색에서 에러가 나는 것을 확인했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 앙상블 모델 | 데이터마다 `Input` → Dense 층 → 출력 텐서, `Concatenate([...])`로 합쳐 Dense → 출력 |
| 다중 입력 | `Model(inputs=[input1, input21], ...)` → `fit([x1, x2], y)` / `predict([x1_pred, x2_pred])` |
| Concatenate | 마지막 축(feature)으로 이어 붙인다 → `(None, 5) + (None, 3)` = `(None, 8)`, param 0 |
| 다중 출력 | 병합 뒤 두 갈래로 나눠 `Model(..., outputs=[last_output1, last_output2])` → `fit(x, [y1, y2])` |
| 다중 출력 loss | `evaluate` 결과 = `[전체 loss, last1 loss, last2 loss]` (전체 = 두 loss의 합) |
| train_test_split | x1, x2, x3, y1, y2를 한 번에 넣어야 같은 행끼리 같이 섞인다 |
| PyPDFLoader | PDF → 페이지마다 Document 1개 (15페이지 → 15개), metadata에 `page` · `source` |
| PDF 챗봇 | 청크 300 / overlap 100 → 205개 → OpenAI 임베딩 → `./_db/Faiss19` |
| HuggingFaceEmbeddings | `HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs={'device': 'cpu'})` |
| 로컬 임베딩 차원 | bge-m3 · Qwen3-Embedding-0.6B 모두 **1024** (OpenAI small 1536) |
| Qwen3 챗봇 | 같은 PDF를 Qwen3로 임베딩 → `./_db/Faiss20`, `score_threshold` 0.2 |
| 차원 불일치 | 1536 DB를 1024 임베딩으로 검색 → `AssertionError` |

---

## 📖 핵심 학습 내용

### 1. 입력 2개 앙상블 모델 (keras69_ensemble1)

```python
x1_datasets = np.array([range(100), range(301, 401)]).T     # (100, 2)  삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511),
                        range(150, 250)]).transpose()        # (100, 3)  원유가, 환율, 금시세
y = np.array(range(3001, 3101))                              # (100,)    화성 화씨 온도

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.7, random_state=77)
# (70, 2) (70, 3) (70,) / (30, 2) (30, 3) (30,)
```

- x1, x2, y를 **한 번에** 넣어야 같은 행끼리 같은 순서로 섞인다 (따로 나누면 x와 y의 짝이 어긋난다)
- 반환 순서 : 넣은 배열마다 `(train, test)` 한 쌍씩

```python
from tensorflow.keras.layers import concatenate, Concatenate

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
...
output1 = Dense(5, activation='relu', name='han4')(dense3)        # (None, 5)

#2-2. 모델2
input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
...
output21 = Dense(3, name='han25')(dense24)                         # (None, 3)

#2-3. 모델 병합
# merge1 = concatenate([output1, output21], name='mg1')            # 함수
merge1 = Concatenate(name='mg1')([output1, output21])              # 층(클래스) → (None, 8)
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

#2-4. 모델 구성
model = Model(inputs=[input1, input21], outputs=last_output)
```

- **Sequential은 입력이 하나뿐**이라 앙상블은 함수형(`Input` / `Model`)으로 만든다
- 모델1 · 2를 따로 `Model`로 만들 필요 없이 **출력 텐서(output1, output21)를 바로 합친다**
- `concatenate`(함수)와 `Concatenate`(층) 둘 중 아무거나 → 마지막 축으로 이어 붙인다 (5 + 3 = 8)
- `from tensorflow.keras.layers.merge import concatenate` → `ModuleNotFoundError` (지금 버전은 `tensorflow.keras.layers`에서 import)
- `layer name`을 주면 `summary`에서 어느 모델의 층인지 알아보기 쉽다 (`han1` ~ `han4` / `han21` ~ `han25` / `mg1` ~ `mg3`)
- summary : 두 입력의 층이 섞여 나오고 `Connected to`로 연결을 확인 → `mg1 (Concatenate)` ← `han4[0][0], han25[0][0]`, Total params **5,339**

```python
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8)   # x 를 [x1, x2] 리스트로
result = model.evaluate([x1_test, x2_test], y_test)

x1_pred = np.array([range(100, 106), range(400, 406)]).T              # (6, 2)
x2_pred = np.array([range(200, 206), range(510, 516), range(249, 255)]).transpose()   # (6, 3)
y_pred = model.predict([x1_pred, x2_pred])                            # (6, 1)
print('예측값: ', y_pred.reshape(-1))                                 # (6,) 로 펴서 한 줄로
```

- 예측 데이터도 학습 데이터처럼 **열 개수**를 맞춘다 (6행 × 2열 / 6행 × 3열)
- 데이터 규칙상 x1 첫 열이 100 ~ 105이면 정답 y는 3101 ~ 3106

### 2. 입력 3개 앙상블 모델 (keras69_ensemble2)

```python
x3_datasets = np.array([range(100), range(301, 401),
                        range(77, 177), range(33, 133)]).T          # (100, 4)

#2-3. 모델3
input100 = Input(shape=(4,))
dense101 = Dense(50, name='han101')(input100)
...
output100 = Dense(5, name='han105')(dense104)                       # (None, 5)

merge1 = Concatenate(name='mg1')([output1, output21, output100])    # (None, 5 + 3 + 5) = (None, 13)
model = Model(inputs=[input1, input21, input100], outputs=last_output)

model.fit([x1_train, x2_train, x3_train], y_train, epochs=100, batch_size=8)
y_pred = model.predict([x1_pred, x2_pred, x3_pred])
```

- 입력이 하나 늘면 `train_test_split` · `Model(inputs)` · `fit` · `evaluate` · `predict` 모두에 x3를 함께 넣는다
- 병합층이 `(None, 13)`이 되어 mg2 param = 13 × 10 + 10 = **140**, Total params **9,634** (ensemble1 5,339)

### 3. 출력 2개 앙상블 모델 (keras69_ensemble3)

```python
y1 = np.array(range(3001, 3101))      # 화성 화씨 온도
y2 = np.array(range(13001, 13101))    # 비트코인 가격

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2, train_size=0.7, random_state=77)

#2-5. 분기1
last_dense1 = Dense(10, name='ld1')(merge3)
last_dense2 = Dense(10, name='ld2')(last_dense1)
last_output1 = Dense(1, name='last1')(last_dense2)

#2-6. 분기2 (레이어 구성 안 하는 경우)
last_output2 = Dense(1, name='last2')(merge3)

#2-7. 모델 구성
model = Model(inputs=[input1, input21, input100], outputs=[last_output1, last_output2])

model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=100, batch_size=8)
result = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
y1_pred, y2_pred = model.predict([x1_pred, x2_pred, x3_pred])
```

- 병합한 `merge3` 뒤를 **두 갈래로 나눈다** : 분기1은 Dense 2층을 더 쌓고, 분기2는 바로 출력
- `outputs=[...]` 리스트 → `fit` · `evaluate`의 y도 `[y1, y2]` 리스트로 넣는다
- `evaluate` 결과 = **`[전체 loss, last1 loss, last2 loss]`** → 전체 loss는 출력마다 계산한 loss의 합 (0.2256 + 0.0138 = 0.2394)
- `predict`도 입력 3개를 리스트로 한 번에 넣고, 출력이 2개라 `[y1 예측, y2 예측]` 리스트로 나온다 → 두 변수로 나눠 받는다
- `evaluate`의 y는 test 데이터(30행)여야 한다 → x는 30행, y가 전체(100행)면 개수가 맞지 않아 에러
- Total params **9,815** (ensemble2에서 last 6 대신 ld1 60 + ld2 110 + last1 11 + last2 6)

### 4. PDF 불러오기 (rag19_PyPDF_1)

```python
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader('./_data/Attention Is All You Need.pdf')
pdf_docs = pdf_loader.load()

print(type(pdf_docs))   # <class 'list'>
print(len(pdf_docs))    # 15
print(pdf_docs[0])      # page_content='... Attention Is All You Need ... Abstract ...'
                        # metadata={'source': './_data/Attention Is All You Need.pdf', 'total_pages': 15, 'page': 0, 'page_label': '1', ...}
```

- `TextLoader`는 파일 1개를 Document 1개로 불러왔다면, **`PyPDFLoader`는 페이지마다 Document 1개** → 15페이지 = 15개
- metadata에 `source`(파일 경로) · `page`(0부터) · `page_label` · `total_pages`와 PDF 정보(producer, creator 등)가 들어 있다

### 5. PDF 논문 FAISS 챗봇 (rag19_PyPDF_2_ChatBot)

Day25 rag18의 FAISS 챗봇에 rag11의 불러오기 · 청킹을 붙여 **문서를 PDF 논문으로** 바꿨다.

```python
pdf_loader = PyPDFLoader(path + 'Attention Is All You Need.pdf')
pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, separators=['\n\n', '\n', ' ', ''])
split_doc = pdf_loader.load_and_split(pdf_splitter)
print(len(split_doc))             # 205 (15페이지 PDF -> 청크 205개)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=api_key, base_url=base_url)   # 1536 차원
db = FAISS.from_documents(documents=split_doc, embedding=embeddings)
db.save_local(folder_path='./_db/Faiss19', index_name='transformer_index19')
db = FAISS.load_local(folder_path='./_db/Faiss19', index_name='transformer_index19',
                      embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type='similarity_score_threshold',
                            search_kwargs={'k': 4, 'score_threshold': 0.05})
```

- `load_and_split(splitter)`로 PDF 불러오기 + 청킹을 한 번에
- 챗봇 구조(점수 거르기 · system 규칙 · 이전 대화 · 출처 · examples)는 rag18 그대로, 문서 이름 · 지시어 예시("그 모델", "그 층") · 화면 문구만 논문에 맞게
- examples : `Transformer 모델 구조를 설명해줘` / `Multi-Head Attention 은 무엇이야?` / `Positional Encoding 은 왜 필요해?` / `삼성전자의 사업 전망은?`(문서 밖)
- 결과 : Multi-Head Attention 질문 → 논문 내용으로 답 + `📄 출처: Attention Is All You Need.pdf`, 삼성전자 질문 → 가장 높은 점수도 -0.11이라 걸러져 "주어진 정보로는 답변할 수 없습니다."

### 6. HuggingFace 로컬 임베딩 (rag20_embedding2_bge-m3 · rag20_embedding3_qwen3)

rag10_Embedding01의 OpenAI 임베딩을 막고, **HuggingFace에 공개된 임베딩 모델을 내 PC에서 직접** 돌렸다.

```python
# pip install langchain_huggingface
# ImportError: Could not import sentence_transformers python package. ...
# 해결 방법 : pip install sentence-transformers
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-m3',               # rag20_embedding3 : 'Qwen/Qwen3-Embedding-0.6B'
    model_kwargs={
        'device' : 'cpu',
        # 'local_files_only' : True,          # 내려받은 뒤에는 인터넷 연결 없이 저장된 모델만
    }
)
vector = embeddings.embed_query('삼성전자의 창업주는 누구인가요?')
print(len(vector))                            # 1024
```

- `langchain_huggingface`는 내부에서 `sentence-transformers`를 쓴다 → 없으면 `ImportError`, 함께 설치
- 처음 실행할 때 모델을 내려받아 `~/.cache/huggingface`에 저장 (bge-m3 약 4.3GB, Qwen3-Embedding-0.6B 약 1.2GB), 이후에는 저장된 모델을 쓴다
- API 키 · 요금 없이 임베딩할 수 있다 (대신 내 PC에서 계산)
- 사용법은 `OpenAIEmbeddings`와 같다 → `embed_query` / `embed_documents`, LangChain 벡터 DB에 그대로 넣을 수 있다

| 모델 | 만든 곳 | 차원 | 벡터 길이(norm) |
|---|---|:---:|:---:|
| text-embedding-3-small | OpenAI (API) | 1536 | - |
| text-embedding-3-large | OpenAI (API) | 3072 | - |
| BAAI/bge-m3 | BAAI (로컬) | **1024** | 1.0 |
| Qwen/Qwen3-Embedding-0.6B | Alibaba Qwen (로컬) | **1024** | 1.0 |

### 7. Qwen3 임베딩 챗봇 (rag20_embedding4_ChatBot)

rag19 챗봇에서 **임베딩만 Qwen3로** 바꿨다.

```python
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='Qwen/Qwen3-Embedding-0.6B', model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(documents=split_doc, embedding=embeddings)
db.save_local(folder_path='./_db/Faiss20', index_name='transformer_index20')    # Faiss19 와 다른 경로

retriever = db.as_retriever(search_type='similarity_score_threshold',
                            search_kwargs={'k': 4, 'score_threshold': 0.2})     # 0.05 -> 0.2
```

- 임베딩 차원이 1536 → 1024로 바뀌므로 **DB 경로를 분리**해 rag19의 OpenAI DB(Faiss19)와 따로 저장한다
- `FAISS.from_documents`가 인덱스를 직접 만들기 때문에 `faiss.IndexFlatL2`를 따로 만들 필요가 없다
- Qwen3는 점수 분포가 달라 `score_threshold`를 다시 정했다
  - 논문 질문 0.42 ~ 0.59 / "삼성전자의 사업 전망은?" -0.03 / "오늘 점심 메뉴 추천해줘" 0.09
  - 0.05로 두면 점심 메뉴 같은 무관한 질문도 통과 → **0.2**

**임베딩 차원과 벡터 DB**

- Qwen3 임베딩(1024)으로 OpenAI DB(Faiss19, 1536)를 불러와 검색하면 `AssertionError`
  - `load_local`은 파일만 읽어서 에러가 나지 않는다 → **검색할 때** 질문 벡터와 DB 벡터의 길이가 달라 에러
  - FAISS 내부 `assert d == self.d`라 에러 메시지가 비어 있다 → `index.d`로 차원을 확인한다
- **임베딩 모델을 바꾸면 벡터 DB도 그 모델로 다시 만들어야 한다** (저장할 때와 검색할 때 같은 임베딩)
- FAISS 파일 크기로도 차원을 알 수 있다 : 205개 × 1536 × 4바이트 ≈ 1.26MB (Faiss19) / 205개 × 1024 × 4바이트 ≈ 0.84MB (Faiss20)

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras2/keras69_ensemble1.py` | 입력 2개(x1 (100, 2) · x2 (100, 3)) 앙상블, `Concatenate`, `Model(inputs=[...])`, params 5,339 |
| `keras2/keras69_ensemble2.py` | 입력 3개(x3 (100, 4) 추가), mg1 `(None, 13)`, params 9,634 |
| `keras2/keras69_ensemble3.py` | 입력 3개 + 출력 2개(y1 · y2), 분기1 · 분기2, `outputs=[...]`, `fit(x, [y1, y2])`, params 9,815 |
| `RAG/rag19_PyPDF_1.py` | `PyPDFLoader.load()` → 페이지 15개, Document의 page_content · metadata 확인 |
| `RAG/rag19_PyPDF_2_ChatBot.py` | PDF 논문 청크 205개 → OpenAI 임베딩 → `./_db/Faiss19` → rag18 구조 Gradio 챗봇 |
| `RAG/rag20_embedding2_bge-m3.py` | `HuggingFaceEmbeddings('BAAI/bge-m3')` CPU 로컬 임베딩, 1024 차원 |
| `RAG/rag20_embedding3_qwen3.py` | `HuggingFaceEmbeddings('Qwen/Qwen3-Embedding-0.6B')` CPU 로컬 임베딩, 1024 차원 |
| `RAG/rag20_embedding4_ChatBot.py` | Qwen3 임베딩 PDF 챗봇, `./_db/Faiss20` (Faiss19 와 경로 분리), threshold 0.2 |
| `_data/Attention Is All You Need.pdf` | RAG 실습용 PDF 논문 (15페이지) |
| `docs/Day26.md` | Day26 학습 기록 신규 작성 |
| `docs/Day25.md` | 하단 nav에 Day26 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day26, `_data` PDF, `_db/Faiss19` · `Faiss20`) / 진행도(26일, 33%) 갱신 (수정) |
| `.gitignore` | `_db/` 주석에 rag18 ~ rag20 FAISS 추가 (수정) |

---

## 📊 실행 결과

### 앙상블 모델

| 파일 | 구조 | Total params | test loss (mse) | 예측 (정답 3101 ~ 3106) |
|---|---|:---:|:---:|---|
| ensemble1 | 입력 2 → 출력 1 | 5,339 | 0.0504 | 3100.60 ~ 3106.89 |
| ensemble2 | 입력 3 → 출력 1 | 9,634 | 0.0093 | 3092.58 ~ 3100.42 |
| ensemble3 | 입력 3 → 출력 2 | 9,815 | 전체 0.2394 (y1 0.2256 + y2 0.0138) | y1 3093.98 ~ 3098.88 / y2 13070.58 ~ 13075.60 (정답 13101 ~ 13106) |

**읽는 법**
- test loss는 작아도 범위 밖(100 ~ 105) 값을 넣은 예측은 정답과 차이가 있었다 → 훈련 범위(0 ~ 99) 밖을 예측하는 것은 어렵다 (Day19 범위 밖 예측과 같은 문제)
- ensemble1은 예측이 정답에 가장 가까웠다

### RAG

| 파일 | 확인한 값 |
|---|---|
| rag19_PyPDF_1 | `type` list, `len` 15 (페이지 수), `pdf_docs[0].metadata['page']` 0 |
| rag19_PyPDF_2_ChatBot | 청크 205개, Faiss19 차원 1536 · ntotal 205, Multi-Head Attention 질문 → 답 + 출처 / 삼성전자 질문 → 답변할 수 없음 |
| rag20_embedding2_bge-m3 | 벡터 차원 1024, norm 1.0 |
| rag20_embedding3_qwen3 | 벡터 차원 1024, norm 1.0 |
| rag20_embedding4_ChatBot | Faiss20 차원 1024 · ntotal 205, Multi-Head Attention 질문 → 답 + 출처 / 삼성전자 질문 → 답변할 수 없음 |

---

## 💻 핵심 개념

### 앙상블 모델 구조

```
x1 (N, 2) ─▶ Input(2,) ─▶ Dense ×4 ─▶ output1   (None, 5) ─┐
x2 (N, 3) ─▶ Input(3,) ─▶ Dense ×5 ─▶ output21  (None, 3) ─┼─▶ Concatenate (None, 13) ─▶ Dense ─▶ Dense ─▶ merge3 (None, 5)
x3 (N, 4) ─▶ Input(4,) ─▶ Dense ×5 ─▶ output100 (None, 5) ─┘                                            │
                                                                                                       ├─▶ ld1 ─▶ ld2 ─▶ last1 (y1)
                                                                                                       └─▶ last2 (y2)
```

### 다중 입력 · 다중 출력에서 리스트로 넣는 곳

```
Model     : Model(inputs=[in1, in2, in3], outputs=[out1, out2])
split     : train_test_split(x1, x2, x3, y1, y2, ...)  → (train, test) 쌍이 넣은 순서대로
fit       : model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train])
evaluate  : model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])  → [전체 loss, loss1, loss2]
predict   : y1_pred, y2_pred = model.predict([x1_pred, x2_pred, x3_pred])
```

### Concatenate

```
Concatenate()([A, B, C])     마지막 축으로 이어 붙인다     param 0
(None, 5) + (None, 3) + (None, 5)  →  (None, 13)
concatenate([A, B, C])       함수 형태 (같은 결과)
```

### 임베딩 모델 비교

```
                 OpenAIEmbeddings                      HuggingFaceEmbeddings
모델           text-embedding-3-small / large         BAAI/bge-m3, Qwen/Qwen3-Embedding-0.6B
실행 위치      OpenAI 서버 (API 키 · 요금)            내 PC (CPU, 처음에 모델 다운로드)
차원           1536 / 3072                            1024 / 1024
사용법         embed_query / embed_documents          embed_query / embed_documents (같음)
```

### 임베딩과 벡터 DB 짝 맞추기

```
rag19 : OpenAI small (1536)  ─▶  ./_db/Faiss19  ─▶  OpenAI small 로 검색      OK
rag20 : Qwen3 (1024)         ─▶  ./_db/Faiss20  ─▶  Qwen3 로 검색             OK
        Qwen3 (1024)         ─▶  ./_db/Faiss19 (1536) 검색                    AssertionError
```

---

## 성과

- 함수형 모델로 입력 2개 → 3개 → 입력 3개 + 출력 2개 앙상블 모델을 차례로 만들었다
- `Concatenate`로 각 모델의 출력 텐서를 합치고 summary의 `Connected to`로 연결 확인
- 다중 입력 · 다중 출력에서 `train_test_split` · `fit` · `evaluate` · `predict`에 리스트로 넣는 법과 `evaluate` 결과 `[전체, loss1, loss2]` 확인
- `PyPDFLoader`로 PDF 논문을 페이지별 Document로 불러오고, 청크 205개로 잘라 FAISS 챗봇에 적용
- `HuggingFaceEmbeddings`로 bge-m3 · Qwen3-Embedding-0.6B를 내 PC에서 돌려 API 없이 1024 차원 임베딩
- Qwen3 임베딩으로 같은 PDF 챗봇을 만들고 `score_threshold`를 0.2로 다시 정했다
- 임베딩 차원이 다른 DB를 검색하면 `AssertionError`가 나는 것을 확인하고, 임베딩마다 DB 경로를 분리

---

## 💡 주요 학습 포인트

1. **앙상블은 함수형으로 만든다**: Sequential은 입력이 하나뿐
2. **모델마다 `Input`을 따로 만든다**: `Input(shape=(2,))`, `Input(shape=(3,))`, `Input(shape=(4,))`
3. **출력 텐서를 바로 합친다**: 모델1 · 2를 각각 `Model`로 만들 필요 없다
4. **`Concatenate`는 마지막 축으로 이어 붙인다**: 5 + 3 + 5 = 13, 파라미터 0
5. **`concatenate`(함수) / `Concatenate`(층)는 같은 결과**: import는 `tensorflow.keras.layers`에서
6. **layer `name`을 주면 summary를 읽기 쉽다**: 어느 모델의 층인지 구분
7. **`train_test_split`에 여러 배열을 한 번에 넣는다**: 같은 행끼리 같이 섞이고 (train, test) 쌍으로 나온다
8. **다중 입력은 x를 리스트로**: `fit([x1, x2, x3], y)`, `predict([x1_pred, x2_pred, x3_pred])`
9. **예측 데이터의 열 개수는 학습 데이터와 같아야 한다**: (6, 2) / (6, 3) / (6, 4)
10. **다중 출력은 `outputs=[...]`, y도 리스트로**: `fit(x, [y1, y2])`
11. **분기마다 층 수를 다르게 할 수 있다**: 분기1은 Dense 2층 추가, 분기2는 바로 출력
12. **다중 출력의 `evaluate`는 `[전체 loss, 출력별 loss ...]`**: 전체 = 출력별 loss의 합
13. **`predict` 결과도 출력 개수만큼 리스트**: `y1_pred, y2_pred = model.predict(...)`
14. **훈련 범위 밖 값의 예측은 어긋나기 쉽다**: test loss가 작아도 100 ~ 105 예측은 정답과 차이
15. **`PyPDFLoader`는 페이지마다 Document 1개**: metadata에 `page` · `source`
16. **`load_and_split(splitter)`로 PDF도 바로 청킹**: 15페이지 → 청크 205개
17. **`HuggingFaceEmbeddings`로 API 없이 임베딩**: 처음 한 번 모델을 내려받아 캐시에 저장
18. **`langchain_huggingface`는 `sentence-transformers`가 필요하다**: 없으면 ImportError
19. **임베딩 모델마다 차원이 다르다**: OpenAI small 1536 / bge-m3 · Qwen3 1024
20. **임베딩 모델을 바꾸면 DB도 다시 만든다**: 저장할 때와 검색할 때 같은 임베딩을 써야 한다
21. **임베딩마다 DB 경로를 나눈다**: 같은 경로에 저장하면 다른 차원의 DB를 덮어쓴다
22. **차원이 다르면 `load_local`이 아니라 검색에서 에러**: FAISS `AssertionError` (`index.d`로 차원 확인)
23. **임베딩을 바꾸면 `score_threshold`도 다시 정한다**: 모델마다 점수 분포가 다르다 (OpenAI 0.05 → Qwen3 0.2)

---

[⬅️ Day25](Day25.md) · [🏠 전체 목차](../README.md)
