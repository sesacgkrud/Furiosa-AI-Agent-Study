# Day30 - BM25 키워드 검색, Chroma 벡터 검색, EnsembleRetriever hybrid 검색

**학습 기간:** 2026-10-15

> 같은 문장 8개와 같은 질문(`'삼성전자의 반도체 사업'`)으로 **검색 방식 3가지**를 비교했다. 먼저 `BM25Retriever`로 **키워드가 글자 그대로 같은 문서**를 찾는 BM25 검색을 해 보고, 같은 문서를 `Chroma.from_documents`로 Chroma12 DB에 저장해 **의미가 가까운 문서**를 찾는 벡터 검색을 했다. 마지막으로 두 검색기를 `EnsembleRetriever(retrievers=[...], weights=[0.5, 0.5])`로 합쳐 **hybrid 검색**을 했다. 한국어는 `삼성전자는` / `삼성전자의`처럼 조사가 붙어 BM25가 단어를 잘 맞히지 못했고, 벡터 검색은 조사가 달라도 의미로 찾았으며, hybrid는 두 결과를 순위대로 합쳐 두 검색기에 모두 나온 문서를 1등으로 올렸다. Day29의 `keras76_all_T_02_cifar100.py`는 `verbose=1`로 바꿔 epoch 진행 상황을 보면서 실행했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 검색 데이터 | `Document(page_content=...)` 문장 8개를 직접 만들고, 질문은 `'삼성전자의 반도체 사업'` 하나로 비교 |
| BM25 검색 | `BM25Retriever.from_documents(texts)` + `.k = 3` → 질문 단어가 글자 그대로 있는 문서 (임베딩 X) |
| BM25 한계 | 띄어쓰기로만 단어를 나눠 `반도체를` · `반도체는`은 `반도체`와 다른 단어 → 8개 중 1개만 점수, 나머지 0점 |
| 벡터 검색 | `Chroma.from_documents(...)` + `as_retriever(search_kwargs={'k':3})` → 의미가 가까운 문서 |
| Chroma12 | rag12 때 만든 DB에 texts 8개가 추가되어, 이전 samsung 텍스트 청크도 검색 결과에 나온다 |
| DB 불러오기 | hybrid에서는 `Chroma(embedding_function=..., persist_directory=..., collection_name=...)`로 저장된 DB만 연결 |
| hybrid 검색 | `EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])` |
| hybrid 결과 | 두 검색기에 모두 나온 `삼성전자는 AI 반도체 기술을 연구한다.`가 1등, 3개 + 3개 중 겹친 1개를 빼고 5개 |
| keras76 cifar100 | `verbose=0` → `verbose=1` (수정) |

---

## 📖 핵심 학습 내용

### 1. BM25 키워드 검색 (rag23_01_bm25_search)

```python
from langchain_core.documents import Document
# pip install rank_bm25

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = '삼성전자의 반도체 사업'

from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(texts)   # 문서 리스트로 BM25 검색기를 만든다
bm25_retriever.k = 3                                    # 검색 결과로 가져올 문서 수

bm25_docs = bm25_retriever.invoke(query)
for doc in bm25_docs:
    print(doc.page_content)
```

- **BM25 (Best Match 25)** : 키워드 기반 검색. 질문의 단어가 문서에 있는지로 점수를 매긴다
  - 질문 단어가 문서에 많이 나올수록 점수가 높다
  - 여러 문서에 흔한 단어보다 드물게 나오는 단어가 맞을 때 점수가 더 높다
  - 긴 문서는 점수를 조금 깎는다
- 파일 · 청킹 없이 `Document(page_content=...)`로 문장을 직접 만들어 넣었다
- 임베딩 모델 · 벡터 DB가 필요 없다 → `pip install rank_bm25`만 설치
- `bm25_retriever.k = 3` : 만든 뒤에 속성으로 가져올 문서 수를 정한다 (벡터 검색기는 `search_kwargs={'k':3}`)
- `invoke(query)` : 다른 retriever와 똑같이 Document 리스트로 돌려준다

**결과 해석**

```
========== BM25 검색 결과 ==========
삼성전자는 AI 반도체 기술을 연구한다.
반도체는 컴퓨터의 핵심 부품이다.
테슬라는 전기자동차를 제조한다.
```

- 기본 설정은 **띄어쓰기로만 단어를 나눈다** → 질문 단어 = `['삼성전자의', '반도체', '사업']`
- `삼성전자는` · `반도체를` · `반도체는`처럼 조사가 붙으면 다른 단어가 된다
- `반도체`가 조사 없이 그대로 있는 `AI 반도체 기술을` 문장만 점수가 있고(1.47), **나머지 7개는 0점**
- 2, 3번째는 0점끼리 채워진 것이라 질문과 관계가 없다 (`테슬라는 전기자동차를 제조한다.`)
- 정작 질문에 가장 맞는 `삼성전자는 반도체를 생산하는 기업이다.`는 찾지 못했다

### 2. Chroma 벡터 검색 (rag23_02_vector_search)

rag23_01을 베이스로 BM25 부분은 주석 처리하고, 같은 문서를 Chroma에 저장해 벡터 검색을 했다.

```python
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)

DB_PATH = './_db/Chroma12/'

vector_store = Chroma.from_documents(
    embedding=embeddings,
    documents=texts,
    persist_directory=DB_PATH,
    collection_name='croma12'
)

vector_retriever = vector_store.as_retriever(
    search_kwargs={'k':3}
)

vector_docs = vector_retriever.invoke(query)
```

- 문장을 임베딩 벡터로 바꿔 **의미가 가까운 문서**를 찾는다 → `삼성전자의` / `삼성전자는`처럼 조사가 달라도 찾는다
- `Chroma.from_documents` : texts 8개를 임베딩해서 `./_db/Chroma12/`의 `croma12` 컬렉션에 저장
- 같은 폴더 · 같은 컬렉션 이름이면 **기존 문서는 그대로 두고 새 문서가 추가된다** (실행할 때마다 8개씩 다시 추가)
  - croma12 컬렉션 : 이전 rag12 실습 때 저장한 rag_data 청크 208개 + texts 8개 = 216개
  - 그래서 결과 3번째에 texts가 아닌 `삼성전자 사업 전망` 청크가 나왔다

```
========== Vector DB 검색 결과 ==========
삼성전자는 반도체를 생산하는 기업이다.
삼성전자는 AI 반도체 기술을 연구한다.
삼성전자 사업 전망

삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다.
...
```

- BM25에서 0점이던 `삼성전자는 반도체를 생산하는 기업이다.`가 1등
- BM25 결과의 `테슬라는 전기자동차를 제조한다.`처럼 관계없는 문장은 나오지 않았다

### 3. EnsembleRetriever hybrid 검색 (rag23_03_hybrid_search)

rag23_02를 베이스로 BM25 검색기는 다시 살리고, 벡터 DB는 rag23_02에서 저장한 것을 불러와 두 검색기를 합쳤다.

```python
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3

vector_store = Chroma(                  # 저장된 DB 불러오기 (문서를 다시 넣지 않는다)
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='croma12'
)
vector_retriever = vector_store.as_retriever(search_kwargs={'k':3})

from langchain_classic.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever,
    ],
    weights=[0.5,0.5] # 검색 중요도
)

hybrid_docs = hybrid_retriever.invoke(query)
```

- **hybrid 검색** = 키워드 검색(BM25) + 벡터 검색 → 글자가 같은 문서와 의미가 가까운 문서를 함께 찾는다
- `retrievers` : 합칠 검색기 리스트, 각자 `k=3`개씩 찾는다
- `weights` : 검색기마다 결과를 얼마나 중요하게 볼지 (`[0.5, 0.5]` → 같은 비중)
- 두 검색기의 **순위**에 weights를 곱해 점수를 합치고 높은 순으로 정렬, **겹치는 문서는 한 번만** 남긴다
- `Chroma.from_documents` 대신 `Chroma(embedding_function=...)` : 저장된 DB를 연결만 하므로 문서가 또 추가되지 않는다 (불러올 때도 저장 때와 같은 임베딩 모델)

```
========== hybrid DB 검색 결과 ==========
삼성전자는 AI 반도체 기술을 연구한다.      <- BM25 1등 + 벡터 2등 (두 검색기 모두)
삼성전자는 반도체를 생산하는 기업이다.      <- 벡터 1등
반도체는 컴퓨터의 핵심 부품이다.           <- BM25 2등
테슬라는 전기자동차를 제조한다.            <- BM25 3등
삼성전자 사업 전망 ...                    <- 벡터 3등
```

- 3개 + 3개 = 6개 중 `삼성전자는 AI 반도체 기술을 연구한다.`가 겹쳐서 결과는 **5개**
- 두 검색기에 모두 나온 문서가 1등으로 올라간다
- BM25의 0점 문장(테슬라)도 함께 들어온다 → 키워드 검색이 약한 한국어 문장에서는 BM25 결과가 섞여 들어오는 것을 확인

### 4. keras76_all_T_02_cifar100 verbose 변경 (수정)

```python
model.fit(x_train, y_train, epochs=9999999, batch_size=32,
          verbose=1,                  # cifar100 훈련 진행 상황을 epoch 마다 확인
          validation_split=0.2, callbacks=[es])
```

- Day29에서 모델이 여러 개라 `verbose=0`으로 껐던 epoch 출력을 `verbose=1`로 켜서 진행 상황을 보며 실행했다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `RAG/rag23_01_bm25_search.py` | 문장 8개 Document + `BM25Retriever` 키워드 검색 (k=3), 조사 때문에 1개만 점수 |
| `RAG/rag23_02_vector_search.py` | 같은 문서를 `Chroma.from_documents`로 Chroma12에 저장 + 벡터 검색 (k=3) |
| `RAG/rag23_03_hybrid_search.py` | BM25 검색기 + 저장된 Chroma12 벡터 검색기를 `EnsembleRetriever`(weights 0.5 / 0.5)로 합친 hybrid 검색 |
| `keras2/keras76_all_T_02_cifar100.py` | `verbose=0` → `verbose=1` (수정) |
| `docs/Day30.md` | Day30 학습 기록 신규 작성 |
| `docs/Day29.md` | 하단 nav에 Day30 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day30) / 진행도(30일, 38%) 갱신 (수정) |

---

## 📊 실행 결과

질문 : `'삼성전자의 반도체 사업'`, 각 검색기 k = 3

| 순위 | BM25 (rag23_01) | 벡터 (rag23_02) | hybrid (rag23_03) |
|:---:|---|---|---|
| 1 | 삼성전자는 AI 반도체 기술을 연구한다. | 삼성전자는 반도체를 생산하는 기업이다. | 삼성전자는 AI 반도체 기술을 연구한다. |
| 2 | 반도체는 컴퓨터의 핵심 부품이다. (0점) | 삼성전자는 AI 반도체 기술을 연구한다. | 삼성전자는 반도체를 생산하는 기업이다. |
| 3 | 테슬라는 전기자동차를 제조한다. (0점) | 삼성전자 사업 전망 ... (rag12 청크) | 반도체는 컴퓨터의 핵심 부품이다. |
| 4 | | | 테슬라는 전기자동차를 제조한다. |
| 5 | | | 삼성전자 사업 전망 ... (rag12 청크) |

- BM25 : 글자가 같은 단어만 맞아서 8개 중 1개만 점수, 관계없는 문장이 2 · 3등
- 벡터 : 의미로 찾아서 삼성전자 · 반도체 문장이 모두 위로
- hybrid : 두 검색기에 모두 나온 문장이 1등, 겹친 1개를 빼고 5개

---

## 💻 핵심 개념

### 검색 방식 비교

```
BM25 (키워드)   질문 ─▶ 띄어쓰기로 단어 나누기 ─▶ 문서에 같은 단어가 있는지 점수 ─▶ 상위 k개
                       임베딩 X, 벡터 DB X                 조사가 붙으면 다른 단어

벡터 (의미)     질문 ─▶ 임베딩 벡터 ─▶ Chroma DB 의 문서 벡터와 거리 비교 ─▶ 가까운 k개
                       OpenAIEmbeddings                    조사가 달라도 의미로 찾는다

hybrid          ┌─ bm25_retriever   (k=3) ─┐
                │                          ├─▶ EnsembleRetriever (weights 0.5 / 0.5) ─▶ 순위 합산 · 중복 제거
                └─ vector_retriever (k=3) ─┘
```

### Chroma 저장 vs 불러오기

```
Chroma.from_documents(embedding=..., documents=texts, persist_directory=..., collection_name=...)
    → 문서를 임베딩해서 저장 (같은 컬렉션이면 기존 문서 + 새 문서, 실행할 때마다 추가)   (rag23_02)

Chroma(embedding_function=..., persist_directory=..., collection_name=...)
    → 저장된 DB 연결만, 문서 추가 X                                                  (rag23_03)
```

### retriever 만들기

```
BM25Retriever.from_documents(texts)    → .k = 3                       (속성으로 개수 지정)
vector_store.as_retriever(...)         → search_kwargs={'k':3}        (인자로 개수 지정)
EnsembleRetriever(retrievers=[...], weights=[...])
모두 .invoke(query) → Document 리스트
```

---

## 성과

- `BM25Retriever`로 임베딩 없이 키워드만으로 문서를 찾는 BM25 검색을 해 봤다
- 띄어쓰기로 단어를 나누는 BM25는 한국어 조사 때문에 8개 중 1개만 점수가 나오는 것을 확인했다
- 같은 문서를 `Chroma.from_documents`로 저장해 벡터 검색을 하고, 조사가 달라도 의미가 가까운 문서를 찾는 것을 비교했다
- 기존 Chroma12 컬렉션에 문서가 추가되어 이전 실습 청크도 검색 결과에 나오는 것을 확인했다
- `EnsembleRetriever`로 BM25 검색기와 벡터 검색기를 합쳐 hybrid 검색을 하고, 두 검색기에 모두 나온 문서가 1등이 되는 것을 확인했다

---

## 💡 주요 학습 포인트

1. **BM25는 키워드 검색**: 질문 단어가 문서에 글자 그대로 있는지로 점수를 매기고, 임베딩 · 벡터 DB가 필요 없다
2. **BM25는 띄어쓰기로 단어를 나눈다**: `반도체를` · `반도체는`은 `반도체`와 다른 단어라 한국어 조사에 약하다
3. **점수가 0이어도 k개를 채운다**: BM25 2 · 3등은 0점 문장이라 질문과 관계없을 수 있다
4. **BM25Retriever는 `.k`, 벡터 검색기는 `search_kwargs={'k':...}`**: 가져올 문서 수를 정하는 방법이 다르다
5. **벡터 검색은 의미로 찾는다**: 조사 · 표현이 달라도 의미가 가까운 문서를 찾는다
6. **같은 컬렉션에 `from_documents`하면 문서가 추가된다**: 이전에 저장한 청크도 함께 검색되고, 다시 실행하면 같은 문서가 또 들어간다
7. **저장된 DB는 `Chroma(embedding_function=...)`로 불러온다**: 문서를 다시 넣지 않고 연결만 한다
8. **hybrid 검색 = `EnsembleRetriever`로 검색기 합치기**: `retrievers`에 검색기 리스트, `weights`에 검색 중요도
9. **hybrid는 순위를 합치고 중복을 뺀다**: 두 검색기에 모두 나온 문서가 위로 올라가고, 3개 + 3개 중 겹친 1개를 빼서 5개
10. **같은 질문 · 같은 문서로 비교해야 검색 방식 차이가 보인다**: 문서 8개와 질문 하나를 세 파일에 그대로 썼다

---

[⬅️ Day29](Day29.md) · [🏠 전체 목차](../README.md)
