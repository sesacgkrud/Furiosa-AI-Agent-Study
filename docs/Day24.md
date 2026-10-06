# Day24 - Jena 기온 예측 Conv2D · Conv1D, Conv1D padding='same', Chroma 벡터 DB 저장 · 검색 · Retriever

**학습 기간:** 2026-10-06

> Day20 ~ 21에서 LSTM · Bidirectional로 풀었던 **Jena Climate 기온 예측**을 이번에는 합성곱으로 풀었다. 먼저 데이터를 **4차원 `(N, 144, 13, 1)`로 바꿔 Conv2D**를 쓰고, 다음으로 **3차원 그대로 Conv1D**를 써서 RMSE와 훈련 시간을 비교했다. 그 전에 Day18 ~ 19의 1 ~ 10 / 범위 밖 예측 데이터를 **Conv1D + `padding='same'` + Flatten**으로 바꿔 11 / 80을 맞췄다. LangChain 쪽에서는 txt 문서를 불러와 **청크로 자르고(TextLoader · RecursiveCharacterTextSplitter)**, OpenAI 임베딩으로 벡터를 만들어 **Chroma 벡터 DB에 저장 → 불러오기 → `similarity_search` · `as_retriever`로 검색**했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| Jena Conv2D | split_x 결과 `(N, 144, 13)` → `reshape(-1, 144, 13, 1)` → Conv2D (3,3) 2층 + Flatten, 기온 RMSE **2.4416** |
| Conv1D | 3차원 `(N, timesteps, feature)` 그대로, 커널이 **시간축만** 따라 움직이며 한 번에 feature 전체를 본다 |
| Jena Conv1D | `(N, 144, 13)` → Conv1D 2층 + Flatten, RMSE **2.4624**, 시간 **320초** (Conv2D 921초) |
| Conv1D param | `(kernel_size × 입력 feature + 1) × filters` |
| `padding='same'` | timesteps가 줄지 않는다 (valid면 kernel_size - 1씩 줄어든다) |
| Conv1D 1 ~ 10 | `[8,9,10]` → **11.0099** (SimpleRNN 10.85) |
| Conv1D 범위 밖 | `[50,60,70]` → **80.0045** (LSTM 78.98) |
| TextLoader | txt 파일 → Document, 한글 파일은 `encoding='utf-8'` |
| RecursiveCharacterTextSplitter | `chunk_size` 언저리로 자르고 `chunk_overlap`만큼 겹친다 |
| Chroma | `from_documents(persist_directory=...)`로 저장, `Chroma(...)`로 불러오기 |
| 검색 | `similarity_search(query, k)` / `as_retriever(search_kwargs={'k':2}).invoke(query)` |

---

## 📖 핵심 학습 내용

### 1. Jena 기온 예측을 Conv2D로 (keras66_jena_CNN)

keras58_kaggle_jena2(LSTM)를 베이스로, y는 keras58_kaggle_jena1에서 저장한 **기온 `T (degC)`** 그대로다.

```python
x = split_x(x_data, size)   # (420120, 144, 13)
y = split_x(y_data, size)   # (420120, 144)

# 3차원 → 4차원 : 맨 뒤에 채널 1 을 붙인다 (값 개수는 그대로)
x = x.reshape(-1, size, 13, 1)                 # (420120, 144, 13, 1)  세로 144 × 가로 13 × 채널 1
x_predict = x_predict.reshape(1, size, 13, 1)  # (1, 144, 13, 1)
```

- 하루치 데이터를 **세로 144(시점) × 가로 13(컬럼) 흑백 이미지**처럼 본다
- 스케일링은 split_x 전 2차원 `(행, 13)`에서 → 4차원 reshape는 그 뒤에

```python
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(size, 13, 1)))  # (None, 142, 11, 32)  param 320
model.add(Conv2D(16, (3,3), activation='relu'))                             # (None, 140, 9, 16)   param 4624
model.add(Flatten())                                                        # (None, 20160)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(size))                                                      # 다음 하루 144개 시점의 기온
```

- padding 기본값(`valid`) → Conv2D를 지날 때마다 세로 · 가로가 2씩 줄어든다 (144×13 → 142×11 → 140×9)
- param = (커널 세로 × 커널 가로 × 입력 채널 + 1) × filters → (3×3×1+1)×32 = **320**, (3×3×32+1)×16 = **4624**
- 출력 `(None, 144)`를 만들려면 Flatten으로 2차원으로 편 뒤 Dense
- 기온은 연속값 회귀 → y를 원핫하지 않고 `loss='mse'`, 출력층 activation 없음
- ModelCheckpoint `_save/keras66/keras66_jena_conv2d.hdf5`
- 결과 : loss **3.5332**, 마지막 하루 기온 RMSE **2.4416**, **921.35초**

### 2. Conv1D 입문 (keras67_Conv1D_1)

keras54_RNN1(1 ~ 10, SimpleRNN)을 Conv1D로 바꿨다.

```python
x = x.reshape(x.shape[0], x.shape[1], 1)   # (7, 3, 1) → (N, timesteps, features)  RNN 과 같은 3차원

model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, padding='same', input_shape=(3, 1)))  # (None, 3, 10)  param 30
model.add(Conv1D(10, 2, padding='same'))                                           # (None, 3, 10)  param 210
model.add(Flatten())                                                               # (None, 30)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
```

- Conv1D는 RNN과 같은 3차원 입력 → 커널이 **시간축(timesteps)만 따라** 움직인다
- `kernel_size=2` : 연속된 2개 시점을 한 번에 본다
- `padding='same'` : 앞뒤를 0으로 채워서 timesteps 3을 유지 (valid면 3 → 2 → 1)
- param = (kernel_size × 입력 feature + 1) × filters → (2×1+1)×10 = **30**, (2×10+1)×10 = **210**
- Conv1D 출력은 3차원 `(None, 3, 10)` → LSTM과 달리 **Flatten**이 필요하다
- `epochs=1000`
- 결과 : `[8,9,10]` → **11.0099** (SimpleRNN 10.8468)

### 3. Conv1D로 범위 밖 예측 (keras67_Conv1D_2_scale)

keras55_LSTM2_scale(1 ~ 13 + 20 ~ 70, `[50,60,70]` → 80)을 Conv1D로 바꿨다.

```python
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, padding='same', input_shape=(3, 1)))  # (None, 3, 10)
model.add(Conv1D(10, 2, padding='same'))                                           # (None, 3, 10)
model.add(Flatten())                                                               # Conv1D 바로 뒤에 한 번만
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
```

- Flatten 전에 Dense를 쌓으면 3차원 그대로 **시점마다 따로** 계산된다 → Flatten은 Conv1D 바로 뒤에
- 이미 2차원인 데이터에 Flatten을 한 번 더 해도 모양이 바뀌지 않는다
- 결과 : loss **0.000175**, `[50,60,70]` → **80.0045** (LSTM 78.9794)

### 4. Jena 기온 예측을 Conv1D로 (keras67_Conv1D_3_jena)

keras66(Conv2D 4차원)을 Conv1D로 바꿨다.

```python
# Conv1D 는 3차원 (N, timesteps, feature) → split_x 결과 (420120, 144, 13) 를 reshape 없이 그대로
x_predict = x_predict.reshape(1, size, 13)     # (1, 144, 13)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',
                 input_shape=(size, 13)))                    # (None, 144, 64)  param (3×13+1)×64 = 2560
model.add(Conv1D(32, 3, padding='same', activation='relu'))  # (None, 144, 32)  param (3×64+1)×32 = 6176
model.add(Flatten())                                         # (None, 4608)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(size))
```

- 커널이 시간축(144)을 따라 움직이고, 한 번에 **kernel_size개 시점 × 13개 컬럼 전체**를 본다
- 4차원 `(144, 13, 1)`로 넣으면 에러 없이 돌아가지만 Conv1D가 시간축이 아니라 **컬럼축(13)**을 따라 움직인다 → 시간 흐름을 못 배운다
- `padding='same'` → timesteps 144 유지
- ModelCheckpoint `_save/keras67/keras67_jena_conv1d.hdf5`
- 결과 : loss **3.8669**, 마지막 하루 기온 RMSE **2.4624**, **320.37초**

### 5. 문서 불러오기 · 청킹 · Chroma 저장 (rag11_Chroma01_save)

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

path = './_data/rag_data/'
# UnicodeDecodeError: 'cp949' codec can't decode ...  → encoding 지정
loader1 = TextLoader(path + 'samsung_outlook.txt', encoding='utf-8')
loader2 = TextLoader(path + 'nvidia_outlook.txt', encoding='utf-8')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,                        # 300 언저리로 자른다
    chunk_overlap=100,                     # 자를 때 겹치는 구간
    separators=['\n\n', '\n', ' ', ''],    # 문단 → 줄 → 띄어쓰기 → 글자 순서로 자를 곳을 찾는다
)
split_doc1 = loader1.load_and_split(text_splitter)
split_doc2 = loader2.load_and_split(text_splitter)
print(len(split_doc1), len(split_doc2))    # 9 9

db = Chroma.from_documents(
    documents=split_doc1 + split_doc2,     # 청크 18개
    embedding=embeddings,                  # OpenAIEmbeddings(text-embedding-3-small)
    persist_directory='./_db/Chroma11/',   # 폴더에 파일로 저장
    collection_name='croma11',
)
```

- Windows에서 `encoding`을 빼면 기본 cp949로 읽어 한글 utf-8 파일에서 `UnicodeDecodeError`
- `load_and_split(splitter)` : 불러오기 + 청킹을 한 번에 → Document 리스트
- `from_documents` : 청크마다 임베딩 API를 호출해 벡터로 바꾸고 DB에 저장
- `persist_directory`를 주면 `chroma.sqlite3` 등이 폴더에 남는다

### 6. 저장한 Chroma 불러오기 · 검색 (rag11_Chroma02_load)

```python
db = Chroma(
    embedding_function=embeddings,         # 질문을 벡터로 바꿀 때 사용 (저장 때와 같은 모델)
    persist_directory='./_db/Chroma11/',
    collection_name='croma11',
)
aaa = db.similarity_search('삼성전자 사업 전망에 대해 알려줘', k=2)   # default k = 4
```

- 문서 불러오기 · 청킹 · 저장 코드는 주석 처리 → 임베딩을 다시 하지 않고 저장된 DB를 연다
- `persist_directory` / `collection_name`이 저장할 때와 같아야 같은 DB를 찾는다
- `similarity_search` : 질문 벡터와 가장 가까운 청크 k개를 Document로 돌려준다

### 7. 폴더 전체 불러오기 · split_documents · Retriever (rag12_Chroma03_save)

```python
from glob import glob

txt_files = glob(os.path.join(path, '*.txt'))   # 폴더 안의 txt 3개 목록

data = []
for text_file in txt_files:
    loader = TextLoader(text_file, encoding='utf-8')
    data += loader.load()                       # Document 리스트를 이어 붙인다
print(len(data))                                # 3
print([len(doc.page_content) for doc in data])  # [8158, 2049, 1898]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=10,
                                               separators=['\n\n', '\n', ' ', ''])
texts = text_splitter.split_documents(data)     # 문서 3개를 한 번에 청킹
print(len(texts))                               # 52
```

- 파일 이름을 하나씩 쓰지 않고 `glob('*.txt')`로 목록을 만든다
- `split_documents` : 이미 불러온 Document 리스트를 자른다 (`load_and_split`은 불러오기 + 자르기)
- 청크 길이는 `chunk_size` 미만이거나 언저리 (첫 청크 259, 두 번째 282)
- `list()` 없이 `(len(...) for ...)`만 print하면 값이 아니라 `<generator object ...>`가 나온다

```python
vector_store = Chroma.from_documents(documents=texts, embedding=embeddings,
                                     persist_directory='./_db/Chroma12/', collection_name='croma12')
print(vector_store._collection.count())         # 52

query = '삼성전자의 창업자는 누구인가요?'
result = vector_store.similarity_search(query)  # 4개 (default)

retriever = vector_store.as_retriever(search_kwargs={'k':2})
aaa = retriever.invoke(query)                   # 2개
print(aaa[0].page_content[:50])                 # 삼성전자 사업 전망 / 삼성전자는 메모리 반도체, ...
```

- `as_retriever` : 벡터 DB를 **검색기**로 바꾼다 → `invoke(질문)`으로 관련 문서를 가져온다 (LCEL 체인에 연결할 수 있는 형태)
- `embed_query`로 질문 벡터 차원 확인 → **1536** (text-embedding-3-small)
- 검색기는 가장 가까운 청크를 **원문 그대로** 돌려줄 뿐 답을 만들지 않는다
  - rag_data 3개 파일에는 창업자 내용이 없다 → 그래도 가장 가까운 k개(`삼성전자`가 들어간 청크)를 돌려준다
  - `[:50]`으로 앞 50글자만 출력해서 중간에 잘려 보인다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras2/keras66_jena_CNN.py` | keras58_kaggle_jena2 → `reshape(-1, 144, 13, 1)` 4차원 + Conv2D (3,3) 2층 + Flatten, 기온 RMSE 2.4416 |
| `keras2/keras67_Conv1D_1.py` | keras54_RNN1 → Conv1D(10, 2, `padding='same'`) 2층 + Flatten, `[8,9,10]` → 11.0099 |
| `keras2/keras67_Conv1D_2_scale.py` | keras55_LSTM2_scale → Conv1D same 2층 + Flatten 한 번, `[50,60,70]` → 80.0045 |
| `keras2/keras67_Conv1D_3_jena.py` | keras66 → 3차원 그대로 Conv1D(64, 3) · Conv1D(32, 3) same + Flatten, 기온 RMSE 2.4624, 320초 |
| `RAG/rag11_Chroma01_save.py` | TextLoader(utf-8) 2개, RecursiveCharacterTextSplitter(300, 100), `load_and_split`, `Chroma.from_documents` 저장 |
| `RAG/rag11_Chroma02_load.py` | `Chroma(persist_directory, collection_name)` 불러오기, `similarity_search(k=2)` |
| `RAG/rag12_Chroma03_save.py` | `glob('*.txt')` + 반복문 TextLoader, `split_documents` (52청크), `_collection.count()`, `as_retriever(k=2)` + `invoke` |
| `_data/rag_data/*.txt` | RAG 실습용 텍스트 3개 (samsung_outlook / nvidia_outlook / 2026_AI_for_All) |
| `.gitignore` | Chroma DB 폴더 `_db/` 제외 추가 (수정) |
| `docs/Day24.md` | Day24 학습 기록 신규 작성 |
| `docs/Day23.md` | 하단 nav에 Day24 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day24, `_save/keras66 · keras67`, `_db`, `rag_data`) / 진행도(24일, 30%) 갱신 (수정) |

---

## 📊 실행 결과

### Jena 기온 예측 (y = `T (degC)`, 같은 데이터 / 같은 split)

| 파일 | 모델 | 입력 | test loss | 마지막 하루 기온 RMSE | 시간 |
|---|---|:---:|:---:|:---:|:---:|
| keras58_kaggle_jena2 | LSTM(64) | (144, 13) | 3.2113 | 4.8036 | 484.4초 |
| keras59_Bidirectional3_jena | Bidirectional(LSTM(64)) | (144, 13) | 4.1010 | **2.1046** | 919.26초 |
| keras66_jena_CNN | Conv2D (3,3) 2층 | (144, 13, 1) | 3.5332 | 2.4416 | 921.35초 |
| keras67_Conv1D_3_jena | Conv1D 3 2층 same | (144, 13) | 3.8669 | 2.4624 | **320.37초** |

**읽는 법**
- Conv1D는 Conv2D와 RMSE가 거의 같으면서 시간은 약 **1/3**
  - Conv2D는 3×3 커널로 컬럼 3개씩만 보지만, Conv1D는 한 번에 13개 컬럼 전체를 보고 시간축만 따라 움직인다
  - LSTM처럼 시점을 하나씩 순서대로 계산하지 않아서 빠르다
- RMSE는 Bidirectional이 가장 낮았지만 시간은 Conv1D의 약 3배

### Conv1D 작은 데이터

| 파일 | 베이스 (결과) | Conv1D 결과 | 목표 |
|---|---|:---:|:---:|
| keras67_Conv1D_1 | SimpleRNN (10.8468) | **11.0099** | 11 |
| keras67_Conv1D_2_scale | LSTM (78.9794) | **80.0045** | 80 |

### RAG

| 파일 | 확인한 값 |
|---|---|
| rag11_Chroma01_save | 청크 수 samsung 9 / nvidia 9 (chunk 300, overlap 100) |
| rag12_Chroma03_save | 문서 3개 글자 수 [8158, 2049, 1898] → 청크 52개, DB 저장 문서 수 52 |
| rag12_Chroma03_save | `embed_query` 벡터 1536차원, `similarity_search` 4개 (default), retriever 2개 |

---

## 💻 핵심 개념

### Conv2D vs Conv1D 입력

```
Conv2D : (N, 세로, 가로, 채널)   (N, 144, 13, 1)   커널 (3,3) 이 세로 · 가로 두 방향으로 움직인다
Conv1D : (N, timesteps, feature) (N, 144, 13)      커널 3 이 시간축 한 방향으로 움직인다, 한 번에 feature 전체
```

### 파라미터

```
Conv2D Param = (커널 세로 × 커널 가로 × 입력 채널 + 1) × filters
Conv1D Param = (kernel_size × 입력 feature + 1) × filters

Conv1D(64, 3), feature 13 → (3 × 13 + 1) × 64 = 2560
Conv1D(32, 3), feature 64 → (3 × 64 + 1) × 32 = 6176
```

### padding

```
valid (기본값) : 출력 길이 = 입력 길이 - kernel_size + 1     3 → 2 → 1
same           : 앞뒤를 0 으로 채워 출력 길이 = 입력 길이       3 → 3 → 3
```

### RAG 문서 저장 · 검색 흐름

```
TextLoader(encoding='utf-8') ──▶ Document (page_content + metadata)
        │
        ▼
RecursiveCharacterTextSplitter(chunk_size, chunk_overlap) ──▶ 청크 Document 리스트
        │  load_and_split(splitter) / split_documents(docs)
        ▼
Chroma.from_documents(documents, embedding, persist_directory, collection_name)
        │  청크마다 OpenAIEmbeddings 로 벡터 → 폴더에 저장
        ▼
Chroma(embedding_function, persist_directory, collection_name)   ← 다시 임베딩하지 않고 불러오기
        │
        ▼
similarity_search(query, k)  /  as_retriever(search_kwargs={'k':2}).invoke(query)
        │
        ▼
질문과 가장 가까운 청크 k개 (원문 그대로, 답은 만들지 않음)
```

---

## 성과

- Jena 기온 데이터를 `(N, 144, 13, 1)` 4차원으로 바꿔 Conv2D로 기온 RMSE **2.4416** (LSTM 4.8036)
- 같은 데이터를 3차원 그대로 Conv1D로 바꿔 RMSE **2.4624**, 훈련 시간 921초 → **320초**
- Conv1D + `padding='same'` + Flatten으로 `[8,9,10]` → **11.0099**, `[50,60,70]` → **80.0045**
- Conv1D / Conv2D param 공식과 `padding='same'` / `valid`의 출력 길이 차이 확인
- TextLoader(`encoding='utf-8'`)로 한글 txt를 불러오고 RecursiveCharacterTextSplitter로 청킹 (9 / 9, 52개)
- `Chroma.from_documents`로 벡터 DB를 폴더에 저장하고 `Chroma(...)`로 다시 불러와 `similarity_search`
- `glob`으로 폴더 안의 txt를 모두 불러오고 `split_documents`로 한 번에 청킹
- `as_retriever(search_kwargs={'k':2})` + `invoke`로 검색기를 만들고, 검색기는 답이 아니라 원문 청크를 돌려준다는 것 확인

---

## 💡 주요 학습 포인트

1. **Conv2D에 넣으려면 채널 축을 붙인다**: `(N, 144, 13)` → `reshape(-1, 144, 13, 1)`, 값 개수는 그대로
2. **시계열을 이미지처럼 볼 수 있다**: 세로 = 시점 144, 가로 = 컬럼 13
3. **valid Conv2D는 지날 때마다 커널 크기 - 1씩 줄어든다**: (3,3)이면 144×13 → 142×11 → 140×9
4. **Conv2D Param = (커널 세로 × 커널 가로 × 입력 채널 + 1) × filters**
5. **회귀 y는 원핫하지 않는다**: 기온처럼 연속값이면 y 그대로, `mse`, 출력층 activation 없음
6. **Conv1D 입력은 RNN과 같은 3차원**: `(N, timesteps, feature)`
7. **Conv1D 커널은 시간축만 따라 움직인다**: 한 번에 kernel_size개 시점 × feature 전체를 본다
8. **Conv1D에 4차원을 넣으면 엉뚱한 축을 따라 움직인다**: 에러가 나지 않아서 더 조심해야 한다
9. **Conv1D Param = (kernel_size × 입력 feature + 1) × filters**: (3 × 13 + 1) × 64 = 2560
10. **`padding='same'`은 timesteps를 유지한다**: 시점이 3개뿐인 데이터에서 특히 중요하다
11. **Conv1D 출력은 3차원이라 Flatten이 필요하다**: LSTM(return_sequences=False)과 다른 점
12. **Flatten은 Conv 층 바로 뒤에 한 번만**: 3차원에 Dense를 쌓으면 시점마다 따로 계산된다
13. **Conv1D는 LSTM보다 빠르다**: 시점을 순서대로 계산하지 않는다 (Jena 320초 vs Bidirectional 919초)
14. **Conv1D는 Conv2D보다 빠르면서 RMSE가 비슷했다**: 2.4624 vs 2.4416, 320초 vs 921초
15. **한글 txt는 `encoding='utf-8'`**: Windows 기본 cp949로 읽으면 UnicodeDecodeError
16. **청킹은 `chunk_size` 언저리로 자른다**: separators 순서(문단 → 줄 → 띄어쓰기 → 글자)대로 자를 곳을 찾아서 정확히 300이 아니다
17. **`chunk_overlap`은 청크끼리 겹치는 구간**: 문장이 경계에서 잘려도 앞뒤 청크에 내용이 남는다
18. **`load_and_split` vs `split_documents`**: 불러오기 + 자르기 / 이미 불러온 Document 자르기
19. **`persist_directory`로 벡터 DB를 파일로 남긴다**: 다시 실행할 때 임베딩 API를 또 호출하지 않는다
20. **불러올 때는 `Chroma(...)`**: `persist_directory` · `collection_name` · 같은 임베딩 모델
21. **`similarity_search` 기본 k는 4**: 질문 벡터와 가장 가까운 청크 k개
22. **`glob`으로 폴더의 파일 목록을 만든다**: 파일이 늘어나도 코드를 고치지 않는다
23. **`as_retriever`는 벡터 DB를 검색기로 바꾼다**: `invoke(질문)`으로 관련 문서를 가져온다
24. **검색은 답이 아니다**: 문서에 없는 내용을 물어도 가장 가까운 k개를 돌려준다 → 답 문장은 LLM 단계가 만든다

---

[⬅️ Day23](Day23.md) · [🏠 전체 목차](../README.md)
