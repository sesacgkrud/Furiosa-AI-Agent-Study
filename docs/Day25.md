# Day25 - Conv1D · MaxPooling1D 데이터셋 적용, RAG 체인 · Gradio 챗봇, FAISS 벡터 DB 저장 · 불러오기

**학습 기간:** 2026-10-07

> Day24에서 배운 **Conv1D**를 Day14 ~ 23에서 Conv2D · LSTM으로 풀었던 표 데이터(diabetes · 따릉이 · cancer · wine · digits)와 이미지 데이터(fashion · cifar100)에 적용했다. Conv1D 뒤에 **MaxPooling1D**로 시점 축을 절반씩 줄이고 **Dropout**으로 과적합을 막아 기존 기록과 비교했다. LangChain 쪽에서는 Day24의 Chroma 검색 결과를 **ChatOpenAI에 컨텍스트로 넘겨 답을 만들고(RAG)**, `ChatPromptTemplate` + `create_stuff_documents_chain` + `create_retrieval_chain`으로 **체인**을 만든 뒤 **Gradio `ChatInterface`로 챗봇 화면**을 띄웠다. 마지막으로 벡터 DB를 **FAISS**로 바꿔 `save_local` / `load_local`로 저장 · 불러오고, 관련 없는 질문 거르기 · 이전 대화 · 출처 표시를 넣은 FAISS 챗봇을 만들었다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| Conv1D 표 데이터 | 컬럼 N개 → `reshape(-1, N, 1)` → 컬럼을 시점처럼 보고 Conv1D |
| Conv1D 이미지 | `(28, 28)` / `(32, 32×3)` → 이미지 한 행을 한 시점으로 본다 |
| MaxPooling1D | `pool_size=2` → 시점 축이 절반 (9 → 4, 13 → 6 → 3), param 0 |
| 출력층 | 회귀 `Dense(1)` / 이진 `Dense(1, sigmoid)` / 다중 `Dense(클래스 수, softmax)` |
| Conv1D 결과 | diabetes r2 **0.4994** · 따릉이 r2 **0.7335** · wine acc **0.981** · digits acc **0.979** (Conv2D · LSTM 보다 좋음) |
| `temperature` | 0 → 같은 질문에 거의 같은 답 / 1 → 물을 때마다 표현이 달라진다 |
| 컨텍스트 직접 넣기 | `f'{문서}\n\n위 내용에 근거하여 ... {query}'` → `model.invoke` |
| RAG 체인 | `create_stuff_documents_chain(model, prompt)` → `create_retrieval_chain(retriever, docu_chain)` |
| 체인 결과 | `dict_keys(['input', 'context', 'answer'])` |
| Gradio | `gr.ChatInterface(fn=answer_invoke, title=...)` → `demo.launch()` |
| FAISS | `FAISS.from_documents` → `save_local(folder_path, index_name)` → `load_local(..., allow_dangerous_deserialization=True)` |
| FAISS 챗봇 고도화 | `similarity_score_threshold` 거르기 · `from_messages` system 규칙 · history · 출처 · examples |

---

## 📖 핵심 학습 내용

### 1. 표 데이터에 Conv1D + MaxPooling1D (keras68_Conv1D_02 ~ 10)

keras64(LSTM)에서 쓰던 3차원 reshape를 그대로 쓰고, 모델만 Conv1D로 바꿨다.

```python
x_train = x_train.reshape(-1, 10, 1)       # diabetes : 컬럼 10개를 시점 10개로, 시점마다 값 1개

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(10, 1)))  # (None, 10, 64)
model.add(Conv1D(64, 3, padding='same', activation='relu'))   # (None, 10, 64)
model.add(MaxPooling1D(pool_size=2))                          # (None, 5, 64)  시점 축을 절반으로
model.add(Conv1D(32, 2, padding='same', activation='relu'))   # (None, 5, 32)
model.add(Dropout(0.2))
model.add(Flatten())                                          # (None, 160)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))                                           # 회귀 : 숫자 하나
```

- **MaxPooling1D(pool_size=2)** : 시점 2개씩 묶어 큰 값 하나만 남긴다 → 길이가 절반 (홀수면 내림, 9 → 4)
- 파라미터가 없다 (param 0) → 모델을 키우지 않고 Flatten 뒤 크기를 줄인다
- Conv1D 출력은 3차원이라 **Flatten**으로 편 뒤 Dense
- 데이터마다 바꾼 부분

| 파일 | 입력 | MaxPooling1D | 출력층 / loss | 그 밖 |
|---|---|---|---|---|
| 02_diabetes | `(10, 1)` | 1번 (10 → 5) | `Dense(1)` / mse | patience 30, epochs 300, batch 16 |
| 04_ddacon_ddareung | `(9, 1)` | 1번 (9 → 4) | `Dense(1)` / mse | patience 30 |
| 06_cancer | `(30, 1)` | 2번 (30 → 15 → 7) | `Dense(1, sigmoid)` / binary_crossentropy | patience 30, epochs 300 |
| 08_wine | `(13, 1)` | 2번 (13 → 6 → 3) | `Dense(3, softmax)` / categorical_crossentropy | `to_categorical` y |
| 10_digits | `(8, 8)` | 2번 (8 → 4 → 2) | `Dense(10, softmax)` / categorical_crossentropy | 64픽셀을 8행 × 8열로 |

- 이진 분류 + `binary_crossentropy`는 출력을 0 ~ 1 확률로 만드는 **sigmoid**
- 다중 분류 + `categorical_crossentropy`는 클래스 수만큼 칸을 두고 **softmax** (`to_categorical`로 y가 `(N, 3)`)
- 기록 비교는 아래 **실행 결과** 표

### 2. 이미지 데이터에 Conv1D (keras68_Conv1D_12_fashion · 14_cifar100)

Conv1D는 3차원 `(N, timesteps, feature)`만 받는다 → 이미지의 채널 축을 없애고 **한 행을 한 시점**으로 본다.

```python
# fashion : (60000, 28, 28) → 행 28개 = 시점 28개, 시점마다 픽셀 28개
x_train = x_train.reshape(-1, 28, 28)
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28)))
...
model.add(Dense(10, activation='softmax'))   # 옷 클래스 10개, y 는 정수 그대로 (sparse_categorical_crossentropy)

# cifar100 : (50000, 32, 32, 3) → 한 행의 픽셀 32개 × RGB 3 = 96 을 한 시점의 feature 로
x_train = x_train.reshape(-1, 32, 32*3)      # (50000, 32, 96)
```

- cifar100은 클래스가 100개라 모델을 키웠다
  - filters 128 · 128 · 64 · 64, kernel_size 3
  - Conv1D 뒤 **BatchNormalization** 2곳
  - MaxPooling1D 2번 (32 → 16 → 8)
  - Dropout 0.3, Dense 256 · 128
  - Total params 301,156
- EarlyStopping `patience`가 `epochs`와 같으면(100 / 100) 멈출 기회가 없다 → fashion은 patience 10
- ModelCheckpoint 파일 이름은 데이터마다 다르게 (`keras68_conv1d_12_fashion.keras`) → 다른 모델 파일을 덮어쓰지 않는다
- 결과 : fashion acc **0.8996** (Conv2D 0.9258), cifar100 acc **0.3756** (Conv2D + MaxPooling 0.4371)
  - Conv1D는 커널이 위아래 한 방향으로만 움직이며 한 번에 행 전체를 본다
  - Conv2D처럼 가로 · 세로 3×3 작은 무늬를 따로 잡지 못해 이미지에서는 Conv2D보다 낮았다
  - cifar100은 마지막 epoch 훈련 acc 0.564 vs test 0.376 → 과적합

### 3. 저장한 Chroma DB 불러와 검색 (rag12_Chroma04_load)

rag12_Chroma03_save의 불러오기 · 청킹 · 저장 코드는 `''' '''`로 막고, `Chroma(...)`로 저장된 DB만 열었다.

```python
vector_store = Chroma(embedding_function=embeddings, persist_directory='./_db/Chroma12/', collection_name='croma12')
print(vector_store._collection.count())        # 208

result = vector_store.similarity_search(query)  # 4개 (default)
retriever = vector_store.as_retriever(search_kwargs={'k':2})
aaa = retriever.invoke(query)                   # 2개
```

- 문서 수가 **208 = 52 × 4** → 저장 코드(`from_documents`)를 4번 실행해서 같은 청크가 4번씩 쌓였다
  - `from_documents`는 같은 `persist_directory` · `collection_name`이면 덮어쓰지 않고 **추가**한다
  - 그래서 검색하면 같은 청크가 겹쳐 나온다 (retriever k=2에 같은 문장 2개)

### 4. 검색 결과를 LLM에 컨텍스트로 넘기기 (rag13_Chroma_pipeline1)

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0,      # 0 -> 같은 질문에 거의 같은 답 | 1 -> 물을 때마다 표현이 달라진다
    max_tokens=1000,    # 답변 최대 길이
    api_key=api_key,
    base_url=base_url,  # monorouter
)

response = model.invoke('엔비디아는 어떤 기업인가요?')          # 문서 없이 → 모델이 아는 지식으로 길게 답한다

query_with_context = f'''
    {aaa[0].page_content}\n\n
    위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
'''
response = model.invoke(query_with_context)                       # 검색한 청크를 근거로 답한다
```

- **temperature** : 다음 단어를 고를 때 얼마나 무작위로 고를지
  - 0 → 확률이 가장 높은 단어만 → 사실 확인 · RAG 답변
  - 1 → 확률 분포대로 뽑아 표현이 달라진다 → 아이디어 · 글쓰기
- 문서 없이 물으면 설립 연도 · 본사 · 아키텍처 이름까지 **모델이 아는 지식**으로 답한다
- 검색한 청크를 앞에 붙이면 문서 내용(GPU · 데이터센터 가속기 · 소프트웨어 생태계)을 중심으로 답한다
- 삼성전자 창업자처럼 **문서에 없는 질문**은 컨텍스트를 줘도 모델이 아는 지식(이병철)으로 답해 버렸다 → 다음 단계에서 프롬프트로 막는다

### 5. RAG 체인 만들기 (rag14_Chroma_pipeline2)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template('''
    다음 컨텍스트를 바탕으로 질문에 답변해 주세요.
    컨텍스트 관련 정보가 없다면, "주어진 정보로는 답변할 수 없습니다." 라고 말씀해주세요.

    컨텍스트 : {context}
    질문 : {input}
    답변 :
''')

docu_chain = create_stuff_documents_chain(model, prompt)   # 문서 리스트를 {context} 에 채워 넣고 모델 호출 (prompt | model)
rag_chain = create_retrieval_chain(retriever, docu_chain)  # {input} 으로 검색 → 찾은 문서를 docu_chain 에

response = rag_chain.invoke({'input': '삼성전자의 창업자는 누구인가요?'})
print(response.keys())                   # dict_keys(['input', 'context', 'answer'])
print(response['context'][0].page_content)  # 검색한 청크 (삼성전자 사업 전망 ...)
print(response['answer'])                # 주어진 정보로는 답변할 수 없습니다.
```

- `{context}`, `{input}` 이름이 정해져 있다 → `create_retrieval_chain`이 `input`으로 검색하고 결과를 `context`로 넘긴다
- `invoke`에는 `{'input': 질문}` 딕셔너리를 넣는다
- 결과 딕셔너리에서 `context`(근거 문서)와 `answer`(답)를 따로 꺼낼 수 있다
- 프롬프트에 "정보가 없으면 답변할 수 없다"를 넣자 rag13과 달리 창업자 질문을 거절했다

### 6. Gradio 챗봇 (rag15_Chroma_gradio · rag16_chatbot_practice)

```python
import gradio as gr

def answer_invoke(message, history):     # message : 입력한 질문, history : 이전 대화
    response = rag_chain.invoke({'input': message})
    return response['answer']             # 화면에는 답만

demo = gr.ChatInterface(fn=answer_invoke, title='AI Chat Bot')
demo.launch()                             # 로컬 주소 (http://127.0.0.1:7860) 로 챗봇 화면
# demo.launch(share=True)                 # 외부에서 접속할 수 있는 공유 URL
```

- `ChatInterface`는 채팅 화면을 만들어 주고, 입력할 때마다 `fn(message, history)`를 부른다
- rag15에서 벡터 DB에 없는 질문(대한민국 수도, 파이썬 정렬)에도 답이 나왔다
  - API는 monorouter로 정상 호출되고 있었다 (`base_url`)
  - retriever는 관련이 없어도 **가장 가까운 문서 k개를 항상** 넘긴다
  - `gpt-5-nano`가 "정보가 없으면 거절" 지시를 항상 지키지는 않았다 ("일반 상식으로는 서울이 …")
- rag16 : 프롬프트를 "컨텍스트 밖의 지식은 절대 쓰지 말고, 없으면 정확히 '주어진 정보로는 답변할 수 없습니다.'만 출력"으로 강화

### 7. FAISS 벡터 DB 저장 (rag17_FAISS_1_save)

```python
# pip install faiss-cpu
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query('Hello World!')))  # 1536차원, L2(유클리드) 거리로 검색
print(faiss_index.d)                                                           # 1536

faiss_db = FAISS(embedding_function=embeddings, index=faiss_index,
                 docstore=InMemoryDocstore(),      # 문서 원문을 메모리에 보관
                 index_to_docstore_id={})          # 벡터 번호 → 문서 id
print(faiss_db.index.ntotal)                       # 0 (빈 저장소)

db = FAISS.from_documents(documents=split_doc1 + split_doc2, embedding=embeddings)  # 청크 18개
db.save_local(folder_path='./_db/Faiss17', index_name='faiss_index17')
```

- `IndexFlatL2(차원)` : 차원 수는 임베딩 벡터 길이와 같아야 한다 → `embed_query`로 길이를 재서 넣는다
- 직접 만든 빈 FAISS는 `ntotal` 0 → `from_documents`는 인덱스 생성 + 임베딩 + 추가를 한 번에
- `save_local`은 `faiss_index17.faiss`(벡터 인덱스)와 `faiss_index17.pkl`(문서 원문 · id 매핑) 두 파일을 만든다
- Chroma와 달리 메모리에서 만들고 **`save_local`을 불러야** 파일로 남는다 (`folder_path` 인자 이름)

### 8. FAISS 불러오기 · 검색 (rag17_FAISS_2_load)

```python
db = FAISS.load_local(
    folder_path='./_db/Faiss17',
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,   # pkl 파일을 읽는 것을 허용 (내가 만든 파일만)
)
print(db.index_to_docstore_id)   # {0: '8944...', 1: '5a0c...', ..., 17: '779b...'}  18개
print(db.docstore._dict)         # {id: Document(metadata={'source': ...}, page_content=...)}
aaa = db.similarity_search('삼성전자 창업주에 대해 알려줘', k=2)   # samsung_outlook 청크 2개
```

- pkl은 불러올 때 코드가 실행될 수 있어서 `allow_dangerous_deserialization=True`를 직접 줘야 열린다
- `index_to_docstore_id`(벡터 번호 → id)와 `docstore`(id → Document)로 벡터와 원문이 연결된다
- Document의 `metadata['source']`에 원본 파일 경로가 남는다

### 9. FAISS 챗봇 고도화 (rag18_FAISS_gradio_with_FAISS_upgrade)

FAISS DB를 불러와 rag15의 챗봇 구조로 만들고, 4가지를 더했다.

**① 관련 없는 질문 거르기**

```python
retriever = db.as_retriever(
    search_type='similarity_score_threshold',            # 점수가 기준보다 낮은 문서는 버린다
    search_kwargs={'k': 4, 'score_threshold': 0.05},
)
```

- 기준값은 `similarity_search_with_relevance_scores`로 점수를 찍어 보고 정했다
  - DB 안 질문 : 0.139 ~ 0.493
  - DB 밖 질문(김치찌개 · 대한민국 수도 · 날씨) : -0.125 이하
- 남은 문서가 없으면 모델을 부르지 않고 바로 `"주어진 정보로는 답변할 수 없습니다."` → API 호출도 아낀다
- `k=4` : 제목 한 줄짜리 청크(`엔비디아 사업 전망`)가 1등으로 뽑힐 때가 있어서 2개로는 내용이 부족했다

**② 규칙은 system 메시지로**

```python
prompt = ChatPromptTemplate.from_messages([
    ('system', '''너는 삼성전자 · 엔비디아 전망 문서만 보고 답하는 챗봇이다. ...
2. 컨텍스트에 없는 정의, 배경 설명, 숫자, 인물, 예시를 일반 상식으로 덧붙이지 마라.
3. 관련 내용이 전혀 없을 때만 정확히 "{no_answer}" 한 문장만 출력해라. ...
이전 대화 : {history}
컨텍스트 : {context}'''),
    ('human', '{input}'),
])
```

- `from_template` 한 덩어리보다 규칙(system)과 질문(human)을 나누면 규칙을 더 잘 지켰다
- 규칙을 너무 세게만 쓰면 문서에 있는 질문까지 거절했다 → "관련 내용이 있으면 정리해서 답하라"를 함께 넣었다
- 모델이 거절을 "명시되어 있지 않습니다"처럼 다르게 말할 때가 있어 짧은 답에 `'수 없' / '있지 않' / '되지 않'`이 있으면 고정 문구로 통일

**③ 이전 대화 이어서 묻기**

- Gradio 6의 `history`는 `[{'role': 'user', 'content': ...}, {'role': 'assistant', ...}]` → 최근 3턴을 문자열로 `{history}`에
- 검색어 = **직전 질문 + 현재 질문** → "그 회사의 위험 요인은?"만으로는 점수 0.009로 걸러지지만, 앞 질문(엔비디아)을 붙이면 엔비디아 위험 요인을 찾는다
- `create_retrieval_chain` 대신 `retriever.invoke()` → `docu_chain.invoke({'context': docs, 'input': ..., 'history': ...})`로 직접 연결

**④ 출처 표시 · 화면**

```python
sources = sorted({os.path.basename(d.metadata['source']) for d in docs})
return f'{answer}\n\n📄 출처: {", ".join(sources)}'

demo = gr.ChatInterface(
    fn=answer_invoke,
    title='📊 삼성전자 · 엔비디아 RAG 챗봇',
    description='...',
    chatbot=gr.Chatbot(height=500, placeholder='### 무엇이든 물어보세요 ...'),
    textbox=gr.Textbox(placeholder='삼성전자나 엔비디아의 사업 전망을 물어보세요', scale=7),
    examples=['삼성전자의 주요 사업은 뭐야?', '엔비디아의 데이터센터 사업은 어때?',
              '그 회사의 위험 요인은?', '김치찌개 맛있게 끓이는 법 알려줘'],
)
demo.launch(theme=gr.themes.Soft())   # Gradio 6 부터 theme 는 launch() 에서
```

- `examples` 버튼과 `placeholder`는 대화가 비어 있을 때 첫 화면에 보인다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras2/keras68_Conv1D_02_diabetes.py` | `(10, 1)` Conv1D 3층 + MaxPooling1D + Dropout, r2 0.4994 |
| `keras2/keras68_Conv1D_04_ddacon_ddareung.py` | `(9, 1)` Conv1D + MaxPooling1D(9 → 4) + Dropout, r2 0.7335 |
| `keras2/keras68_Conv1D_06_cancer.py` | `(30, 1)` MaxPooling1D 2번, `Dense(1, sigmoid)`, acc 0.965 · loss 0.0970 |
| `keras2/keras68_Conv1D_08_wine.py` | `(13, 1)` MaxPooling1D 2번, `Dense(3, softmax)`, acc 0.981 |
| `keras2/keras68_Conv1D_10_digits.py` | `(8, 8)` MaxPooling1D 2번, `Dense(10, softmax)`, batch 4, acc 0.979 |
| `keras2/keras68_Conv1D_12_fashion.py` | `(28, 28)` 행 = 시점, `Dense(10, softmax)`, patience 10, acc 0.8996 |
| `keras2/keras68_Conv1D_14_cifar100.py` | `(32, 96)` filters 128 · BatchNormalization · Dropout 0.3, acc 0.3756 |
| `RAG/rag12_Chroma04_load.py` | Chroma12 불러오기, 문서 수 208 (52 × 4), `similarity_search` · `as_retriever(k=2)` |
| `RAG/rag13_Chroma_pipeline1.py` | `ChatOpenAI(temperature, max_tokens)`, 문서 없이 답 vs 검색 청크를 컨텍스트로 넣은 답 |
| `RAG/rag14_Chroma_pipeline2.py` | `ChatPromptTemplate`, `create_stuff_documents_chain`, `create_retrieval_chain`, 결과 keys |
| `RAG/rag15_Chroma_gradio.py` | `gr.ChatInterface(fn, title)` + `demo.launch()` Chroma RAG 챗봇 |
| `RAG/rag16_chatbot_practice.py` | rag15 프롬프트를 "컨텍스트 밖 지식 금지"로 강화 |
| `RAG/rag17_FAISS_1_save.py` | `faiss.IndexFlatL2`, `InMemoryDocstore`, `FAISS.from_documents`, `save_local` |
| `RAG/rag17_FAISS_2_load.py` | `FAISS.load_local(allow_dangerous_deserialization)`, `index_to_docstore_id`, `docstore._dict`, `similarity_search` |
| `RAG/rag18_FAISS_gradio_with_FAISS_upgrade.py` | FAISS 챗봇 + `similarity_score_threshold` · `from_messages` · history · 출처 · examples · theme |
| `keras1/dataset.txt` | 데이터셋 목록에 16 jena 추가 (수정) |
| `docs/Day25.md` | Day25 학습 기록 신규 작성 |
| `docs/Day24.md` | 하단 nav에 Day25 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day25, `_save/keras68`, `_db/Faiss17`) / 진행도(25일, 31%) 갱신 (수정) |
| `.gitignore` | `_db/` 주석에 FAISS(rag17) 추가 (수정) |

---

## 📊 실행 결과

### 표 데이터 (GPU)

| 데이터 | 지표 | Conv2D | LSTM | **Conv1D + MaxPooling1D** |
|---|---|:---:|:---:|:---:|
| diabetes | r2 / RMSE | 0.4498 / 56.18 | 0.4294 / 57.21 | **0.4994 / 53.59** |
| 따릉이 | r2 / RMSE | 0.7108 / 46.16 | 0.6562 / 50.33 | **0.7335 / 44.30** |
| cancer | acc / loss | 0.965 / 0.1427 | 0.912 / 0.2408 | **0.965 / 0.0970** |
| wine | acc / loss | 0.963 / 0.1641 | 0.926 / 0.2550 | **0.981 / 0.0354** |
| digits | acc / loss | 0.971 / 0.2607 | 0.972 / 0.1275 | **0.979 / 0.0590** |

### 이미지 데이터 (GPU)

| 데이터 | Conv2D 기록 | **Conv1D** | 시간 |
|---|:---:|:---:|:---:|
| fashion | 0.9258 | 0.8996 | 37.11초 |
| cifar100 | 0.4371 (MaxPooling) / 0.4621 (GlobalAveragePooling2D) | 0.3756 | 156.63초 |

**읽는 법**
- 표 데이터 5개 모두 Conv1D + MaxPooling1D + Dropout이 Conv2D · LSTM 기록보다 좋거나 같았다
- 이미지는 Conv2D가 더 좋았다 → 가로 · 세로 두 방향의 작은 무늬를 보는 Conv2D가 이미지에 맞다

### RAG

| 파일 | 확인한 값 |
|---|---|
| rag12_Chroma04_load | Chroma12 문서 수 208 (고유 청크 52 × 4), 검색 4개 (default) / retriever 2개 |
| rag14_Chroma_pipeline2 | `dict_keys(['input', 'context', 'answer'])`, 창업자 질문 → "주어진 정보로는 답변할 수 없습니다." |
| rag17_FAISS_1_save | 청크 samsung 9 / nvidia 9, `faiss_index.d` 1536, 빈 FAISS `ntotal` 0 |
| rag17_FAISS_2_load | `index_to_docstore_id` 18개, `similarity_search(k=2)` → samsung_outlook 청크 2개 |
| rag18 거르기 | DB 안 질문 점수 0.139 ~ 0.493 / DB 밖 질문 -0.125 이하 → 기준 0.05 |
| rag18 답변 | 엔비디아 데이터센터 → 답 + 출처 / (이어서) 그 회사 위험 요인 → 엔비디아 위험 요인 / 김치찌개 · CEO · 본사 · 주가 → 답변할 수 없음 |

---

## 💻 핵심 개념

### Conv1D 입력 만들기

```
표 데이터  : (N, 컬럼 수)          → reshape(-1, 컬럼 수, 1)   컬럼 = 시점, feature 1
digits     : (N, 64)               → reshape(-1, 8, 8)         행 8 = 시점, feature 8
fashion    : (N, 28, 28)           → reshape(-1, 28, 28)       행 28 = 시점, 픽셀 28
cifar100   : (N, 32, 32, 3)        → reshape(-1, 32, 96)       행 32 = 시점, 픽셀 32 × RGB 3
```

### MaxPooling1D

```
출력 길이 = 입력 길이 // pool_size      param 0
(None, 10, 64) → MaxPooling1D(2) → (None, 5, 64)
(None, 13, 64) → (None, 6, 64) → Conv1D → MaxPooling1D(2) → (None, 3, 32)
```

### 출력층과 loss

```
회귀          : Dense(1)                       loss='mse'
이진 분류     : Dense(1, activation='sigmoid')   loss='binary_crossentropy'
다중 분류     : Dense(클래스 수, 'softmax')       loss='categorical_crossentropy' (원핫 y)
                                                 loss='sparse_categorical_crossentropy' (정수 y)
```

### RAG 체인 흐름

```
질문 {'input': ...}
   │
   ▼
retriever (as_retriever)  ──▶  관련 청크 k개 (context)
   │
   ▼
create_stuff_documents_chain(model, prompt)   청크를 {context} 에 채워 넣고 모델 호출
   │
   ▼
{'input', 'context', 'answer'}  ──▶  Gradio ChatInterface 에 answer 출력
```

### Chroma vs FAISS

```
                 Chroma                                    FAISS
저장           from_documents(persist_directory=...)      from_documents → save_local(folder_path, index_name)
불러오기       Chroma(persist_directory, collection_name)  FAISS.load_local(..., allow_dangerous_deserialization=True)
파일           chroma.sqlite3 등                          .faiss (벡터 인덱스) + .pkl (원문 · id)
같은 곳에 다시 저장   추가된다 (208 = 52 × 4)               save_local 은 파일을 새로 쓴다
문서 수        _collection.count()                         index.ntotal
```

---

## 성과

- 표 데이터 5개(diabetes · 따릉이 · cancer · wine · digits)를 Conv1D + MaxPooling1D + Dropout으로 학습해 모두 기존 Conv2D · LSTM 기록 이상
  - diabetes r2 0.4498 → **0.4994**, 따릉이 0.7108 → **0.7335**, wine acc 0.963 → **0.981**, digits 0.972 → **0.979**
- 이미지(fashion · cifar100)를 한 행 = 한 시점으로 바꿔 Conv1D 적용 (0.8996 / 0.3756), 이미지는 Conv2D가 더 맞다는 것 확인
- 회귀 · 이진 · 다중 분류에 맞는 출력층(`Dense(1)` / sigmoid / softmax)과 loss 정리
- 검색한 청크를 컨텍스트로 넣어 ChatOpenAI가 문서를 근거로 답하게 만들었다 (RAG)
- `ChatPromptTemplate` + `create_stuff_documents_chain` + `create_retrieval_chain`으로 RAG 체인 구성, 결과 `input / context / answer`
- Gradio `ChatInterface`로 RAG 챗봇 화면 실행
- FAISS로 벡터 DB를 만들고 `save_local` / `load_local`로 저장 · 불러오기, `index_to_docstore_id` / `docstore`로 구조 확인
- FAISS 챗봇에 관련 없는 질문 거르기 · system 규칙 · 이전 대화 · 출처 표시 · 예시 질문 화면 추가

---

## 💡 주요 학습 포인트

1. **표 데이터도 Conv1D에 넣을 수 있다**: `reshape(-1, 컬럼 수, 1)`로 컬럼을 시점처럼 본다
2. **MaxPooling1D는 시점 축을 pool_size만큼 줄인다**: 홀수면 내림 (9 → 4), 파라미터 0
3. **Conv1D 뒤에는 Flatten**: 출력이 3차원이라 Dense 전에 편다
4. **Dropout으로 과적합을 줄인다**: Conv1D 뒤 · Dense 사이에 넣었다 (diabetes 훈련 데이터는 약 216개뿐)
5. **출력층은 문제 종류에 맞춘다**: 회귀 `Dense(1)`, 이진 `sigmoid`, 다중 `softmax` + 클래스 수
6. **EarlyStopping patience가 epochs와 같으면 멈추지 않는다**: patience는 epochs보다 충분히 작게
7. **MCP 파일 이름은 파일마다 다르게**: 같은 이름이면 다른 모델을 덮어쓴다
8. **이미지를 Conv1D에 넣으려면 채널을 행에 합친다**: `(32, 32, 3)` → `(32, 96)`
9. **이미지는 Conv2D가 더 맞다**: Conv1D는 한 방향으로만 움직여 작은 2차원 무늬를 못 잡는다
10. **temperature 0은 같은 질문에 거의 같은 답**: RAG처럼 사실을 답할 때
11. **LLM은 문서 없이도 그럴듯하게 답한다**: 문서에 없는 내용(창업자)은 컨텍스트를 줘도 아는 지식으로 답할 수 있다
12. **프롬프트에 "없으면 답할 수 없다"를 넣는다**: 문서 밖 질문을 거절하게 만든다
13. **`create_stuff_documents_chain`은 문서를 `{context}`에 채운다**: prompt | model
14. **`create_retrieval_chain`은 `{'input': 질문}`으로 검색부터 한다**: 결과는 `input / context / answer`
15. **Gradio `ChatInterface(fn)`**: `fn(message, history)`의 return 값이 챗봇 답이 된다
16. **retriever는 관련 없어도 k개를 항상 넘긴다**: 거르려면 `similarity_score_threshold`
17. **같은 Chroma에 `from_documents`를 여러 번 하면 쌓인다**: 208 = 52 × 4, 검색 결과가 겹친다
18. **FAISS `IndexFlatL2`의 차원 = 임베딩 차원**: `embed_query`로 길이를 재서 1536
19. **FAISS는 `save_local`을 해야 파일로 남는다**: `.faiss` + `.pkl`
20. **`load_local`은 `allow_dangerous_deserialization=True`가 필요하다**: pkl은 내가 만든 파일만 연다
21. **`index_to_docstore_id` → `docstore`**: 벡터 번호 → 문서 id → Document(원문 · metadata)
22. **기준 점수는 직접 찍어 보고 정한다**: `similarity_search_with_relevance_scores`로 DB 안 / 밖 질문 비교
23. **규칙은 system 메시지로 분리하면 더 잘 지킨다**: `ChatPromptTemplate.from_messages`
24. **후속 질문은 앞 질문을 붙여 검색한다**: "그 회사"만으로는 검색이 안 된다
25. **`metadata['source']`로 출처를 보여준다**: 답이 어느 문서에서 나왔는지 확인할 수 있다

---

[⬅️ Day24](Day24.md) · [🏠 전체 목차](../README.md)
