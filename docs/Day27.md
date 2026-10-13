# Day27 - 단일 입력 다중 출력 모델, 주가 데이터 LSTM · Conv1D 앙상블 (삼성전자 시가 예측 · 가중치 제출), 대화 기억 RAG 챗봇

**학습 기간:** 2026-10-12

> Day26 앙상블에서 x2 · x3를 빼고 **입력 1개 → 출력 2개** 모델을 만들어, 병합 없이 한 모델의 출력을 두 분기로 나누는 법을 확인했다. 이어서 실제 주가 CSV 3개(삼성전자 · 아모레 · 하이닉스)로 **LSTM과 Conv1D를 섞은 입력 3개 앙상블 모델**을 만들어 **다음 거래일 삼성전자 시가**를 예측했다. 데이터 값과 y는 건드리지 않고 시작 일자만 맞췄으며, 훈련 코드에서 `save_weights`로 저장한 가중치를 예측 코드에서 `load_weights`로 불러와 **predict만 해도 같은 값**이 나오도록 제출용 파일을 나눴다. 같은 문제를 배운 스타일(제출용)과 개선된 방법(개선용) 두 가지로 작성했다. LangChain 쪽에서는 Gradio 대화 기록을 **`HumanMessage` · `AIMessage` 리스트로 바꿔 `MessagesPlaceholder`에 넣어** 이전 대화를 기억하는 챗봇을 만들었다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 단일 입력 다중 출력 | `Model(inputs=input1, outputs=[last_output1, last_output2])` → `fit(x1, [y1, y2])` |
| 분기 입력 | 병합층 없이 단일 모델의 출력 `output1`을 분기 두 곳에 그대로 넣는다 |
| CSV 숫자 읽기 | `pd.read_csv(..., encoding='cp949', thousands=',')` → `"260,000"`을 숫자 260000으로 |
| 시작 일자 맞추기 | 가장 늦게 시작하는 삼성전자(2022/03/14) 기준 1120 거래일로 아모레 · 하이닉스를 자른다 |
| 순서 뒤집기 | CSV가 최신 → 과거 순서라 `[::-1]`로 과거 → 최신 (값은 그대로) |
| 스케일링 | x만 종목별 StandardScaler, y(삼성전자 시가)는 원 단위 그대로 |
| split_x | 20 거래일씩 잘라 `(1100, 20, 6)`, y = 다음 날 시가, 마지막 20일은 예측용 |
| 앙상블 | 아모레 LSTM / 하이닉스 Conv1D / 삼성전자 Conv1D → LSTM → `Concatenate` → 시가 1개 |
| 가중치 제출 | 훈련 코드 `save_weights` → 예측 코드는 같은 구조 + `load_weights` + `predict`만 |
| 결과 | test r2 0.9908, 2026/10/13 삼성전자 시가 예측 **259,291.11원** |
| 대화 기억 | Gradio history → `HumanMessage` / `AIMessage` 리스트 → `MessagesPlaceholder('chat_history')` |
| 검색용 질문 바꾸기 | `create_history_aware_retriever`로 "그 층은?"을 혼자 뜻이 통하는 질문으로 바꿔 검색 |

---

## 📖 핵심 학습 내용

### 1. 입력 1개 · 출력 2개 모델 (keras69_ensemble4)

Day26 ensemble3에서 x2 · x3 데이터셋과 모델2 · 3, 병합층을 빼고 **x1 하나만** 남겼다.

```python
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, y1, y2, train_size=0.7, random_state=77)

#2-1. 모델1 (단일 모델)
input1 = Input(shape=(2,))
...
output1 = Dense(5, activation='relu', name='han4')(dense3)

#2-2. 분기1 (모델1 의 출력 output1 을 입력으로 가져온다)
last_dense1 = Dense(10, name='ld1')(output1)
last_dense2 = Dense(10, name='ld2')(last_dense1)
last_output1 = Dense(1, name='last1')(last_dense2)

#2-3. 분기2 (같은 output1 을 입력으로 가져온다)
last_output2 = Dense(1, name='last2')(output1)

model = Model(inputs=input1, outputs=[last_output1, last_output2])
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=8)
y1_pred, y2_pred = model.predict(x1_pred)
```

- 입력이 하나라 **`Concatenate`가 필요 없다** → 분기하는 곳에 단일 모델의 출력(`output1`)을 그대로 넣는다
- 입력은 리스트 없이 `inputs=input1`, `fit(x1_train, ...)`, `predict(x1_pred)` / 출력은 여전히 `[y1, y2]` 리스트
- Total params **1,222** (han1 ~ han4 1,035 + ld1 60 + ld2 110 + last1 11 + last2 6)
- 예측 : y1 3614.19 ~ 3667.33 (정답 3101 ~ 3106) / y2 15145.82 ~ 15367.88 (정답 13101 ~ 13106) → 입력 3개 앙상블(ensemble3)보다 정답과 차이가 컸다

### 2. 주가 데이터 준비 (keras70_samsung1_ensemble)

`_data/samsung/`의 CSV 3개(`삼성전자261012.csv` · `아모레261012.csv` · `하이닉스261012.csv`)로 **다음 거래일 삼성전자 시가 1개**를 예측했다.

| 파일 | 행 수 | 시작 일자 | 마지막 일자 |
|---|:---:|:---:|:---:|
| 삼성전자 | 1120 | 2022/03/14 | 2026/10/12 |
| 아모레 | 2060 | 2018/05/16 | 2026/10/12 |
| 하이닉스 | 1300 | 2021/06/17 | 2026/10/12 |

```python
samsung = pd.read_csv(path + '삼성전자261012.csv', encoding='cp949', thousands=',')
amore = pd.read_csv(path + '아모레261012.csv', encoding='cp949', thousands=',')
hynix = pd.read_csv(path + '하이닉스261012.csv', encoding='cp949', thousands=',')

# 시작 일자 맞추기 : 삼성전자 기준 1120일 (2022/03/14 ~ 2026/10/12)
amore = amore[:1120]
hynix = hynix[:1120]

# 최신 → 과거 순서라 과거 → 최신으로 뒤집는다
samsung = samsung[::-1]
amore = amore[::-1]
hynix = hynix[::-1]

cols = ['시가', '고가', '저가', '종가', '거래량', '금액(백만)']
x1_data = amore[cols].values        # (1120, 6)
x2_data = hynix[cols].values        # (1120, 6)
x3_data = samsung[cols].values      # (1120, 6)
y_data = samsung['시가'].values     # (1120,)
```

- `encoding='cp949'` : 한글 CSV, `thousands=','` : `"260,000"` 같은 문자열을 숫자로 읽는다 (파일 자체는 바꾸지 않는다)
- 세 파일의 2022/03/14 이후 거래일 1120일은 **날짜가 모두 같고 결측값이 없다** → 앞부분만 잘라 시작 일자를 맞춘다
- 파일이 최신 → 과거 순서다. 뒤집지 않고 `split_x`를 하면 "다음 날"이 아니라 "전날" 시가를 맞추게 된다
- 컬럼은 종목마다 6개(조건 5개 이상), 대비 · 등락률 · 신용비 · 개인 · 기관 · 외인 등은 버린다
- **데이터 값과 y는 수정하지 않는다** : 값 채우기 · 삭제 · 로그 변환 없이 원본 그대로

```python
scaler1 = StandardScaler()          # 종목마다 스케일러를 따로
x1_data = scaler1.fit_transform(x1_data)
...

size = 20
x1 = split_x(x1_data[:-1], size)    # (1100, 20, 6)  마지막 날은 정답(다음 날 시가) 이 없으니 빼고 자른다
y = y_data[size:]                   # (1100,)        i 번째 묶음(i ~ i+19 일) 의 정답 = i+20 일 시가

x1_pred = x1_data[-size:].reshape(1, size, 6)   # 예측용 : 마지막 20일 (2026/09/14 ~ 2026/10/12)
```

- 스케일링은 **x만**, 종목마다 가격대(아모레 13만 원대, 하이닉스 160만 원대)와 컬럼 단위(가격 vs 거래량)가 달라서 맞춘다
- **y는 스케일링하지 않는다** → 예측값이 바로 원 단위, `inverse_transform`도 필요 없다
- 1119행을 20일씩 자르면 1100묶음 → y도 1100개로 개수가 맞는다
- `train_test_split(x1, x2, x3, y, train_size=0.8, random_state=333)` → train 880 / test 220

### 3. LSTM + Conv1D 앙상블 모델 (keras70_samsung1_ensemble)

```python
#2-1. 모델1 아모레 (LSTM)
input1 = Input(shape=(size, 6))
dense1 = LSTM(64, name='am1')(input1)
dense1 = Dense(32, activation='relu', name='am2')(dense1)
output1 = Dense(16, activation='relu', name='am3')(dense1)

#2-2. 모델2 하이닉스 (Conv1D)
input2 = Input(shape=(size, 6))
dense2 = Conv1D(64, 3, padding='same', activation='relu', name='hy1')(input2)
dense2 = Conv1D(32, 3, padding='same', activation='relu', name='hy2')(dense2)
dense2 = Flatten(name='hy3')(dense2)
output2 = Dense(16, activation='relu', name='hy4')(dense2)

#2-3. 모델3 삼성전자 (Conv1D + LSTM)
input3 = Input(shape=(size, 6))
dense3 = Conv1D(64, 3, padding='same', activation='relu', name='ss1')(input3)
dense3 = LSTM(64, name='ss2')(dense3)
...

#2-4. 모델 병합
merge1 = Concatenate(name='mg1')([output1, output2, output3])    # (None, 48)
merge2 = Dense(64, activation='relu', name='mg2')(merge1)
merge3 = Dense(32, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)                       # 다음 거래일 삼성전자 시가 1개

model = Model(inputs=[input1, input2, input3], outputs=last_output)

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
model.fit([x1_train, x2_train, x3_train], y_train, epochs=3000, batch_size=32,
          validation_split=0.2, callbacks=[es])

model.save_weights(save_path + 'keras70_samsung.weights.h5')
```

- 입력마다 다른 층을 쓸 수 있다 : LSTM(아모레), Conv1D 2층 + Flatten(하이닉스), Conv1D → LSTM(삼성전자)
- `Conv1D` 출력이 3차원 `(None, 20, 64)`라 그대로 `LSTM`에 넣을 수 있다
- 삼성전자 과거 데이터도 입력으로 넣어 앙상블을 키웠다 (아모레 · 하이닉스만으로는 삼성전자 가격대를 알기 어렵다)
- Total params **80,529**
- y가 10만 ~ 27만 원대로 커도 기본 `adam`으로 419 epoch에서 EarlyStopping
- `restore_best_weights=True`로 되돌린 최적 가중치를 `save_weights`로 저장 → `_save/keras70/keras70_samsung.weights.h5`

### 4. 가중치만 불러와 예측 (keras70_samsung2_load_weights)

```python
# 데이터 처리는 훈련 코드와 동일 (같은 CSV, 같은 순서로 스케일러 fit)
x1_pred = x1_data[-size:].reshape(1, size, 6)
...

# 모델 구조도 훈련 코드와 동일
input1 = Input(shape=(size, 6))
...
model = Model(inputs=[input1, input2, input3], outputs=last_output)
model.load_weights(save_path + 'keras70_samsung.weights.h5')

y_pred = model.predict([x1_pred, x2_pred, x3_pred])     # [[259291.11]]
```

- **가중치를 갱신하는 곳은 훈련 코드 한 곳뿐** : 예측 코드에는 `compile` · `fit`이 없다
- 예측 코드에도 모델 구조가 필요한 이유 : `.weights.h5`에는 **가중치 숫자만** 있고 층 구성은 없다 → 같은 구조의 빈 틀을 만들고 `load_weights`로 채운다 (구조가 다르면 에러, Day11 keras29_6과 같은 원리)
- 스케일러는 저장하지 않았다 → 같은 데이터로 다시 fit하면 평균 · 표준편차가 같아 같은 입력이 만들어진다
- 제출 파일 : 원 코드(앙상블) `.py` + 예측 코드 `.py` + 가중치 `.weights.h5` → 예측 코드를 두 번 돌려도 **259,291.11원** 동일
- 모델 전체를 `model.save()` / `load_model()`로 저장하면 예측 코드에 구조가 필요 없지만, 그건 모델 파일 제출이다
- 같은 가중치라도 GPU와 CPU는 LSTM 계산 방식이 달라 소수점 아래 ~ 수십 원 차이가 날 수 있다 (약 20원 차이 확인)

### 5. 개선용 코드 (keras70_samsung3_ensemble_improved · keras70_samsung4_load_weights_improved)

제출용과 같은 데이터 · 모델 구조를 다른 방법으로 작성하고 주석을 자세히 달았다.

| 제출용 (배운 스타일) | 개선용 | 이유 |
|---|---|---|
| `amore[:1120]` | `amore[amore['일자'] >= start_date]` | 행 개수를 직접 세지 않고 날짜로 자른다 |
| `samsung[::-1]` | `sort_values('일자').reset_index(drop=True)` | 파일 순서와 상관없이 날짜로 정렬, 행 번호 0부터 다시 |
| - | `assert (samsung['일자'] == amore['일자']).all()` | 세 파일의 같은 행이 같은 날짜인지 코드가 검사 |
| `split_x(x_data[:-1])` + `x_data[-20:]` | `split_x(x_data)` 후 `x[-1:]` / `x[:-1]` | 마지막 묶음을 예측용으로 떼어낸다 (같은 결과) |
| 모델을 풀어서 작성 | `def build_model():` | 훈련 · 예측 코드가 같은 구조를 쓰도록 한 곳에 묶는다 |
| `optimizer='adam'` | `Adam(learning_rate=0.005)` | 큰 y를 빨리 따라가게 학습률을 키운다 |
| - | `random.seed` · `np.random.seed` · `tf.random.set_seed` | 다시 훈련해도 결과가 비슷하게 |

- `reset_index`가 필요한 이유 : 뒤집거나 자르면 행 번호가 1119, 1118 ...로 남는데, pandas Series 비교(`==`)는 **행 번호끼리** 짝을 짓는다 → `.values`로 꺼내 쓰면 필요 없다
- 개선용은 제출용 가중치를 덮어쓰지 않도록 `keras70_samsung_improved.weights.h5`로 저장한다

### 6. 이전 대화를 기억하는 RAG 챗봇 (rag21_Gradio_history)

Day26 rag20_embedding4_ChatBot을 바탕으로, 대화 기록을 **문자열이 아니라 메시지 리스트**로 넘겼다.

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ('system', '''너는 Attention Is All You Need.pdf 만 보고 답하는 챗봇이다. ...
컨텍스트 :
{context}'''),
    MessagesPlaceholder(variable_name='chat_history'),    # 이전 대화 메시지 리스트가 들어갈 자리
    ('human', '{input}'),
])
prompt = prompt.partial(no_answer=NO_ANSWER, history='(컨텍스트 아래에 이어지는 대화 메시지 참고)')

docu_chain = create_stuff_documents_chain(model, prompt)
rephrase_prompt = ChatPromptTemplate.from_messages([
    ('system', '이전 대화를 참고해서, 마지막 질문을 이전 대화 없이도 이해할 수 있는 하나의 질문으로 다시 써라. ...'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}'),
])
history_retriever = create_history_aware_retriever(model, retriever, rephrase_prompt)
rag_chain = create_retrieval_chain(history_retriever, docu_chain)

def answer_invoke(message, history):
    chat_history = []
    for item in history[-10:]:                  # 최근 10개 메시지(5번 주고받은 대화) 만
        content = item['content']
        if item['role'] == 'user':
            chat_history.append(HumanMessage(content=content))
        elif item['role'] == 'assistant':
            chat_history.append(AIMessage(content=content))

    response = rag_chain.invoke({'input' : message, 'chat_history' : chat_history})
    return response['answer']
```

- **`MessagesPlaceholder`** : 프롬프트 안에 메시지 리스트가 통째로 들어갈 자리, `variable_name`과 invoke의 키(`chat_history`)가 같아야 한다
- Gradio `history`는 `{'role': 'user' | 'assistant', 'content': ...}` 딕셔너리 리스트 → `HumanMessage` / `AIMessage`로 바꿔 넣는다
- `answer_invoke`가 불릴 때 `history`에는 **지금 질문 이전 대화만** 있고, 지금 질문은 `message`로 따로 들어온다
- `history[-10:]` : 최근 10개만 넘겨서 대화가 길어져도 프롬프트 토큰이 계속 늘지 않게 한다 (`history[-10]`은 메시지 1개)
- `prompt.partial(...)` : invoke에서 넘기지 않는 프롬프트 변수(`{no_answer}`, `{history}`)를 미리 채워 둔다
- **`create_history_aware_retriever`** : "그 층은?"을 그대로 검색하면 "그 층"이 무엇인지 몰라 문서를 못 찾는다 → 이전 대화를 보고 질문을 바꾼 뒤 검색 (첫 질문은 그대로 검색)
- `create_retrieval_chain` 결과는 딕셔너리 → `response['answer']`가 답변, `response['context']`가 검색된 문서
- rag20의 검색 결과 없음 처리 · 거절 문장 처리 · 출처 표시는 빠지고 답변만 반환한다 → 문서 밖 질문은 시스템 규칙 3번으로 "주어진 정보로는 답변할 수 없습니다."

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras2/keras69_ensemble4.py` | 입력 1개 · 출력 2개, 병합 없이 `output1`을 분기 두 곳에 입력, params 1,222 |
| `keras2/keras70_samsung1_ensemble.py` | (제출용 원 코드) 삼성전자 · 아모레 · 하이닉스 CSV → LSTM + Conv1D 입력 3개 앙상블 → 다음 거래일 삼성전자 시가, `save_weights` |
| `keras2/keras70_samsung2_load_weights.py` | (제출용 예측 코드) 같은 구조 + `load_weights` + `predict`만 → 259,291.11원 |
| `keras2/keras70_samsung3_ensemble_improved.py` | (개선용 원 코드) `sort_values` · 날짜 기준 자르기 · `assert` · `build_model()` · 시드 · `Adam(learning_rate=0.005)`, 상세 주석 |
| `keras2/keras70_samsung4_load_weights_improved.py` | (개선용 예측 코드) `build_model()` + `load_weights`, 같은 예측값이 나오는 조건 주석 |
| `RAG/rag21_Gradio_history.py` | `MessagesPlaceholder` + `HumanMessage` / `AIMessage`로 이전 대화를 기억하는 PDF 챗봇, `create_history_aware_retriever` |
| `_data/samsung/` | 삼성전자 · 아모레 · 하이닉스 일별 주가 CSV 3개 (2026/10/12 기준) |
| `docs/Day27.md` | Day27 학습 기록 신규 작성 |
| `docs/Day26.md` | 하단 nav에 Day27 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day27, `_data/samsung`, `_save/keras70`) / 진행도(27일, 34%) 갱신 (수정) |

---

## 📊 실행 결과

### 단일 입력 다중 출력 (keras69_ensemble4)

| 파일 | 구조 | Total params | 예측 y1 (정답 3101 ~ 3106) | 예측 y2 (정답 13101 ~ 13106) |
|---|---|:---:|---|---|
| ensemble3 (Day26) | 입력 3 → 출력 2 | 9,815 | 3093.98 ~ 3098.88 | 13070.58 ~ 13075.60 |
| ensemble4 | 입력 1 → 출력 2 | 1,222 | 3614.19 ~ 3667.33 | 15145.82 ~ 15367.88 |

### 삼성전자 시가 예측 (keras70_samsung1 · 2)

| 항목 | 값 |
|---|---|
| 데이터 | 1120 거래일 (2022/03/14 ~ 2026/10/12) → 1100묶음 (20일, 6컬럼) |
| train / test | 880 / 220 (validation_split 0.2) |
| 훈련 | 419 epoch에서 EarlyStopping (patience 100) |
| test loss (mse) | 37,825,364 (RMSE 약 6,150원) |
| test r2 | **0.9908** |
| 2026/10/13 시가 예측 | **259,291.11원** (훈련 코드 · 예측 코드 동일) |

### 대화 기억 챗봇 (rag21_Gradio_history)

| 질문 | 답변 |
|---|---|
| Transformer 인코더는 몇 개의 층으로 되어 있어? | 인코더는 N = 6 identical layers로 구성되어 있다. |
| **그 층** 하나는 어떤 서브 레이어로 구성돼? | 두 개의 서브 레이어 : 다중 헤드 자기 attention, 위치 기반 피드포워드 네트워크 → "그 층"을 인코더 층으로 알아들었다 |
| 삼성전자의 사업 전망은? | 주어진 정보로는 답변할 수 없습니다. |

---

## 💻 핵심 개념

### 단일 입력 다중 출력 구조

```
x1 (N, 2) ─▶ Input(2,) ─▶ Dense ×4 ─▶ output1 (None, 5) ─┬─▶ ld1 ─▶ ld2 ─▶ last1 (y1)
                                                         └─▶ last2 (y2)
```

### 삼성전자 시가 앙상블 구조

```
아모레   (N, 20, 6) ─▶ LSTM(64) ─▶ Dense 32 ─▶ Dense 16 ──────────────┐
하이닉스 (N, 20, 6) ─▶ Conv1D 64 ─▶ Conv1D 32 ─▶ Flatten ─▶ Dense 16 ──┼─▶ Concatenate (None, 48) ─▶ Dense 64 ─▶ Dense 32 ─▶ 시가 1개
삼성전자 (N, 20, 6) ─▶ Conv1D 64 ─▶ LSTM(64) ─▶ Dense 32 ─▶ Dense 16 ──┘
```

### 시계열 x · y 짝 맞추기

```
일자        0  1  2 ... 19 | 20 | ... 1118 | 1119 (2026/10/12)
묶음 0      [0 ~ 19]       → y = 20일 시가
묶음 1099   [1099 ~ 1118]  → y = 1119일 시가
예측용      [1100 ~ 1119]  → 2026/10/13 시가 (정답 없음)
```

### 가중치 저장 · 불러오기 제출 흐름

```
keras70_samsung1_ensemble.py         모델 구성 ─▶ compile ─▶ fit ─▶ save_weights ─▶ keras70_samsung.weights.h5
keras70_samsung2_load_weights.py     같은 모델 구성 ─▶ load_weights(keras70_samsung.weights.h5) ─▶ predict
                                     (compile · fit 없음 → 가중치 그대로 → 항상 같은 예측값)
```

### 이전 대화 기억 흐름

```
Gradio history [{'role': 'user', ...}, {'role': 'assistant', ...}, ...]
   └─ 최근 10개 → [HumanMessage, AIMessage, ...] = chat_history
        ├─ rephrase_prompt + model ─▶ 바꾼 질문 ─▶ retriever ─▶ context
        └─ prompt : system(context) + MessagesPlaceholder(chat_history) + human(input) ─▶ model ─▶ answer
```

---

## 성과

- x2 · x3를 뺀 입력 1개 모델의 출력을 분기 두 곳에 넣어 다중 출력 모델을 만들었다
- 실제 주가 CSV 3개의 시작 일자를 맞추고, 데이터와 y를 건드리지 않은 채 다음 거래일 삼성전자 시가를 예측했다
- LSTM과 Conv1D를 입력마다 다르게 쓰는 입력 3개 앙상블 모델을 만들어 test r2 0.9908
- 훈련 코드(`save_weights`)와 예측 코드(`load_weights` + `predict`)를 나눠 가중치 파일만으로 같은 예측값이 나오게 했다
- 같은 문제를 제출용(배운 스타일)과 개선용(`sort_values` · `assert` · `build_model()` · 시드) 두 가지로 작성
- Gradio 대화 기록을 `HumanMessage` / `AIMessage` 리스트로 `MessagesPlaceholder`에 넣어 이전 대화를 기억하는 RAG 챗봇을 만들었다
- `create_history_aware_retriever`로 "그 층" 같은 후속 질문도 검색되게 했다

---

## 💡 주요 학습 포인트

1. **입력이 하나면 병합층이 필요 없다**: 단일 모델의 출력을 분기마다 입력으로 넣는다
2. **입력 1개 · 출력 2개**: `inputs=input1`, `outputs=[...]` → `fit(x1, [y1, y2])`
3. **`thousands=','`로 쉼표 숫자를 읽는다**: CSV 파일은 그대로, 읽는 방식만 바꾼다
4. **시작 일자는 가장 늦게 시작하는 파일에 맞춘다**: 세 파일의 날짜가 모두 같은지 먼저 확인
5. **최신 → 과거 순서 파일은 뒤집는다**: 그대로 자르면 "전날" 값을 맞추는 모델이 된다
6. **y는 건드리지 않는다**: 스케일링은 x만, 예측값이 바로 원 단위
7. **종목마다 스케일러를 따로 둔다**: 가격대와 컬럼 단위가 종목마다 다르다
8. **마지막 묶음은 정답이 없다**: x에서 빼고 예측 입력으로 쓴다 → x와 y 개수가 맞는다
9. **입력마다 다른 층을 쓸 수 있다**: LSTM / Conv1D / Conv1D → LSTM을 `Concatenate`로 합친다
10. **Conv1D 출력은 3차원이라 LSTM에 바로 넣을 수 있다**: `(None, 20, 64)`
11. **`load_weights`에는 같은 구조가 먼저 필요하다**: 가중치 파일에는 숫자만 있다
12. **가중치 갱신은 훈련 코드 한 곳**: 예측 코드는 `compile` · `fit` 없이 `predict`만
13. **스케일러는 같은 데이터로 다시 fit해도 같다**: 저장하지 않아도 같은 입력이 만들어진다
14. **`model.save`는 구조 + 가중치, `save_weights`는 가중치만**: 제출 형태에 맞게 고른다
15. **GPU와 CPU는 LSTM 결과가 조금 다를 수 있다**: 같은 가중치로 약 20원 차이
16. **`reset_index`는 pandas 비교 때문에 필요하다**: `.values`로 꺼내 쓰면 행 번호는 상관없다
17. **`MessagesPlaceholder`는 메시지 리스트가 들어갈 자리**: `variable_name`과 invoke 키를 맞춘다
18. **Gradio history는 `HumanMessage` / `AIMessage`로 바꿔 넣는다**: 지금 질문은 `message`로 따로 온다
19. **`history[-10:]`로 최근 대화만 기억**: 대화가 길어져도 토큰이 계속 늘지 않는다
20. **`partial`로 invoke에서 넘기지 않는 프롬프트 변수를 미리 채운다**
21. **대화 기억은 검색에도 필요하다**: `create_history_aware_retriever`로 질문을 바꿔 검색해야 "그 층"도 찾는다

---

[⬅️ Day26](Day26.md) · [🏠 전체 목차](../README.md) · [Day28 ➡️](Day28.md)
