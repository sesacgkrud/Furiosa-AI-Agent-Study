# Day28 - 가중치 확인 · 동결 (trainable), VGG16 전이학습 (cifar10 · cifar100), RAG 검색 문서 확인 · score_threshold · 프롬프트 규칙

**학습 기간:** 2026-10-13

> `model.weights` / `model.trainable_weights`로 모델 안의 가중치를 직접 꺼내 보고, **`model.trainable = False`로 가중치를 동결**하면 fit을 해도 가중치가 바뀌지 않는다는 것을 확인했다. 훈련이 끝난 가중치만으로 예측값을 **손으로 계산**해 Dense 연산(`y = x · W + b`)을 따라가 봤다. 이어서 imagenet으로 미리 훈련된 **VGG16**을 `keras.applications`에서 불러와 `include_top`에 따라 구조가 어떻게 바뀌는지 보고, **cifar10 · cifar100에 전이학습**을 적용해 이전 최고 기록과 **동결 O / 동결 X**의 정확도 · 소요 시간을 비교했다. LangChain 쪽에서는 `response['context']`로 **실제 검색된 문서를 터미널에 출력**하고, `similarity_score_threshold` 0.7로 검색 결과가 달라지는 것을 본 뒤, **프롬프트 규칙**을 바꿔 인사 · 이전 대화 질문에도 답하는 챗봇을 만들었다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 가중치 꺼내기 | `model.weights` → 층마다 kernel(W) · bias(b), `len()` = 층 수 x 2 |
| 동결 | `model.trainable = False` → `trainable_weights`가 `[]`, summary의 Trainable params 0 |
| 동결 후 훈련 | compile · fit은 돌아가지만 가중치가 그대로 → 처음 랜덤 가중치로 예측 |
| 가중치로 직접 계산 | `y = ((x·W1 + b1)·W2 + b2)·W3 + b3` → activation이 없으면 1차 함수 하나 |
| VGG16 | Conv2D 13개 + Dense 3개, imagenet(1000종)으로 미리 훈련된 모델, Total params 138,357,544 |
| include_top | `False` → Flatten · fc1 · fc2 · predictions가 빠지고 `input_shape`를 바꿀 수 있다 |
| 전이학습 | `model.add(vgg16)` + Flatten + Dense → cifar10 · cifar100 분류 |
| 동결 O / X | `vgg16.trainable = False` (Dense만 훈련) / `True` (VGG16까지 전부 훈련) |
| cifar10 결과 | 동결 X **0.8158** > 이전 최고 0.7727 > 동결 O 0.6266 |
| cifar100 결과 | 이전 최고 0.4621 > 동결 O 0.381 > 동결 X 0.01 (학습 안 됨) |
| 검색 문서 확인 | `response['context']`를 `enumerate`로 출력 → 어떤 문서가 컨텍스트로 들어갔는지 확인 |
| score_threshold 0.7 | Qwen3 논문 질문 유사도 0.42 ~ 0.59 → 모든 질문의 검색 결과가 0개 |
| 프롬프트 규칙 | 인사 · 이전 대화 질문은 자연스럽게, 논문 질문은 컨텍스트로만, 없으면 "주어진 정보로는 답변할 수 없습니다." |

---

## 📖 핵심 학습 내용

### 1. model.weights · trainable_weights와 동결 (keras71_trainable_weights1)

```python
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

print(model.weights)                 # [dense/kernel (1, 3), dense/bias (3,), dense_1/kernel (3, 2), ...]
print(len(model.weights))            # 6
print(len(model.trainable_weights))  # 6

model.trainable = False              # 동결
print(len(model.weights))            # 6
print(len(model.trainable_weights))  # 0
print(model.trainable_weights)       # []
```

- `model.weights` : 층마다 **kernel(가중치 W)** 과 **bias(b)** 두 개씩 → 층 3개라 6개
- 처음 kernel은 랜덤값(`dense/kernel = [[ 0.13603318, -0.03480017, 0.7743634 ]]`), bias는 모두 0
- 파라미터 = 입력 수 x 출력 수 + bias : `1x3+3 = 6`, `3x2+2 = 8`, `2x1+1 = 3` → Total 17
- `model.trainable = False` : 가중치 6개는 **그대로 있지만** `trainable_weights`(훈련할 가중치 목록)가 비어 버린다
- summary : Total params 17 그대로, **Trainable params 17 → 0 / Non-trainable params 0 → 17**

### 2. 동결한 모델을 훈련하면 (keras71_trainable_weights2)

```python
model = Sequential()
model.add(Dense(30, input_dim=1))
... # Dense 20 → 20 → 20 → 10 → 10 → 1

model.trainable = False
# model.trainable = True

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=100, verbose=1)
print(model.predict(x))
```

- 동결해도 compile · fit은 에러 없이 돌아간다 → loss만 계산되고 **가중치는 갱신되지 않는다**
- 동결 X : `[[1.0008] [2.0005] [3.0002] [3.9999] [4.9996]]` → y = x에 맞게 훈련됨
- 동결 O : `[[-0.3131] [-0.6262] [-0.9393] [-1.2524] [-1.5655]]` → 처음 랜덤 가중치 그대로 예측
- bias가 처음에 0이고 activation이 없어서 동결 O 예측값은 정확히 `-0.3130919 x` 꼴 (x = 2 → 2배)

### 3. 가중치만으로 예측값 직접 계산 (keras71_trainable_weights3)

데이터를 `x = [1]`, `y = [1]` 하나로 줄이고 Dense 3 → 2 → 1을 1000 epoch 훈련한 뒤, `model.weights`로 꺼낸 값으로 예측값을 손으로 계산했다.

```
입력 x = 1
1) Dense(3) : h1 = x · W1 + b1
   h1 = [1 x 0.08450541 - 0.05152767, 1 x (-0.08842915) - 0.05362988, 1 x 0.8283203 + 0.05395508]
      = [0.03297774, -0.14205903, 0.88227538]
2) Dense(2) : h2 = h1 · W2 + b2 = [-0.71161452, -0.78508734]
3) Dense(1) : y  = h2 · W3 + b3
      = (-0.71161452) x (-0.08098985) + (-0.78508734) x (-1.1340631) + 0.05222915
      = 0.05763356 + 0.89033859 + 0.05222915 = 1.0002013  ≒ 1
```

- Dense 연산은 `y = x · W + b` (activation 없음 = linear)
- 한 줄로 쓰면 `y = ((x·W1 + b1)·W2 + b2)·W3 + b3`
- **activation이 없으면 층을 여러 개 쌓아도 결국 `y = a·x + c` 1차 함수 하나**
  - `a = W1·W2·W3 = 0.72779308`, `c = b1·W2·W3 + b2·W3 + b3 = 0.27240818` → `0.72779308 x 1 + 0.27240818 = 1.0002013`

### 4. VGG16 구조와 include_top (keras72_1_vgg16_include_top)

```python
from keras.applications import VGG16

model = VGG16()                       # = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
model = VGG16(
    # weights='imagenet',             # 주석 처리해도 기본값이 'imagenet'
    include_top=False,
    input_shape=(100, 100, 3)
)
```

- **VGG16** : 옥스퍼드 VGG(Visual Geometry Group) 팀의 CNN, 가중치가 있는 층 16개(Conv2D 13개 + Dense 3개)
  - 모든 Conv2D가 3x3 · `padding='same'` · relu → 크기는 그대로, 필터만 64 → 128 → 256 → 512
  - block1 ~ block5 끝마다 MaxPooling2D → 224 → 112 → 56 → 28 → 14 → 7
  - imagenet(사진 약 120만 장, 1000종)으로 훈련된 가중치를 `weights='imagenet'`으로 내려받는다 (`None`이면 랜덤 가중치)
- `VGG16()` 기본값 : `include_top=True`, `(224, 224, 3)`, Total params **138,357,544**
  - `block1_conv1` : (3x3x3 + 1) x 64 = 1,792
  - `flatten` : 7 x 7 x 512 = 25,088 → `fc1` : (25,088 + 1) x 4,096 = 102,764,544 (전체의 약 74%)
  - `predictions` : (4,096 + 1) x 1,000 = 4,097,000 (imagenet 1000종 softmax)
- `include_top=True`는 fc1 입력이 25,088로 정해져 있어 **224x224로만** 넣을 수 있다
- `include_top=False`
  1. 맨 위 Flatten · fc1 · fc2 · predictions가 빠지고 `block5_pool`에서 끝난다 → Total params **14,714,688**
  2. `input_shape`를 원하는 데이터 크기로 바꿀 수 있다 : `(100, 100, 3)` → 100 → 50 → 25 → 12 → 6 → 3, 출력 `(3, 3, 512)`
  - Conv2D 파라미터는 입력 크기와 상관없어서 `(224, 224, 3)`이든 `(100, 100, 3)`이든 14,714,688로 같다

### 5. VGG16 전이학습 cifar10 (keras73_vgg16_01_cifar10)

Day14 `keras40_GAP03_cifar10`(이전 cifar10 최고 0.7727)의 데이터 · 훈련 · 평가 코드에 VGG16을 붙였다.

```python
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)     # 32 → 16 → 8 → 4 → 2 → 1 → 출력 (1, 1, 512)
)

vgg16.trainable=False      # 2. 가중치 동결
# vgg16.trainable=True     # 3. 가중치 동결 X

model = Sequential()
model.add(vgg16)                                # vgg16 (Functional) 한 줄로 들어간다
model.add(Flatten())                            # (1, 1, 512) → (512,)
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
```

- **전이학습** : 큰 데이터로 이미 훈련된 모델의 가중치를 가져와 내 데이터에 다시 쓴다 → 특징을 뽑는 Conv2D 부분만 가져오고 분류하는 Dense는 cifar10에 맞게 직접 붙인다
- 이전 cifar10과 달라진 점 : Conv2D를 직접 쌓지 않고 `model.add(vgg16)`, 가중치가 랜덤이 아니라 **imagenet 가중치에서 시작**
- 스케일링 `-1 ~ 1` · 원핫 · `categorical_crossentropy` · EarlyStopping(`val_acc`, patience 25) · epochs 80 · batch 128 · validation_split 0.1은 이전과 같게 → 모델만 바꿨을 때를 공정하게 비교
- `vgg16.trainable`은 keras71의 `model.trainable`을 **VGG16 부분에만** 적용한 것 (아래 Dense는 항상 훈련)

| | 동결 O (`trainable=False`) | 동결 X (`trainable=True`) |
|---|---|---|
| 훈련되는 가중치 | Dense 3개만 **62,410** | VGG16 포함 전체 **14,777,098** |
| Non-trainable params | 14,714,688 (VGG16) | 0 |
| VGG16의 역할 | imagenet으로 배운 특징을 그대로 뽑는 도구 | imagenet 가중치에서 출발해 cifar10에 맞게 다시 고친다 |
| epoch당 시간 | 짧다 (약 4.1초) | 길다 (약 22초) |

- Total params는 동결해도 14,777,098 그대로 → 가중치는 있지만 훈련만 안 되는 것 (keras71과 같은 원리)

### 6. VGG16 전이학습 cifar100 (keras73_vgg16_02_cifar100)

cifar10 VGG16 코드에서 데이터를 cifar100으로, 마지막 층을 `Dense(100, activation='softmax')`로 바꿨다 (이전 최고 Day14 `keras40_GAP04_cifar100` 0.4621).

- 클래스가 100개라 한 종류당 훈련 이미지 500장 → cifar10보다 훨씬 어렵다
- 마지막 Dense (100 + 1) x 100 = 10,100 → Trainable params 동결 O **71,500** / 동결 X **14,786,188**
- 동결 X는 1 epoch부터 loss **4.605 = ln(100)**, acc **0.01 = 1/100** → 100개 클래스에 똑같은 확률을 주는 찍기 수준에서 학습이 되지 않았다
  - adam 기본 학습률 0.001로 VGG16 전체를 고치다가 imagenet 가중치가 크게 바뀌어 망가진 것으로 볼 수 있다

### 7. 실제 검색된 문서 확인 (rag22_Gradio_1_doc_check)

Day27 rag21_Gradio_history에서 retriever를 `k=2`로 줄이고, `response['context']`를 터미널에 출력했다.

```python
retriever = db.as_retriever(search_kwargs={'k': 2})
...
response = rag_chain.invoke({'input' : message, 'chat_history' : chat_history})

for i, doc in enumerate(response['context'], start=1):
    print(f'\n검색 문서 {i}')
    print('출처: ', doc.metadata.get('source'))
    print('내용: ', doc.page_content)
    print('========================================')

return response['answer']
```

- `create_retrieval_chain` 결과의 `response['context']` = 실제로 컨텍스트에 들어간 문서 리스트
- threshold 없이 `k=2`면 **"안녕"처럼 논문과 관계없는 질문도 가장 비슷한 문서 2개가 항상 들어온다**
- 시스템 규칙 3번("관련 내용이 전혀 없을 때만 정확히 `{no_answer}` 한 문장")에 따라 "안녕" → "주어진 정보로는 답변할 수 없습니다."

### 8. similarity_score_threshold 0.7 (rag22_Gradio_2_Similarity)

```python
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 2, 'score_threshold': 0.7},
)
...
if not response['context']:
    return NO_ANSWER
```

- `similarity_score_threshold` : 유사도 점수가 `score_threshold`보다 **낮은** 문서는 버린다
- Qwen3 임베딩 유사도 : 논문 질문 0.42 ~ 0.59, "안녕" 0.11, "내가 누구게" 0.04, "삼성전자의 사업 전망은?" -0.03
- 0.7은 논문 질문도 넘지 못해 **모든 질문의 검색 결과가 0개** → 터미널에 `No relevant docs were retrieved using the relevance score threshold 0.7`
- 컨텍스트가 비어 있어도 모델이 자기 지식으로 답하는 경우가 있어서, Day26 rag20의 `if not docs` 처리처럼 검색 문서가 0개면 `NO_ANSWER`를 돌려준다
- 결과 : rag22_1에서는 답하던 "Multi-Head Attention 은 무엇이야?"도 "주어진 정보로는 답변할 수 없습니다." → **threshold는 임베딩 모델의 점수 분포에 맞춰 정해야 한다**

### 9. 프롬프트 규칙 바꾸기 (rag22_Gradio_3_prompt)

```python
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 2, 'score_threshold': 0.2},     # Day26 rag20 에서 Qwen3 기준으로 정한 값
)

prompt = ChatPromptTemplate.from_messages([
    ('system', '''너는 Attention Is All You Need.pdf 를 보고 답하는 챗봇이다. 아래 규칙을 반드시 지켜라.
1. "안녕", "고마워" 같은 인사나 일상 대화는 자연스럽게 답해라.
2. 사용자가 자기 자신에 대해 묻거나("내가 누구게" 등) 이전 대화에서 알려 준 정보(이름, 직업 등) 를 물으면,
   컨텍스트와 상관없이 이전 대화를 보고 답해라.
3. 그 밖의 질문은 컨텍스트에 쓰여 있는 내용으로만 답해라. ...
4. 3번 질문인데 컨텍스트에 관련 내용이 없으면, 다른 말 없이 정확히 "{no_answer}" 한 문장만 출력해라.
...'''),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}'),
])
```

- rag22_1 · rag22_2 규칙은 "PDF만 보고 답하기"라서 "안녕", "내가 누구게"도 "주어진 정보로는 답변할 수 없습니다."였다
- 규칙 1 · 2를 추가해 **인사 · 이전 대화 질문은 검색 문서가 없어도 답하게** 했다
- threshold 0.2 : 논문 질문(0.42 ~ 0.59)만 문서가 들어오고, "안녕" · "삼성전자"는 0개
- 검색 문서 0개여도 인사에는 답해야 하므로 `if not response['context']: return NO_ANSWER`는 지우고 **거절은 프롬프트 규칙 4번**에 맡긴다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras2/keras71_trainable_weights1.py` | `model.weights` · `trainable_weights` 출력, `model.trainable = False` 전후 개수 · summary 비교 |
| `keras2/keras71_trainable_weights2.py` | Dense 7층 모델을 동결 O / 동결 X로 100 epoch 훈련해 예측값 비교 |
| `keras2/keras71_trainable_weights3.py` | x=1, y=1로 1000 epoch 훈련 후 `model.weights`만으로 예측값 1.0002013을 수식으로 계산 |
| `keras2/keras72_1_vgg16_include_top.py` | `VGG16()` 기본 구조 · 파라미터, `include_top=False` + `input_shape=(100, 100, 3)` 구조 비교 |
| `keras2/keras73_vgg16_01_cifar10.py` | VGG16 전이학습 cifar10, 동결 O 0.6266 / 동결 X 0.8158 (이전 최고 0.7727) |
| `keras2/keras73_vgg16_02_cifar100.py` | VGG16 전이학습 cifar100, 동결 O 0.381 / 동결 X 0.01 (이전 최고 0.4621) |
| `RAG/rag22_Gradio_1_doc_check.py` | `response['context']`로 실제 검색된 문서 2개를 터미널에 출력 |
| `RAG/rag22_Gradio_2_Similarity.py` | `similarity_score_threshold` 0.7 → 모든 질문 검색 결과 0개, 0개면 `NO_ANSWER` |
| `RAG/rag22_Gradio_3_prompt.py` | 인사 · 이전 대화 규칙 추가, threshold 0.2 |
| `RAG/rag21_Gradio_history.py` | 저장된 `./_db/Faiss20`을 불러오기만 (`from_documents` · `save_local` 주석), retriever `k=4` (수정) |
| `docs/Day28.md` | Day28 학습 기록 신규 작성 |
| `docs/Day27.md` | 하단 nav에 Day28 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day28) / 진행도(28일, 35%) 갱신 (수정) |

---

## 📊 실행 결과

### 동결 O / 동결 X 예측값 (keras71_trainable_weights2, y = x)

| x | 동결 X | 동결 O |
|:---:|:---:|:---:|
| 1 | 1.0008212 | -0.3130919 |
| 2 | 2.0005221 | -0.6261838 |
| 3 | 3.0002232 | -0.9392756 |
| 4 | 3.9999237 | -1.2523676 |
| 5 | 4.9996247 | -1.565459 |

### VGG16 cifar10 (keras73_vgg16_01_cifar10)

| 모델 | Trainable params | EarlyStopping | test acc | 소요 시간 |
|---|:---:|:---:|:---:|:---:|
| 이전 최고 (keras40_GAP03, GlobalAveragePooling2D) | - | - | 0.7727 | 199.62초 |
| VGG16 동결 O | 62,410 | Epoch 68 | 0.6266 | 276.84초 |
| VGG16 동결 X | 14,777,098 | Epoch 46 | **0.8158** | 1011.2초 |

### VGG16 cifar100 (keras73_vgg16_02_cifar100)

| 모델 | Trainable params | EarlyStopping | test acc | 소요 시간 |
|---|:---:|:---:|:---:|:---:|
| 이전 최고 (keras40_GAP04, GlobalAveragePooling2D) | - | - | **0.4621** | 227.22초 |
| VGG16 동결 O | 71,500 | Epoch 43 (best 18) | 0.381 | 749.54초 |
| VGG16 동결 X | 14,786,188 | Epoch 28 (best 3) | 0.01 | 369.26초 |

### RAG 챗봇 비교 (rag22_1 · 2 · 3)

| 질문 | rag22_1 (k=2) | rag22_2 (threshold 0.7) | rag22_3 (프롬프트 규칙 · 0.2) |
|---|---|---|---|
| 안녕 | 주어진 정보로는 답변할 수 없습니다. | 주어진 정보로는 답변할 수 없습니다. | 안녕! 어떻게 도와줄까? |
| Multi-Head Attention 은 무엇이야? | 논문 내용으로 답변 (검색 문서 2개 출력) | 주어진 정보로는 답변할 수 없습니다. | 논문 내용으로 답변 |
| 삼성전자의 사업 전망은? | 답변할 수 없음 | 주어진 정보로는 답변할 수 없습니다. | 주어진 정보로는 답변할 수 없습니다. |
| (나는 AI 개발자다. 다음) 내가 누구게 | 답변할 수 없음 | 주어진 정보로는 답변할 수 없습니다. | 이전 대화에서 AI 개발자라고 말씀하셨습니다. |

- rag22_3의 "내가 누구게"는 3번 중 2번 AI 개발자라고 답했다 (나머지 1번은 질문을 바꾸는 단계에서 논문 문서가 검색되어 거절)

---

## 💻 핵심 개념

### 동결 전후 비교

```
                     model.weights   model.trainable_weights   summary
trainable = True     6개             6개                        Trainable 17 / Non-trainable 0
trainable = False    6개             0개 []                     Trainable 0  / Non-trainable 17
                     (값은 그대로)    (fit 해도 갱신 안 됨)
```

### VGG16 include_top

```
include_top=True  (224, 224, 3) ─▶ block1 ~ block5 (Conv2D 13 + MaxPooling 5) ─▶ (7, 7, 512) ─▶ Flatten 25,088 ─▶ fc1 4096 ─▶ fc2 4096 ─▶ predictions 1000
                                    └──────────── 14,714,688 ────────────┘                    └──────────── 123,642,856 (top) ───────────┘
include_top=False (32, 32, 3)   ─▶ block1 ~ block5 ─▶ (1, 1, 512)  ─▶ 직접 붙이는 Flatten ─▶ Dense ─▶ Dense ─▶ Dense(10 or 100, softmax)
```

### 전이학습 동결 O / 동결 X

```
동결 O (vgg16.trainable=False)   [VGG16 imagenet 가중치 고정] ─▶ [Flatten · Dense 훈련]   Trainable 62,410
동결 X (vgg16.trainable=True)    [VGG16 imagenet 가중치에서 출발해 같이 훈련] ─▶ [Flatten · Dense 훈련]   Trainable 14,777,098
```

### RAG 검색 결과에 따른 답변 흐름

```
질문 ─▶ retriever ─┬─ k=2 (threshold 없음)  ─▶ 항상 문서 2개 ─▶ 규칙 3번으로 관계없으면 거절           (rag22_1)
                   ├─ threshold 0.7         ─▶ 모든 질문 0개   ─▶ if not context → NO_ANSWER          (rag22_2)
                   └─ threshold 0.2         ─▶ 논문 질문만 문서 ─▶ 인사 · 이전 대화는 규칙 1 · 2로 답변 (rag22_3)
```

---

## 성과

- `model.weights` · `trainable_weights`로 가중치를 꺼내고, `trainable = False`로 동결하면 훈련해도 가중치가 그대로라는 것을 확인했다
- 훈련된 가중치만으로 예측값 1.0002013을 손으로 계산해 Dense 연산과 activation 없는 층의 의미를 확인했다
- VGG16의 구조 · 파라미터와 `include_top`에 따라 달라지는 부분을 summary로 비교했다
- VGG16 전이학습(동결 X)으로 cifar10 이전 최고 기록 0.7727을 **0.8158**로 갱신했다
- cifar100에서는 동결 O · 동결 X 모두 이전 기록보다 낮았고, 동결 X는 학습이 되지 않는 것(loss = ln100)을 확인했다
- `response['context']`로 실제 검색된 문서를 확인하고, `score_threshold`를 0.7로 올리면 모든 검색 결과가 사라지는 것을 확인했다
- 프롬프트 규칙을 바꿔 인사 · 이전 대화 질문에 답하면서 논문 밖 질문은 거절하는 챗봇을 만들었다

---

## 💡 주요 학습 포인트

1. **`model.weights`는 층마다 kernel과 bias**: 처음 kernel은 랜덤, bias는 0
2. **`model.trainable = False`는 가중치를 지우지 않는다**: `trainable_weights`만 비고 Non-trainable로 옮겨 간다
3. **동결한 모델도 compile · fit은 된다**: 가중치가 갱신되지 않아 처음 랜덤 가중치로 예측한다
4. **Dense 연산은 `y = x · W + b`**: 가중치만 있으면 예측값을 손으로 계산할 수 있다
5. **activation이 없으면 층을 쌓아도 1차 함수 하나**: `a = W1·W2·W3`, `c = b1·W2·W3 + b2·W3 + b3`
6. **`keras.applications`로 미리 훈련된 모델을 불러온다**: `VGG16(weights='imagenet')`
7. **VGG16 파라미터의 약 74%는 fc1 하나**: Flatten 25,088 x 4,096
8. **`include_top=False`면 분류 부분이 빠지고 입력 크기를 바꿀 수 있다**: True는 224x224만
9. **Conv2D 파라미터는 입력 크기와 상관없다**: `(224, 224, 3)`이든 `(32, 32, 3)`이든 14,714,688
10. **`include_top=False` 입력은 32 이상**: MaxPooling 5번으로 32 → 1
11. **전이학습은 특징 추출 부분을 가져오고 분류 부분은 직접 붙인다**: `model.add(vgg16)` + Flatten + Dense
12. **`vgg16.trainable`은 VGG16 부분에만 적용된다**: 아래에 붙인 Dense는 항상 훈련된다
13. **동결 O는 빠르지만 내 데이터에 맞게 특징을 고치지 못한다**: cifar10 0.6266
14. **동결 X는 느리지만 imagenet 가중치에서 출발해 더 잘 맞출 수 있다**: cifar10 0.8158, epoch당 약 5배
15. **동결 X는 학습률이 크면 미리 훈련된 가중치가 망가질 수 있다**: cifar100 loss = ln(100), acc = 0.01
16. **비교할 때는 모델 외 설정을 같게 둔다**: 스케일링 · 원핫 · compile · EarlyStopping · fit 동일
17. **`response['context']`로 실제 검색된 문서를 확인한다**: threshold 없으면 관계없는 질문도 k개가 들어온다
18. **threshold는 임베딩 모델의 점수 분포에 맞춰 정한다**: Qwen3는 0.7이면 논문 질문도 0개
19. **검색 문서 0개는 직접 처리할 수 있다**: `if not response['context']: return NO_ANSWER`
20. **프롬프트 규칙으로 답하는 범위를 정한다**: 인사 · 이전 대화는 답하고, 논문 밖 질문은 정해진 문장으로 거절

---

[⬅️ Day27](Day27.md) · [🏠 전체 목차](../README.md) · [Day29 ➡️](Day29.md)
