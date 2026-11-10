# Day43 - KFold 교차 검증 Kaggle 데이터 적용 & train_test_split + cross_val_predict & all_estimators 전체 모델 비교 (KFold) · furiosa-llm LLM · Tool calling

**학습 기간:** 2026-11-03

> Day42의 `KFold` + `cross_val_score` 교차 검증을 Kaggle Bike Sharing(회귀) · Santander(이진 분류)에 적용했다. 이어서 `train_test_split`으로 test를 먼저 떼어 두고 **train으로 교차 검증 → test는 `cross_val_predict`로 예측 → `accuracy_score` / `r2_score`로 평가**하는 흐름(m03)을 iris · california · bike · santander에 만들었다. 마지막으로 `all_estimators`로 sklearn에 들어 있는 **분류 모델 44개 · 회귀 모델 55개를 for 문으로 전부 돌려** 비교했고(m04), 여기에 KFold 교차 검증을 합쳐 모델마다 5번 평균 점수로 비교했다(m05).
>
> RNGD 에서는 서버 없이 파이썬 코드 안에서 `furiosa_llm.LLM` 으로 모델을 직접 올려 `SamplingParams` 로 생성 옵션을 정하고 답변을 받았다(f02). 또 `furiosa-llm serve` 서버에 `tools` · `tool_choice` 를 넘겨 모델이 함수 이름과 인자(JSON)를 골라 주는 Tool calling 을 했고(f03), `fxb` 인자로 컴파일된 모델 파일(.fxb)을 직접 지정해 불러왔다(f04).

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| Kaggle 교차 검증 | `pd.read_csv` → x / y 분리 → `cross_val_score` (bike : RandomForestRegressor, santander : XGBClassifier) |
| train_test_split + KFold | test를 먼저 떼어 두고 train 120개로만 교차 검증 |
| cross_val_predict | `cross_val_predict(model, x_test, y_test, cv=kfold)` : x_test를 다시 5조각으로 나눠 훈련 · 예측 → 전체 예측값 |
| 점수 함수 | Classifier → `accuracy_score(y_test, y_predict)`, Regressor → `r2_score` |
| all_estimators | `all_estimators(type_filter="classifier" / "regressor")` → (이름, 모델 클래스) 리스트 44개 / 55개 |
| 전체 모델 돌리기 | `for name, model in all:` → `model()` 로 생성 → `try / except` 로 오류 모델 건너뜀 |
| all_estimators + KFold | 모델마다 `cross_val_score` 5번 평균 → 한 번 나눈 test 점수보다 믿을 만한 비교 |
| furiosa_llm.LLM | 서버 없이 파이썬 안에서 RNGD 에 모델을 올려 `llm.generate` 로 답변 생성 |
| SamplingParams | `max_tokens` · `top_p` · `top_k` 로 생성 길이와 다음 토큰 후보를 정한다 |
| Tool calling | `tools` 에 함수 설명(JSON 스키마)을 넘기면 모델이 호출할 함수 이름 · 인자를 돌려준다 |
| fxb | `LLM(모델, fxb="경로.fxb")` : RNGD 용으로 컴파일된 모델 파일을 직접 지정 |

---

## 📖 핵심 학습 내용

### 1. KFold 교차 검증 Kaggle 데이터 적용 (m02_kfold_05_kaggle_bike · m02_kfold_07_santander)

Day42의 m02_00 / m02_01을 베이스로, keras 실습과 같은 방식으로 Kaggle csv를 불러와 교차 검증했다.

```python
path = "./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)    # [10886 rows x 11 columns]

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   # (10886, 8)
y = train_csv['count']                                          # (10886,)

kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
model = RandomForestRegressor(random_state=333)
scores = cross_val_score(model, x, y, cv=kfold)
```

```python
path = "./_data/kaggle_santander/"
x = train_csv.drop(['target'], axis=1)      # (200000, 200)
y = train_csv['target']                     # 0 : 179902 / 1 : 20098

model = XGBClassifier(random_state=333)
scores = cross_val_score(model, x, y, cv=kfold)
```

- bike : `casual + registered = count`라서 두 컬럼은 정답을 알려 주는 셈 → x에서 뺀다
- test.csv는 정답이 없는 제출용이라 교차 검증에는 train.csv만 쓴다
- 모델에 `random_state`를 줘서 실행할 때마다 같은 점수가 나온다
- santander는 target이 0 : 약 90% / 1 : 약 10% → 전부 0으로 찍어도 acc 0.8995, 그보다 얼마나 높은지를 봐야 한다

| 파일 | 모델 | fold별 점수 | 평균 |
|---|---|---|:---:|
| m02_05 (bike) | RandomForestRegressor | r2 `[0.2917 0.2747 0.2978 0.3066 0.3023]` | 0.29 |
| m02_07 (santander) | XGBClassifier | acc `[0.9142 0.9112 0.9133 0.9111 0.9145]` | 0.9128 |

### 2. train_test_split + KFold + cross_val_predict (m03_kfold_train_test_split00_iris)

m02_00을 베이스로, test를 먼저 떼어 두고 train으로만 교차 검증한 뒤 test는 `cross_val_predict`로 예측했다.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
model = DecisionTreeClassifier()

#3. 훈련, 평가
scores = cross_val_score(model, x_train, y_train, cv=kfold)
# 평균 acc :  0.97

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_test)       # [2 0 2 1 1 1 2 0 2 0 0 0 0 2 2 2 0 2 0 1 2 0 1 1 2 0 1 1 1 1] -> 원값
print(y_predict)    # [2 0 2 1 1 1 1 0 2 0 0 0 0 2 2 2 0 1 0 1 1 0 1 1 2 0 1 1 1 1] -> 예측값

# accuracy_score 만들어 보기
acc = accuracy_score(y_test, y_predict)
# cross_val_predict acc :  0.9 → 30 개 중 27 개 맞힘
```

- `StandardScaler` : x_train에만 `fit_transform`, x_test는 `transform`만 (데이터 누수 방지)
- `cross_val_score`는 train 120개를 5조각으로 교차 검증 (훈련 96 / 검증 24 × 5번)
- `cross_val_predict`는 앞의 train 모델로 예측하는 것이 아니라, **x_test 30개를 다시 5조각으로 나눠 24개로 훈련 · 6개 예측을 5번** 반복해 30개 전체의 예측값을 만든다 → 훈련 데이터가 적어서 교차 검증 평균보다 낮게 나온다
- `accuracy_score(y_test, y_predict)` : 맞힌 개수 / 전체 개수

### 3. train_test_split + cross_val_predict 데이터셋 적용 (m03_01 · m03_05 · m03_07)

| 파일 | 베이스 | 바뀐 점 |
|---|---|---|
| m03_01 (california) | m03_00 | `DecisionTreeRegressor`, `stratify` 제거 (회귀), `accuracy_score` → `r2_score` |
| m03_05 (bike) | m03_01 | 데이터 불러오기 · 모델은 m02_05와 같다 (`RandomForestRegressor`) |
| m03_07 (santander) | m03_00 | 데이터 불러오기 · 모델은 m02_07과 같다 (`XGBClassifier`), `stratify=y` 유지 |

```python
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, y_predict)        # 회귀 : r2_score
```

| 파일 | 교차 검증 평균 (x_train) | cross_val_predict (x_test) |
|---|:---:|:---:|
| m03_00 (iris) | acc 0.97 | acc 0.9 |
| m03_01 (california) | r2 0.61 | r2 0.5367 |
| m03_05 (bike) | r2 0.30 | r2 0.1943 |
| m03_07 (santander) | acc 0.9128 | acc 0.908 |

- 네 파일 모두 cross_val_predict 점수가 더 낮다 → x_test만으로 다시 훈련해서 훈련 데이터가 적기 때문

### 4. all_estimators 분류 모델 전체 돌리기 (m04_all_estimators1_classifier)

m03_00의 데이터 준비(train_test_split + StandardScaler)를 그대로 두고, sklearn 분류 모델 전체를 for 문으로 돌렸다.

```python
from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings("ignore")

all = all_estimators(type_filter="classifier")
print("sk version : ", sk.__version__)          # sk version :  1.9.1
print("모델의 갯수 : ", len(all))                 # 모델의 갯수(classifier) :  44

for name, model in all:
    try:
        model = model()

        #3. 훈련
        model.fit(x_train, y_train)

        #4. 평가
        acc = model.score(x_test, y_test)
        print(name, "의 정답률 : ", acc)

    except:
        print(name, "모델 오류")
```

- `all_estimators(type_filter=...)` : `(이름, 모델 클래스)` 튜플 리스트 → `model()`로 기본값 모델을 만든다
- `try / except` : 다른 모델을 인자로 받아야 하는 모델 등은 오류 → 건너뛰고 다음 모델로
- `warnings.filterwarnings("ignore")` : 모델마다 나오는 경고를 숨긴다

```
LinearDiscriminantAnalysis 의 정답률 :  0.9666666666666667       ← 가장 높음
QuadraticDiscriminantAnalysis 의 정답률 :  0.9666666666666667
GaussianNB · GradientBoosting · LogisticRegression · RandomForest · SVC 등 : 0.9333333333333333
DummyClassifier 의 정답률 :  0.3333333333333333                  ← 한 클래스로만 찍는 기준 모델
```

- 44개 중 31개 성공 · 13개 모델 오류

### 5. all_estimators 회귀 모델 전체 돌리기 (m04_all_estimators2_regressor)

m04_1을 베이스로 회귀 모델로 바꿨다.

```python
x, y = load_diabetes(return_X_y=True)      # (442, 10) 회귀 데이터

all = all_estimators(type_filter="regressor")   # 모델의 갯수(regressor) :  55

r2 = model.score(x_test, y_test)        # Regressor : R2
print(name, "의 r2 : ", r2)
```

- 회귀 데이터라서 iris → `load_diabetes`, `stratify` 제거
- 회귀 모델의 `model.score`는 정답률이 아니라 **R2** → 음수도 나온다 (y 평균으로만 찍는 것보다 못하면 음수)
- 55개 중 43개 성공 · 12개 모델 오류, 1등 `ElasticNet` r2 0.4875, 꼴찌 `KernelRidge` -3.7129

### 6. all_estimators + KFold 교차 검증 (m05_all_estimators1_kfold · m05_all_estimators2_kfold)

m04의 for 문 안에 m03의 교차 검증 흐름을 넣었다.

```python
for name, model in all:
    try:
        model = model()

        #3. 훈련, 평가 (교차 검증)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)

        #4. 예측
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_predict)         # 회귀 : r2_score

        print("==============", name, "==============")
        print("acc : ", scores, "\n평균 acc : ", round(np.mean(scores), 4))
        print("cross_val_predict acc : ", round(acc, 4))

    except:
        print(name, "모델 오류")
```

| 파일 | 데이터 | 성공 / 오류 | 교차 검증 1등 |
|---|---|:---:|---|
| m05_1 (classifier) | iris | 30 / 14 | LinearDiscriminantAnalysis 평균 acc 0.975 |
| m05_2 (regressor) | diabetes | 43 / 12 | ARDRegression 평균 r2 0.5055 |

- m04_2(test 한 번 평가)에서는 `ElasticNet`이 1등이었지만, 5번 평균으로 보면 `ARDRegression` · `Lasso` 등이 앞선다 → 한 번 나눈 test 점수만 보고 모델을 고르면 운에 좌우될 수 있다
- `HistGradientBoostingClassifier`는 평균 acc 0.9417인데 cross_val_predict acc 0.1667 → x_test 24개로 훈련하면 잎 하나 최소 20개(min_samples_leaf) 조건 때문에 트리를 나누지 못하고 한 클래스로만 찍는다
- m05_1의 오류 14개 = m04_1의 13개 + `RadiusNeighborsClassifier` (반경 안에 이웃이 없는 데이터)

### 7. 모델 오류 정리

| 오류 모델 | 이유 |
|---|---|
| `ClassifierChain` · `MultiOutput~` · `OneVsOne~` · `OneVsRest~` · `OutputCode~` · `SelfTraining~` · `Stacking~` · `Voting~` · `FixedThreshold~` · `TunedThreshold~` · `RegressorChain` | 안에 넣을 모델(estimator)을 인자로 줘야 만들 수 있다 |
| `CategoricalNB` · `ComplementNB` · `MultinomialNB` | x에 음수가 있으면 안 된다 (StandardScaler 후 음수) |
| `MultiTaskElasticNet(CV)` · `MultiTaskLasso(CV)` | y가 2차원(출력 여러 개)이어야 한다 |
| `CCA` · `PLSCanonical` | 기본 `n_components=2`인데 y가 1개 |
| `IsotonicRegression` | x가 컬럼 1개일 때만 동작 |
| `RadiusNeighbors~` | 반경(`radius=1.0`) 안에 이웃이 없는 데이터가 있으면 오류 |

### 8. furiosa_llm.LLM 으로 모델 직접 올리기 (f02_tutorial)

Day42 의 f01 은 `furiosa-llm serve` 서버에 OpenAI 클라이언트로 질문을 보냈다. f02 는 서버 없이 파이썬 코드 안에서 `LLM` 으로 RNGD 에 모델을 올린다.

```python
from furiosa_llm import LLM, SamplingParams

with LLM("furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic") as llm:
    params = SamplingParams(max_tokens=1000, top_p=0.3, top_k=100)
    messages = [{"role" : "user", "content" : "오늘 엔비디아 주가가 어떻게 돼?"}]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False)

    response = llm.generate([prompt], params)

    print(response[0].outputs[0].text)
```

- `with LLM(...) as llm:` : 블록이 끝나면 모델을 내리고 NPU 를 돌려준다
- `SamplingParams` 의 `max_tokens` 기본값은 16 → 긴 답변을 받으려면 직접 정한다 (`min_tokens` 는 최소 생성 길이라 `max_tokens` 보다 크면 오류)
- `top_k=100` : 확률 높은 토큰 100개 중에서, `top_p=0.3` : 누적 확률 30% 안의 토큰 중에서 다음 토큰을 고른다
- `apply_chat_template(messages, tokenize=False)` : messages 를 모델의 대화 형식 문자열로 바꾼다
- `llm.generate([prompt], params)` → 프롬프트마다 결과, `response[0].outputs[0].text` 가 답변
- 모델은 실시간 정보를 모르므로 "오늘 주가" 는 답하지 못하고 확인 방법을 안내했다

### 9. Tool calling (f03_tool_calling)

f01 처럼 `furiosa-llm serve` 서버에 OpenAI 클라이언트로 요청하면서 `tools` 를 같이 보낸다.

```
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:0 \
    --host 0.0.0.0 \
    --port 8000
```

```python
def get_weather(location:str, unit:str):
    return f"Getting the weather for {location} in {unit}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location", "unit"]
            }
        }
    },
    ...     # get_time(timezone)
]

response = client.chat.completions.create(
    model = client.models.list().data[0].id,
    messages = [{"role" : "user", "content" : "오늘 판교의 날씨는 어때?"}],
    tools = tools,
    tool_choice = {"type" : "function", "function" : {"name" : "get_weather"}},
    temperature = 0.0,
)

tool_call = response.choices[0].message.tool_calls[0].function
print(f"Result : {get_weather(**json.loads(tool_call.arguments))}")
```

- `tools` : 함수 이름 · 설명 · 인자(JSON 스키마)를 모델에게 알려 준다. `enum` 으로 고를 수 있는 값을 정하고 `required` 로 필수 인자를 정한다
- `tool_choice` : 호출할 함수를 `get_weather` 로 지정 (지정하지 않으면 모델이 고른다)
- 모델은 함수를 직접 실행하지 않고 **함수 이름과 인자(JSON 문자열)** 만 돌려준다 → `json.loads` 로 dict 로 바꿔 `**` 로 파이썬 함수에 넣어 실행한다
- `message.tool_calls` 는 리스트 (함수를 여러 개 부를 수 있다) → `[0]`
- `os.getenv("OPENAI_BASE_URL", 기본값)` : 환경 변수가 없으면 로컬 서버 주소를 쓴다

```
Arguments : {"location": "판교, 한국", "unit": "celsius"}
Result : Getting the weather for 판교, 한국 in celsius
```

### 10. fxb 파일 직접 지정 (f04_fxb_direct_select)

```python
from furiosa_llm import LLM

llm = LLM("Qwen/Qwen3-8B-FP8", fxb="/root/qwen3-8b-fp8.fxb")
```

- `.fxb` : RNGD 에서 실행할 수 있게 컴파일된 모델 파일. `furiosa-ai/...` 모델은 Hugging Face 저장소 안의 fxb 를 자동으로 쓰고, `fxb=` 로 파일을 직접 지정할 수도 있다 (서버는 `furiosa-llm serve 모델 --fxb 경로`)
- 상대 경로는 실행 위치 기준이라 다른 폴더에서 실행하면 파일을 못 찾는다 → 절대 경로
- NPU 를 다른 프로세스(실행 중인 `furiosa-llm serve`)가 쓰고 있으면 `No available devices found` 오류 → `furiosa-smi ps` 로 확인하고 서버를 끈 뒤 실행

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `ml/m02_kfold_05_kaggle_bike.py` | Kaggle Bike `RandomForestRegressor` + `cross_val_score` → 평균 r2 0.29 |
| `ml/m02_kfold_07_santander.py` | Kaggle Santander `XGBClassifier` + `cross_val_score` → 평균 acc 0.9128 |
| `ml/m03_kfold_train_test_split00_iris.py` | iris train_test_split + StandardScaler + `cross_val_score` + `cross_val_predict` + `accuracy_score` |
| `ml/m03_kfold_train_test_split01_california.py` | california `DecisionTreeRegressor` + `cross_val_predict` + `r2_score` |
| `ml/m03_kfold_train_test_split05_kaggle_bike.py` | Kaggle Bike train_test_split + `cross_val_predict` + `r2_score` |
| `ml/m03_kfold_train_test_split07_santander.py` | Kaggle Santander train_test_split(stratify) + `cross_val_predict` + `accuracy_score` |
| `ml/m04_all_estimators1_classifier.py` | `all_estimators(type_filter="classifier")` 44개 iris `model.score` 비교 |
| `ml/m04_all_estimators2_regressor.py` | `all_estimators(type_filter="regressor")` 55개 diabetes R2 비교 |
| `ml/m05_all_estimators1_kfold.py` | 분류 모델 전체 + KFold `cross_val_score` · `cross_val_predict` (iris) |
| `ml/m05_all_estimators2_kfold.py` | 회귀 모델 전체 + KFold `cross_val_score` · `cross_val_predict` (diabetes) |
| `furiosa01_rngd/f02_tutorial.py` | `furiosa_llm.LLM` + `SamplingParams` 로 Llama-3.1-8B 를 직접 올려 `llm.generate` |
| `furiosa01_rngd/f03_tool_calling.py` | `furiosa-llm serve` Qwen3-8B 에 `tools` · `tool_choice` 로 Tool calling → `get_weather` 실행 |
| `furiosa01_rngd/f04_fxb_direct_select.py` | `LLM(..., fxb=)` 로 컴파일된 fxb 파일을 직접 지정해 모델 불러오기 |
| `ml/m02_kfold_00_iris.py` | `#3. 훈련, 평가` 섹션 이름 수정 (수정) |
| `ml/m02_kfold_01_california.py` | `#3. 훈련, 평가` 섹션 이름 수정 (수정) |
| `docs/Day43.md` | Day43 학습 기록 신규 작성 |
| `docs/Day42.md` | 코드 예시의 섹션 이름 반영, 하단 nav에 Day43 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(ml/ · furiosa01_rngd/ Day43) / 진행도(43일, 54%) 갱신 (수정) |

---

## 📊 실행 결과

| 파일 | 결과 |
|---|---|
| m02_05 (bike) | 평균 r2 0.29 |
| m02_07 (santander) | 평균 acc 0.9128 (전부 0일 때 0.8995) |
| m03_00 (iris) | 평균 acc 0.97 / cross_val_predict acc 0.9 |
| m03_01 (california) | 평균 r2 0.61 / cross_val_predict r2 0.5367 |
| m03_05 (bike) | 평균 r2 0.30 / cross_val_predict r2 0.1943 |
| m03_07 (santander) | 평균 acc 0.9128 / cross_val_predict acc 0.908 |
| m04_1 (classifier 44개) | 31개 성공, LDA · QDA 0.9667 |
| m04_2 (regressor 55개) | 43개 성공, ElasticNet 0.4875 |
| m05_1 (classifier + KFold) | 30개 성공, LDA 평균 acc 0.975 |
| m05_2 (regressor + KFold) | 43개 성공, ARDRegression 평균 r2 0.5055 |
| f02 (LLM + SamplingParams) | 실시간 주가는 모른다고 답하고 확인 방법(증권 앱 · 금융 사이트) 안내 |
| f03 (Tool calling) | `get_weather` · `{"location": "판교, 한국", "unit": "celsius"}` |
| f04 (fxb 지정) | Qwen3-8B fxb 로딩 성공 (출력 없음) |

- `random_state`를 주지 않은 모델(DecisionTree · ExtraTree · MLP 등)은 실행할 때마다 점수가 조금씩 달라진다

---

## 💻 핵심 개념

### m02 → m03 → m04 → m05 흐름

```
m02 : 전체 x, y            → cross_val_score                           (교차 검증만)
m03 : train_test_split     → cross_val_score(x_train)                  (train 교차 검증)
                           → cross_val_predict(x_test) → accuracy_score / r2_score
m04 : all_estimators       → for 문으로 모델 전체 fit → score(x_test)   (한 번 평가)
m05 : all_estimators + m03 → 모델마다 cross_val_score 평균 + cross_val_predict
```

### cross_val_score vs cross_val_predict

```
cross_val_score(model, x, y, cv=kfold)    → fold 별 점수 배열   [0.95 0.95 0.95 1.0 0.95]
cross_val_predict(model, x, y, cv=kfold)  → 데이터별 예측값 배열 [2 0 2 1 1 ...]
                                             (각 데이터는 자기가 검증 조각일 때의 모델로 예측된다)
```

### all_estimators

```
all = all_estimators(type_filter="classifier")   # [('AdaBoostClassifier', <class>), ...]  44 개
all = all_estimators(type_filter="regressor")    # 55 개 (sklearn 1.9.1)

for name, model in all:
    try:
        model = model()          # 클래스 → 기본값 모델
        ...
    except:
        print(name, "모델 오류")
```

---

## 성과

- Day42의 KFold 교차 검증을 Kaggle Bike · Santander에 적용했다
- test를 먼저 떼어 두고 train으로 교차 검증, test는 `cross_val_predict`로 예측해 `accuracy_score` / `r2_score`로 평가하는 흐름을 4개 데이터에 만들었다
- `all_estimators`로 sklearn 분류 44개 · 회귀 55개 모델을 한 번에 돌려 비교하고, 오류 나는 모델의 이유를 정리했다
- 전체 모델 비교에 교차 검증을 합쳐, 한 번 평가와 5번 평균의 순위가 다를 수 있음을 확인했다
- RNGD 에서 서버 없이 `LLM` 으로 모델을 올려 답변을 생성하고, fxb 파일을 직접 지정해 불러왔다
- `tools` 로 함수 설명을 넘겨 모델이 고른 함수 이름 · 인자로 파이썬 함수를 실행했다

---

## 💡 주요 학습 포인트

1. **Kaggle 데이터도 같은 흐름**: `pd.read_csv` → x / y 분리 → `cross_val_score`
2. **정답을 알려 주는 컬럼은 뺀다**: bike의 casual · registered (합이 count)
3. **불균형 데이터의 acc는 기준선과 비교**: santander는 전부 0이어도 0.8995
4. **test는 먼저 떼어 두고 train으로 교차 검증**: 스케일러도 x_train에만 fit
5. **`cross_val_predict`는 점수가 아니라 예측값**: x_test 안에서 다시 나눠 훈련하므로 점수가 낮게 나올 수 있다
6. **`accuracy_score` / `r2_score`로 예측값 평가**: (실제값, 예측값) 순서
7. **`all_estimators`로 모델 전체 비교**: `type_filter`로 classifier / regressor 선택, `model()`로 생성
8. **`try / except`로 오류 모델 건너뛰기**: 인자가 필요한 모델, 음수 입력 불가 모델 등
9. **회귀 `score`는 R2**: 정답률이 아니고 음수도 나온다
10. **한 번 평가보다 교차 검증 평균**: m04_2와 m05_2의 1등이 다르다
11. **`SamplingParams` 의 `max_tokens` 기본값은 16**: 긴 답변은 직접 늘린다
12. **Tool calling 은 모델이 함수를 고르기만 한다**: 실행은 `json.loads(arguments)` 로 우리 코드가 한다
13. **NPU 는 한 프로세스만 쓴다**: 서버가 떠 있으면 `LLM()` 이 장치를 못 찾는다 (`furiosa-smi ps`)

---

[⬅️ Day42](Day42.md) · [🏠 전체 목차](../README.md) · [Day44 ➡️](Day44.md)
