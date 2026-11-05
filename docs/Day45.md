# Day45 - 파이썬 모듈 · 패키지 import & 데이터셋별 all_estimators 최고 모델 비교 & for 문 하이퍼파라미터 튜닝 · GridSearchCV & furiosa-llm LangChain 연결 · 스트리밍

**학습 기간:** 2026-11-05

> 파이썬 파일(.py) 하나가 모듈이라는 것부터 시작해, `import 모듈` · `from 모듈 import 함수` 로 다른 파일의 함수를 불러 쓰고, `__name__` 과 `if __name__ == "__main__":` 으로 **직접 실행할 때와 import 될 때를 구분**했다. 이어서 `machine` 폴더(패키지)와 그 하위 폴더 `machine/test` 안의 모듈을 `import 폴더.모듈` · `from 폴더 import 모듈` 로 불러왔다(python/).
>
> 머신러닝에서는 Day43 의 all_estimators + 교차 검증을 **여러 데이터셋에 한 번에** 돌려 데이터마다 최고 모델을 찾았다(분류 4 개 · 회귀 2 개, m06). 그다음 하이퍼파라미터 조합을 **이중 for 문으로 직접 바꿔 가며** 최고 점수와 최적 매개변수를 찾았고(m07), 이를 `GridSearchCV` 로 바꿔 파라미터 42 조합을 교차 검증으로 자동 탐색했다(iris · digits · diabetes, m08).
>
> RNGD 에서는 `furiosa-llm serve` 로 띄운 서버를 OpenAI 클라이언트 대신 LangChain `ChatOpenAI` 로 불러 `llm.invoke` 로 답을 받았고(f05), 응답 시간을 `time.perf_counter` 로 재서 초당 토큰 수를 계산했다(f06). 답변을 한 번에 받지 않고 조각(chunk)으로 받는 **스트리밍**을 OpenAI 클라이언트(`stream=True`)와 LangChain(`llm.stream`) 두 방식으로 해 봤고(f07), 오늘 배운 모듈 import 를 이용해 LLM 설정을 `a_kingkeras.py` 의 `get_llm()` 함수로 빼서 다른 파일에서 불러 썼다(f08).

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 모듈 | 파이썬 파일 하나 = 모듈 하나 → `import p01_car` → `p01_car.drive()` |
| import 시 실행 | import 하는 순간 그 파일이 위에서부터 한 번 실행된다 |
| `__name__` | 직접 실행하면 `"__main__"`, import 되면 파일 이름(`"p11_car"`) |
| `if __name__ == "__main__":` | 직접 실행할 때만 안쪽 코드 실행, import 때는 건너뜀 |
| 이름 공간 | 모듈마다 변수가 따로 → 같은 이름 `x` 라도 모듈마다 다른 값 |
| `from ~ import` | `from p31_sample import test` → 모듈 이름 없이 `test()` |
| 패키지 | 폴더 안의 모듈 → `import machine.car` / `from machine import car, tv` / `from machine.test import car` |
| 여러 데이터셋 비교 | `datasets` 리스트 + `enumerate` → 데이터마다 all_estimators 전체 모델 교차 검증 |
| StratifiedKFold | fold 마다 클래스 비율을 맞춰 나눈다 (분류), 회귀는 `KFold` |
| 최고 모델 찾기 | 이름 · 점수 리스트 → `np.array` → `np.argmax` 로 최고 위치 |
| for 문 튜닝 | learning_rate 6 개 × max_depth 6 개 = 36 조합 → 최고 점수일 때의 값 저장 |
| GridSearchCV | 파라미터 dict 의 리스트 42 조합 × 5 fold = 210 번 훈련 → `best_params_` · `best_score_` |
| ChatOpenAI 로 RNGD 서버 호출 | `base_url="http://127.0.0.1:8000/v1"` · `api_key="EMPTY"` → `llm.invoke(질문).content` |
| 처리 속도 측정 | `time.perf_counter()` 로 시작 · 끝 시간 → `max_tokens / 걸린 시간` = tokens/s |
| 스트리밍 | OpenAI `stream=True` → `chunk.choices[0].delta.content` / LangChain `llm.stream()` → `chunk.content` |
| LLM 설정 모듈화 | `a_kingkeras.py` 의 `get_llm()` → `from a_kingkeras import get_llm` |

---

## 📖 핵심 학습 내용

### 1. 모듈 만들고 import 하기 (p01_car · p02_tv · p03_do)

```python
# p01_car.py
def drive():
    print("운전하다")

drive()
```

```python
# p03_do.py
import p01_car
import p02_tv

print("==========")
p01_car.drive()
p02_tv.watch()
```

```
운전하다       ← import p01_car 때 실행
시청하다       ← import p02_tv 때 실행
==========
운전하다       ← p01_car.drive()
시청하다       ← p02_tv.watch()
```

- 같은 폴더의 파이썬 파일은 `import 파일이름` (`.py` 없이) 으로 불러오고 `모듈이름.함수()` 로 호출한다
- import 하는 순간 그 파일이 한 번 실행된다 → 파일 맨 아래의 `drive()` · `watch()` 호출도 같이 실행된다

### 2. `__name__` 확인 (p11_car · p12_tv · p13_do)

```python
# p11_car.py
print("p11_car.py의 module 이름은 : ", __name__)
# p11_car.py의 module 이름은 :  __main__      ← 직접 실행
```

```python
# p13_do.py
import p11_car
import p12_tv
# p11_car.py의 module 이름은 :  p11_car       ← import 되면 파일 이름
# p12_tv.py의 module 이름은 :  p12_tv

print("p13_do.py의 module 이름은 : ", __name__)
# p13_do.py의 module 이름은 :  __main__        ← 직접 실행한 파일만 __main__
```

### 3. `if __name__ == "__main__":` (p21_car · p22_tv · p23_main_do)

```python
# p21_car.py
def drive():
    print("운전하다")

drive()

if __name__ == "__main__":
    drive()
```

| 실행 방법 | `__name__` | 출력 |
|---|---|---|
| `python p21_car.py` (직접 실행) | `"__main__"` | 운전하다 × 2 (if 밖 + if 안) |
| `import p21_car` (p23_main_do) | `"p21_car"` | 운전하다 × 1 (if 밖만) |

```
# p23_main_do.py 실행 결과
운전하다       ← import p21_car 때 if 밖의 drive()
시청하다       ← import p22_tv 때 if 밖의 watch()
==========
운전하다
시청하다
```

- if 안의 코드는 import 될 때 실행되지 않는다 → import 때 아무것도 실행되지 않게 하려면 호출을 if 안에만 둔다 (machine/car.py)

### 4. 모듈마다 따로인 변수 & from ~ import (p31_sample · p32_main_func · p33_from)

```python
# p31_sample.py
x = 111
def test():
    print("x : ", x)
```

```python
# p33_from.py
from p31_sample import test

x = 222
def main_func():
    print("x : ", x)

main_func()     # x :  222
test()          # x :  111   ← test 는 p31_sample 의 x 를 쓴다
```

- 모듈마다 변수 공간(이름 공간)이 따로 있어서 이름이 같은 `x` 라도 서로 다른 변수다
- `from 모듈 import 함수` 로 가져오면 모듈 이름 없이 `test()` 로 바로 호출한다
- p32_main_func 는 `import p31_sample` → `p31_sample.test()` 로 같은 결과 (`x :  111`)

### 5. 패키지(폴더) 안의 모듈 import (p41 · p42 · p51, machine/)

```
python/
├── machine/
│   ├── car.py          def drive() → "machine 운전하다"
│   ├── tv.py           def watch() → "machine 시청하다"
│   └── test/
│       ├── car.py
│       └── tv.py
├── p41_import_module.py
├── p42_from_module.py
└── p51_from_module.py
```

```python
# machine/car.py
def drive():
    print("machine 운전하다")

if __name__ == "__main__":
    drive()
```

```python
# p41_import_module.py
import machine.car
import machine.tv

# car.drive() # (x)
machine.car.drive()
machine.tv.watch()
```

```python
# p42_from_module.py
from machine.car import drive       # 함수만 가져오기
from machine.tv import watch
drive()
watch()

from machine import car, tv         # 모듈 가져오기 (쉼표로 여러 개)
car.drive()
tv.watch()
```

```python
# p51_from_module.py (하위 폴더)
from machine.test import car, tv
car.drive()
tv.watch()
```

| 불러오는 방법 | 호출 |
|---|---|
| `import machine.car` | `machine.car.drive()` (`car.drive()` 는 오류) |
| `from machine.car import drive` | `drive()` |
| `from machine import car, tv` | `car.drive()` · `tv.watch()` |
| `from machine.test import car, tv` | `car.drive()` (machine/test 의 모듈) |

- machine 의 모듈은 호출을 `if __name__ == "__main__":` 안에만 둬서 import 할 때 아무것도 출력되지 않는다

### 6. 여러 데이터셋 × all_estimators 분류 모델 (m06_all_estimators1_to_data)

Day43 의 m05_1(all_estimators + 교차 검증)을 베이스로, 데이터셋 4 개를 for 문으로 돌려 데이터마다 최고 모델을 찾았다.

```python
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digits]
data_name = ['아이리스', '캔서', '와인', '디지트']

all = all_estimators(type_filter="classifier")
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

start = time.time()
for index, value in enumerate(datasets):
    x, y = value
    ...     # train_test_split(stratify=y) + StandardScaler

    model_name = []
    model_acc = []

    for name, model in all:
        try:
            model = model()
            scores = cross_val_score(model, x_train, y_train, cv=kfold)
            y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

            acc = np.round(accuracy_score(y_test, y_pred), 4)
            model_name.append(name)
            model_acc.append(acc)
        except:
            continue

    model_name = np.array(model_name)
    model_acc = np.array(model_acc)
    max_index = np.argmax(model_acc)
    print("최고 모델 : ", model_name[max_index], "ACC : ", model_acc[max_index])

end = time.time()
print("걸린 시간 : ", round(end - start, 2), "초")
```

- `load_~(return_X_y=True)` 는 `(x, y)` 튜플 → 리스트에 담아 두고 `x, y = value` 로 꺼낸다
- `enumerate(datasets)` : `(index, value)` → `data_name[index]` 로 같은 순서의 이름 출력
- `StratifiedKFold` : fold 마다 클래스 비율을 y 전체 비율과 같게 나눈다
- 오류 나는 모델은 `continue` 로 건너뛰고, 성공한 모델의 이름 · acc 만 리스트에 쌓는다
- `np.argmax(model_acc)` : 가장 큰 값의 위치 → 같은 위치의 `model_name` 이 최고 모델
- `time.time()` : 시작 · 끝 시간의 차이로 걸린 시간(초)을 잰다

```
========== 아이리스 ==========
최고 모델 :  BaggingClassifier ACC :  0.9667
========== 캔서 ==========
최고 모델 :  AdaBoostClassifier ACC :  0.9825
========== 와인 ==========
최고 모델 :  NuSVC ACC :  1.0
========== 디지트 ==========
최고 모델 :  SVC ACC :  0.95
걸린 시간 :  90.14 초
```

### 7. 여러 데이터셋 × all_estimators 회귀 모델 (m06_all_estimators2_to_data)

m06_1 을 회귀로 바꿨다.

```python
california = fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)

datasets = [california, diabetes]
data_name = ['캘리포니아', '디아벳']

all = all_estimators(type_filter="regressor")
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

r2 = np.round(r2_score(y_test, y_pred), 4)
```

- y 가 연속값이라 `StratifiedKFold` (클래스 비율 맞추기) 대신 `KFold`, `train_test_split` 의 `stratify` 도 뺀다
- 평가는 `accuracy_score` → `r2_score`

```
========== 캘리포니아 ==========
최고 모델 :  HistGradientBoostingRegressor R2 :  0.7947
========== 디아벳 ==========
최고 모델 :  GammaRegressor R2 :  0.4407
걸린 시간 :  3747.97 초 (약 62 분)
```

- california 는 20640 행이라 55 개 모델을 모두 교차 검증하는 데 오래 걸렸다 (diabetes 는 442 행)

### 8. for 문으로 하이퍼파라미터 튜닝 (m07_for_parameter_tuning)

```python
x, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001, 0.3]
max_depth = [3, 4, 5, 6, 7, 8]
best_score = 0
best_parameters = ''

for i, lr in enumerate(learning_rate): # i = index, lr = value
    for j, md in enumerate(max_depth):
        model = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_depth=md,
        )
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)

        if score > best_score:
            best_score = score
            best_parameters = lr, md
        print(">>>>>>>>>>", i, ",", j, "번째 score : ", round(score, 4), "도는 중... >>>>>>>>>>")

print("최고 점수 : {:.2f}".format(best_score))
print("최적 매개변수 : ", best_parameters)
```

```
최고 점수 : 0.97
최적 매개변수 :  (0.3, 4)
```

- 6 × 6 = 36 가지 조합을 하나씩 훈련 · 평가한다
- 지금까지의 최고 점수보다 높으면 `best_score` · `best_parameters` 를 바꾼다 → `lr, md` 는 튜플 `(0.3, 4)` 로 저장된다
- `"{:.2f}".format(값)` : 소수점 둘째 자리까지 출력
- learning_rate=0.3, max_depth=4 (5 , 1 번째) 가 score 0.9694 로 가장 높다

### 9. GridSearchCV 하이퍼파라미터 탐색 (m08_GridSearchCV_00 · 01_digits · 02_diabetes)

m07 에서 이중 for 문으로 직접 하던 조합 탐색을 `GridSearchCV` 로 바꿨다. 조합마다 교차 검증 평균을 내서 가장 좋은 조합을 자동으로 고른다.

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

x, y = load_iris(return_X_y=True)
...     # train_test_split(stratify=y)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [                                                                                          # 총 42번
    {"n_estimators" : [100, 500], "max_depth" : [6, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},     # 18번
    {"max_depth" : [6, 8, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},                               # 12번
    {"min_child_weight" : [2, 3, 5, 10], "learning_rate" : [0.1, 0.01, 0.001]},                         # 12번
]

#2. 모델 구성
xgb = XGBClassifier()

model = GridSearchCV(
    estimator = xgb, # 하이퍼파라미터를 튜닝할 모델
    param_grid = parameters, # 탐색할 하이퍼파라미터 조합
    cv = kfold, # StratifiedKFold 5-Fold 교차검증
    verbose = 1, # 학습 진행 상황 출력
    n_jobs = -1, # 사용 가능한 모든 CPU 코어를 활용하여 병렬 처리
    # scoring='accuracy',  # 평가 지표: 정확도 (분류 모델의 기본값이므로 생략 가능)
    # refit=True,          # 최적 파라미터로 전체 훈련 데이터를 다시 학습 (기본값)
)

#3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

#4. 평가, 예측
print("model.score : ", model.score(x_test, y_test))
print("best_score : ", model.best_score_)
print("걸린시간 : ", round(end - start, 2), "초")
```

```
Fitting 5 folds for each of 42 candidates, totalling 210 fits
최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
model.score : 0.9
best_score : 0.95
걸린시간 : 4.39 초
```

- `parameters` 는 dict 의 리스트 : dict 하나 안의 값들은 모든 조합을 만들고 (2 × 3 × 3 = 18), dict 끼리는 따로 탐색한다 → 18 + 12 + 12 = 42 조합 × 5 fold = 210 번 훈련
- `n_estimators` · `max_depth` · `learning_rate` · `min_child_weight` 는 XGBoost 의 파라미터, dict 에 없는 파라미터는 기본값
- `refit=True` (기본값) : 탐색이 끝나면 최적 조합으로 x_train 전체를 다시 훈련 → `model.score` · `model.predict` 는 그 최적 모델로 동작
- `best_estimator_` : 최적 조합으로 다시 훈련된 모델 객체 / `best_params_` : 고른 조합 (dict)
- `best_score_` : 최적 조합의 교차 검증 평균 (x_train) / `model.score(x_test, y_test)` : 처음 떼어 둔 test 평가
- `n_jobs=-1` : CPU 코어를 모두 써서 병렬로 훈련

m08_00 을 베이스로 데이터를 바꿔 같은 방법으로 탐색했다.

| 파일 | 데이터 | 바뀐 점 |
|---|---|---|
| m08_01 | digits (1797, 64) | `load_iris` → `load_digits` (나머지 같음) |
| m08_02 | diabetes (442, 10) 회귀 | `XGBRegressor`, `stratify` 제거, `StratifiedKFold` → `KFold`, 점수는 R2 |

| 파일 | 최적 파라미터 | best_score (교차 검증) | model.score (test) | 걸린 시간 |
|---|---|:---:|:---:|:---:|
| m08_00 (iris) | `learning_rate=0.1, max_depth=6, n_estimators=500` | acc 0.95 | acc 0.9 | 4.39 초 |
| m08_01 (digits) | `learning_rate=0.1, max_depth=6, n_estimators=500` | acc 0.961 | acc 0.9472 | 18.52 초 |
| m08_02 (diabetes) | `learning_rate=0.01, min_child_weight=10` | r2 0.3763 | r2 0.352 | 8.16 초 |

- iris · digits 는 첫 번째 dict 에서, diabetes 는 세 번째 dict (min_child_weight) 에서 최적 조합이 나왔다
- digits 는 데이터 · 컬럼 · 클래스가 많아서 같은 210 번 훈련이라도 iris 보다 오래 걸린다
- diabetes 의 R2 0.35 는 Day43 의 선형 모델(ElasticNet 0.4875 · ARDRegression 평균 0.5055)보다 낮다

### 10. LangChain ChatOpenAI 로 furiosa-llm 서버 호출 (f05_langchain)

Day42 의 f01 은 `OpenAI` 클라이언트로 `client.chat.completions.create` 를 불렀다. f05 는 같은 서버(`furiosa-llm serve furiosa-ai/Qwen3-4B-FP8 --devices npu:0 --host 0.0.0.0 --port 8000`)를 LangChain `ChatOpenAI` 로 부른다.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(                            # Request, Response 역할
    base_url = "http://127.0.0.1:8000/v1",
    api_key = "EMPTY"
)

response = llm.invoke("NPU와 GPU의 차이점을 설명해줘")
print(response.content)
```

| OpenAI 클라이언트 (f01) | LangChain ChatOpenAI (f05) |
|---|---|
| `client.models.list().data[0].id` 로 모델 이름 확인 | 모델 이름 생략 |
| `messages=[{"role": "user", "content": ...}]` | 질문 문자열을 바로 `invoke` |
| `response.choices[0].message.content` | `response.content` |

- `base_url` 만 RNGD 서버 주소로 바꾸면 LangGraph 때 쓰던 `ChatOpenAI` 코드를 그대로 쓸 수 있다
- 로컬 서버라 `api_key` 는 아무 값(`"EMPTY"`)이나 넣는다

### 11. 응답 시간 · 초당 토큰 수 측정 (f06_tps)

```python
import time

max_tokens = 1000

llm = ChatOpenAI(
    base_url = "http://127.0.0.1:8000/v1",
    api_key = "EMPTY",
    max_tokens = max_tokens,
    temperature = 0,
    top_p = 0.9,
)

start = time.perf_counter()
response = llm.invoke("NPU와 GPU의 차이점을 설명해줘")
end = time.perf_counter()

elapsed = round(end - start, 2)

print("추론 시간 : ", elapsed, "초")
print("출력 토큰 : ", max_tokens)
print("평균 처리량 : ", round(max_tokens/elapsed, 2), "tokens/s")
```

- `time.perf_counter()` 는 시간 차이를 재는 용도의 정밀한 시계 → `end - start` 가 걸린 초
- `max_tokens` (최대 생성 토큰) · `temperature` (0 = 가장 확률 높은 답) · `top_p` (누적 확률 상위 90% 후보에서 선택) 를 `ChatOpenAI` 에 직접 준다
- 처리량은 `max_tokens` 를 그대로 출력 토큰 수로 보고 나눈 값이다 → 답이 1000 토큰보다 일찍 끝나면 실제보다 크게 계산된다

### 12. 스트리밍 (f07_stream01 · f07_stream02_lc)

`invoke` 는 답변이 다 만들어질 때까지 기다렸다가 한 번에 받는다. 스트리밍은 만들어지는 대로 조각(chunk)을 받아 바로 출력한다.

```python
# f07_stream01.py - OpenAI 클라이언트
streaming = client.chat.completions.create(
    model = model,
    messages=[{"role" : "user", "content" : "NPU와 GPU의 차이점을 설명해줘"}],
    max_tokens=3000,
    stream=True
)

for chunk in streaming:
    if chunk.choices:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
```

```python
# f07_stream02_lc.py - LangChain
for chunk in llm.stream("NPU와 GPU의 차이점을 설명해줘"):
    print(chunk.content, end="", flush=True)
```

- OpenAI 는 `stream=True` 를 주면 응답이 chunk 의 반복이 되고, 내용은 `message` 가 아니라 `delta.content` 에 들어 있다
- `choices` 가 빈 chunk · `content` 가 `None` 인 chunk 가 섞여 오므로 `if` 로 거른다
- LangChain 은 `invoke` 대신 `stream` 만 부르면 되고 chunk 마다 `.content`
- `end=""` 로 줄바꿈 없이 이어 붙이고, `flush=True` 로 받는 즉시 화면에 출력

### 13. LLM 설정을 모듈로 분리 (a_kingkeras · f08_pkg01_invoke · f08_pkg02_stream)

오늘 앞에서 배운 모듈 import 를 그대로 적용해, 파일마다 반복되던 `ChatOpenAI(...)` 설정을 함수 하나로 뺐다.

```python
# a_kingkeras.py
from langchain_openai import ChatOpenAI

def get_llm(
        max_tokens = 1000,
        temperature = 0,
        top_p = 0.9,
):
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
# f08_pkg01_invoke.py - 값을 바꿔서 호출
from a_kingkeras import get_llm

llm = get_llm(
    max_tokens = 1000,
    temperature = 0.9,
    top_p = 1.0,
)

response = llm.invoke("NPU와 GPU의 차이점을 설명해줘")
print(response.content)
```

```python
# f08_pkg02_stream.py - 기본값 그대로 스트리밍
from a_kingkeras import get_llm

llm = get_llm()

for chunk in llm.stream("NPU와 GPU의 차이점을 설명해줘"):
    print(chunk.content, end="", flush=True)
```

- 함수 매개변수에 기본값을 두면 `get_llm()` 처럼 비워 부를 수도, 필요한 값만 바꿔 부를 수도 있다
- 서버 주소가 바뀌어도 `a_kingkeras.py` 한 곳만 고치면 된다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `python/p01_car.py` · `python/p02_tv.py` | `drive()` · `watch()` 함수를 만들고 호출하는 모듈 |
| `python/p03_do.py` | `import p01_car` · `import p02_tv` → `모듈.함수()` 호출 |
| `python/p11_car.py` · `python/p12_tv.py` | 직접 실행 시 `__name__` = `"__main__"` |
| `python/p13_do.py` | import 된 모듈의 `__name__` = 파일 이름 |
| `python/p21_car.py` · `python/p22_tv.py` | `if __name__ == "__main__":` |
| `python/p23_main_do.py` | import 때 if 안의 호출은 실행되지 않는 것 확인 |
| `python/p31_sample.py` | 변수 `x = 111` · `test()` 만 있는 모듈 |
| `python/p32_main_func.py` | 모듈마다 따로인 변수 (`x` 222 / 111) |
| `python/p33_from.py` | `from p31_sample import test` |
| `python/machine/car.py` · `python/machine/tv.py` | 패키지(machine 폴더) 안의 모듈 |
| `python/machine/test/car.py` · `python/machine/test/tv.py` | 하위 폴더(machine/test) 안의 모듈 |
| `python/p41_import_module.py` | `import machine.car` → `machine.car.drive()` |
| `python/p42_from_module.py` | `from machine.car import drive` · `from machine import car, tv` |
| `python/p51_from_module.py` | `from machine.test import car, tv` (하위 폴더) |
| `ml/m06_all_estimators1_to_data.py` | iris · cancer · wine · digits × 분류 모델 전체 + `StratifiedKFold` → 데이터별 최고 모델 |
| `ml/m06_all_estimators2_to_data.py` | california · diabetes × 회귀 모델 전체 + `KFold` → 데이터별 최고 모델 |
| `ml/m07_for_parameter_tuning.py` | 이중 for 문으로 `learning_rate` × `max_depth` 36 조합 튜닝 |
| `ml/m08_GridSearchCV_00.py` | iris `GridSearchCV` + `XGBClassifier` 42 조합 × `StratifiedKFold` 5 → `best_params_` · `best_score_` |
| `ml/m08_GridSearchCV_01_digits.py` | digits `GridSearchCV` + `XGBClassifier` |
| `ml/m08_GridSearchCV_02_diabetes.py` | diabetes `GridSearchCV` + `XGBRegressor` + `KFold` (R2) |
| `rngd/f05_langchain.py` | `ChatOpenAI` 로 furiosa-llm 서버(Qwen3-4B-FP8) 호출 → `llm.invoke` |
| `rngd/f06_tps.py` | `time.perf_counter` 로 추론 시간 측정 → 초당 토큰 수 계산 |
| `rngd/f07_stream01.py` | OpenAI 클라이언트 `stream=True` → `delta.content` 스트리밍 출력 |
| `rngd/f07_stream02_lc.py` | LangChain `llm.stream` 스트리밍 출력 |
| `rngd/a_kingkeras.py` | `get_llm()` — `ChatOpenAI` 설정을 함수로 만든 모듈 |
| `rngd/f08_pkg01_invoke.py` | `from a_kingkeras import get_llm` → 값을 바꿔 `invoke` |
| `rngd/f08_pkg02_stream.py` | `get_llm()` 기본값으로 `stream` |
| `docs/Day45.md` | Day45 학습 기록 신규 작성 |
| `docs/Day44.md` | 하단 nav에 Day45 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(python/ · ml/ · rngd/ Day45) / 진행도(45일, 56%) 갱신 (수정) |

---

## 📊 실행 결과

| 파일 | 결과 |
|---|---|
| p03_do | 운전하다 · 시청하다 (import 때) → ========== → 운전하다 · 시청하다 |
| p13_do | import 된 모듈 `__name__` : `p11_car` · `p12_tv`, 직접 실행 : `__main__` |
| p21_car (직접 실행) | 운전하다 × 2 |
| p23_main_do | import 때 if 밖만 실행 → 운전하다 · 시청하다 각 1 번 |
| p32_main_func · p33_from | `x :  222` / `x :  111` |
| p41 · p42 · p51 | machine 운전하다 · machine 시청하다 |
| m06_1 (분류 4 개) | iris BaggingClassifier 0.9667 · cancer AdaBoostClassifier 0.9825 · wine NuSVC 1.0 · digits SVC 0.95 (90.14 초) |
| m06_2 (회귀 2 개) | california HistGradientBoostingRegressor 0.7947 · diabetes GammaRegressor 0.4407 (3747.97 초) |
| m07 (36 조합) | 최고 점수 0.97 · 최적 매개변수 (0.3, 4) |
| m08_00 (iris GridSearchCV) | `{'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}` · best_score 0.95 · test 0.9 (4.39 초) |
| m08_01 (digits GridSearchCV) | `{'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}` · best_score 0.961 · test 0.9472 (18.52 초) |
| m08_02 (diabetes GridSearchCV) | `{'learning_rate': 0.01, 'min_child_weight': 10}` · best_score r2 0.3763 · test r2 0.352 (8.16 초) |

- `random_state` 를 주지 않은 모델(트리 · 앙상블 등)은 실행할 때마다 최고 모델이 바뀔 수 있다

---

## 💻 핵심 개념

### import 방법 정리

```
import p01_car                     → p01_car.drive()
from p31_sample import test        → test()
import machine.car                 → machine.car.drive()
from machine.car import drive      → drive()
from machine import car, tv        → car.drive() · tv.watch()
from machine.test import car, tv   → car.drive() (하위 폴더 모듈)
```

### `__name__` 과 `if __name__ == "__main__":`

```
python p21_car.py   → __name__ = "__main__"  → if 안 실행 O
import p21_car      → __name__ = "p21_car"   → if 안 실행 X  (if 밖의 코드는 import 때 실행)
```

### 하이퍼파라미터 탐색 흐름

```
m07 : 이중 for 문   → 조합마다 fit · score(x_test) → 최고 점수 · 매개변수 저장 (test 한 번 평가)
m08 : GridSearchCV  → 파라미터 dict 리스트의 모든 조합 × 교차 검증 → best_params_ · best_score_ (refit 으로 최적 모델 재훈련)
```

### invoke vs stream

```
OpenAI     : client.chat.completions.create(...)              → response.choices[0].message.content
             client.chat.completions.create(..., stream=True) → for chunk : chunk.choices[0].delta.content
LangChain  : llm.invoke(질문)  → response.content  (다 만들어진 뒤 한 번에)
             llm.stream(질문)  → for chunk : chunk.content  (만들어지는 대로)
```

---

## 성과

- 파이썬 파일을 모듈로 만들어 다른 파일에서 `import` · `from ~ import` 로 불러 썼다
- `__name__` 값으로 직접 실행과 import 를 구분하고, `if __name__ == "__main__":` 으로 import 때 실행을 막았다
- 폴더(패키지)와 하위 폴더 안의 모듈을 불러오는 여러 방법을 비교했다
- all_estimators + 교차 검증을 분류 4 개 · 회귀 2 개 데이터셋에 돌려 데이터마다 최고 모델을 찾았다
- 하이퍼파라미터 36 조합을 for 문으로 직접 탐색해 최적 매개변수를 찾고, `GridSearchCV` 로 42 조합을 교차 검증으로 자동 탐색했다 (iris · digits · diabetes)
- RNGD furiosa-llm 서버를 LangChain `ChatOpenAI` 로 호출하고, 응답 시간 · 초당 토큰 수를 측정했다
- OpenAI 클라이언트와 LangChain 두 방식으로 스트리밍 출력을 했다
- LLM 설정을 `get_llm()` 모듈로 분리해 다른 파일에서 import 해 썼다

---

## 💡 주요 학습 포인트

1. **파이썬 파일 하나 = 모듈 하나**: `import 파일이름` (`.py` 없이)
2. **import 하면 그 파일이 한 번 실행된다**: 파일 맨 아래의 함수 호출도 같이 실행
3. **`__name__`**: 직접 실행은 `"__main__"`, import 되면 파일 이름
4. **`if __name__ == "__main__":`**: 직접 실행할 때만 돌릴 코드는 이 안에 둔다
5. **모듈마다 변수가 따로**: 같은 이름 `x` 라도 모듈마다 다른 값
6. **패키지 import**: `import machine.car` 로 불러오면 `machine.car.drive()` 전체 경로로 호출
7. **여러 데이터셋은 리스트 + `enumerate`**: `data_name[index]` 로 이름을 같이 출력
8. **분류는 `StratifiedKFold`, 회귀는 `KFold`**: 연속값 y 는 클래스 비율을 맞출 수 없다
9. **`np.argmax` 로 최고 모델 찾기**: 이름 · 점수 리스트를 같은 순서로 쌓는다
10. **for 문 튜닝 = GridSearchCV 의 원리**: 조합마다 훈련 · 평가 → 최고 점수일 때의 값 저장
11. **GridSearchCV 파라미터는 dict 의 리스트**: dict 안은 모든 조합, dict 끼리는 따로 탐색
12. **`best_score_` 는 교차 검증 평균, `model.score` 는 test 점수**: `refit=True` 로 최적 조합 모델이 남는다
13. **ChatOpenAI 는 base_url 만 바꾸면 RNGD 서버에도 붙는다**: `api_key="EMPTY"`, 답은 `response.content`
14. **`time.perf_counter()` 로 걸린 시간 측정**: `end - start`
15. **스트리밍**: OpenAI 는 `stream=True` + `delta.content`, LangChain 은 `llm.stream()` + `chunk.content`
16. **`print(..., end="", flush=True)`**: chunk 를 줄바꿈 없이 바로바로 출력
17. **자주 쓰는 설정은 함수 모듈로**: `from a_kingkeras import get_llm` → 기본값 / 원하는 값으로 호출

---

[⬅️ Day44](Day44.md) · [🏠 전체 목차](../README.md)
