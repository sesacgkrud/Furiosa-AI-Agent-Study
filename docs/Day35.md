# Day35 - LangGraph 병렬 결과 LLM 통합 (merge) & 머신러닝 트리 모델 feature_importances_ · XGBoost F score / Gain score

**학습 기간:** 2026-10-22

> Day34의 병렬 실행 그래프(search · analysis → merge)에서 merge 노드가 두 결과를 `=====`로 **단순히 이어 붙이던(concat)** 것을, merge 노드에서도 **LLM을 한 번 더 불러 두 결과를 종합한 최종 답변**을 만들도록 바꿨다. Day34의 `lg09_Parallel_Execution_absorption2.py`는 `lg09_Parallel_Execution_absorption2_concat.py`로 이름을 바꿨다. 이어서 새 `ml/` 폴더에서 머신러닝 트리 계열 모델 4종(DecisionTree · RandomForest · GradientBoosting · XGBoost)을 iris 분류 · diabetes 회귀에 적용하고, `model.score`와 `feature_importances_`로 **컬럼 중요도**를 확인했다. 마지막으로 XGBoost의 중요도 계산 방식인 **F score(weight, 빈도수)** 와 **Gain score(gain, 손실함수 개선 정도)** 를 `get_booster().get_score`와 `plot_importance` 그래프로 비교했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| concat merge | 두 결과를 `'====='`로 이어 붙여 `('assistant', result)`로 추가 (absorption2_concat) |
| LLM merge | merge 노드에서 `llm.invoke(prompt)` → 검색 + 분석을 종합한 하나의 답변 (absorption3) |
| 트리 모델 4종 | `DecisionTree` · `RandomForest` · `GradientBoosting` · `XGB` (Classifier / Regressor) |
| 머신러닝 훈련 | compile · epochs 없이 `model.fit(x_train, y_train)` 한 번 |
| model.score | Classifier → accuracy, Regressor → R2 |
| feature_importances_ | 컬럼별 중요도, 모두 합하면 1, 값이 클수록 중요한 컬럼 |
| F score | `importance_type='weight'` : 컬럼이 분기에 쓰인 횟수 (빈도수) |
| Gain score | `get_booster().get_score(importance_type='gain')` : 분기로 손실함수가 개선된 정도 |
| plot_importance | `from xgboost import plot_importance` → 중요도 막대 그래프 |

---

## 📖 핵심 학습 내용

### 1. merge 노드에서 LLM으로 결과 통합 (lg09_Parallel_Execution_absorption3)

lg09_Parallel_Execution_absorption2_concat을 베이스로, 그래프 구조와 search · analysis 노드는 그대로 두고 **merge 노드만** 바꿨다.

```python
# 결과 통합 에이전트 (단순 concat 보다 강력한 merge)
def merge_agent(state : State):
    print('결과 통합 에이전트 실행')

    prompt = f'''
    다음 두 결과를 종합하여 최종 답변을 작성해줘.

    [웹 검색 결과]
    {state["search_result"]}

    [AI 분석 결과]
    {state["analysis_result"]}

    검색 결과를 우선적으로 참고하고,
    최신 정보와 기존 분석을 구분해서 설명해줘.

    '''

    response = llm.invoke(prompt)

    return {'messages' : [response]}
```

- concat 버전은 검색 결과 딕셔너리(`{'query': ..., 'results': [...]}`)가 그대로 보이고 그 아래 LLM 분석이 붙었다
- LLM merge 버전은 `[웹 검색 결과]` / `[AI 분석 결과]`를 프롬프트에 나눠 넣고 **검색 결과를 우선 참고**하라고 지시 → 하나의 정리된 답변
- `llm.invoke`가 돌려준 `AIMessage`를 그대로 `messages`에 넣는다 → `add_messages`가 AI 답변으로 추가 (concat 때의 `('assistant', result)`와 같은 역할)
- LLM 호출이 analysis · merge 두 번이 된다

```
분석 에이전트 실행
검색 에이전트 실행
결과 통합 에이전트 실행
Assistant : ## 종합 결론
엔비디아의 미래는 여전히 AI 인프라 확장의 최대 수혜 기업 중 하나라는 점에서 긍정적입니다. ...
## 1. 검색 결과 기준의 최신 흐름        ← 기사 · 출처 URL
## 2. 장기적으로 긍정적인 이유
## 3. 기존 분석 관점에서의 주요 위험    ← 기존 지식 분석
```

### 2. absorption2 파일 이름 변경

- `lg09_Parallel_Execution_absorption2.py` → `lg09_Parallel_Execution_absorption2_concat.py` (단순 concat 버전, LLM merge 버전은 absorption3)
- merge 함수에 `# 단순 concat` 주석 추가
- `docs/Day34.md`와 README 디렉토리 구조의 파일 이름도 함께 바꿨다

### 3. 트리 모델 4종 분류 · feature_importances_ (m01_feature_importances01)

iris(150, 4)를 트리 계열 분류 모델 4개로 for 문을 돌려 훈련 · 평가했다.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

x, y = load_iris(return_X_y=True)

model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]

for model in models:
    model.fit(x_train, y_train)
    print('====================', model.__class__.__name__, '====================')

    score = model.score(x_test, y_test)     # Classifier : accuracy / Regressor : R2
    print('acc: ', score)
    print(model.feature_importances_)
```

- `load_iris(return_X_y=True)` : x, y를 바로 나눠 받는다
- 머신러닝 모델은 compile · epochs 없이 `fit` 한 번으로 훈련, `evaluate` 대신 `score`
- `model.__class__.__name__` : 모델 클래스 이름 문자열
- `feature_importances_` : 컬럼 4개의 중요도, 모두 합하면 1 → 4개 모델 모두 3번째 컬럼(petal length)이 가장 높다
- XGBClassifier는 `random_state` 123 → 12313으로 바꾸면 acc 1.0

| 모델 | acc | feature_importances_ |
|---|:---:|---|
| DecisionTreeClassifier | 0.8333 | `[0.0125 0.03 0.9213 0.0362]` |
| RandomForestClassifier | 0.9333 | `[0.0886 0.0215 0.4872 0.4027]` |
| GradientBoostingClassifier | 0.9667 | `[0.0016 0.0215 0.8248 0.1521]` |
| XGBClassifier | 0.9333 | `[0.0243 0.0247 0.7377 0.2133]` |

### 4. 트리 모델 4종 회귀 (m01_feature_importances02)

m01_01을 베이스로 diabetes(442, 10) 회귀에 `Regressor` 4종을 적용했다.

```python
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=random_state,
)                                           # 회귀라서 stratify 제외

score = model.score(x_test, y_test)         # Regressor : R2
print('r2: ', score)
```

| 모델 | r2 |
|---|:---:|
| DecisionTreeRegressor | 0.1580 |
| RandomForestRegressor | 0.5266 |
| GradientBoostingRegressor | 0.5585 |
| XGBRegressor | 0.3907 |

- 컬럼 10개 → 중요도 10개, 모델 모두 3번째(bmi) · 9번째(s5) 컬럼 중요도가 높다

### 5. XGBoost F score / Gain score (m01_feature_importances03_gain)

XGBClassifier 하나만 남겨 중요도 계산 방식을 비교했다.

```python
gain_score = model.get_booster().get_score(importance_type='gain')
print('Gain score: ', gain_score)
```

- **F score (weight)** : 그 컬럼이 트리 분기에 쓰인 **횟수 (빈도수)**
- **Gain score (gain)** : 그 컬럼으로 분기했을 때 **손실함수가 개선된 정도** → 높으면 개선이 많이 된 것
- `get_booster()` : sklearn 형식 모델 안의 xgboost 원래 모델(Booster)을 꺼낸다
- `get_score(...)` : `{'f0': ..., 'f1': ...}` 딕셔너리 (f0 = 0번째 컬럼)
- XGBoost의 `feature_importances_` = **Gain score를 합이 1이 되게 나눈 값** (f0 : 0.0928 / 3.8183 = 0.0243)

```
acc:  0.9333333333333333
feature_importances_:  [0.02430454 0.02472077 0.7376847  0.21328993]
Gain score:  {'f0': 0.09280245006084442, 'f1': 0.09439174085855484, 'f2': 2.8167145252227783, 'f3': 0.8144086599349976}
```

### 6. plot_importance 그래프 (m01_feature_importances04_graph)

m01_03을 베이스로 중요도를 막대 그래프로 그렸다.

```python
import matplotlib.pyplot as plt
from xgboost import plot_importance

# F-Score Graph
# plot_importance(model, importance_type='weight')
# plt.title('F-Score')
# plt.show()

# Gain-Score Graph
plot_importance(model, importance_type='gain')
plt.title('Gain-Score')
plt.show()
```

| 컬럼 | F score (weight) | Gain score (gain) |
|:---:|:---:|:---:|
| f0 (sepal length) | 63 | 0.0928 |
| f1 (sepal width) | 66 | 0.0944 |
| f2 (petal length) | 145 | 2.8167 |
| f3 (petal width) | 83 | 0.8144 |

- 빈도수(F score)로 보면 f0 · f1도 꽤 쓰이지만, Gain으로 보면 f2의 기여가 압도적이다
- 막대 끝에 값이 표시되고 값이 큰 컬럼이 위로 정렬된다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `LangGraph/lg09_Parallel_Execution_absorption3.py` | merge 노드에서 `llm.invoke`로 검색 결과 + 분석 결과를 종합한 최종 답변 |
| `LangGraph/lg09_Parallel_Execution_absorption2_concat.py` | `lg09_Parallel_Execution_absorption2.py`에서 이름 변경, `# 단순 concat` 주석 추가 (수정) |
| `ml/m01_feature_importances01.py` | iris 분류 트리 모델 4종 for 문 → `score`(acc) · `feature_importances_` |
| `ml/m01_feature_importances02.py` | diabetes 회귀 Regressor 4종 → `score`(R2) · `feature_importances_` |
| `ml/m01_feature_importances03_gain.py` | XGBClassifier `feature_importances_` vs `get_booster().get_score(importance_type='gain')` |
| `ml/m01_feature_importances04_graph.py` | `plot_importance`로 F score(weight) · Gain score(gain) 막대 그래프 |
| `docs/Day35.md` | Day35 학습 기록 신규 작성 |
| `docs/Day34.md` | absorption2 파일 이름 변경 반영, 하단 nav에 Day35 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(Day34 lg09 이름, Day35, ml/) / 진행도(35일, 44%) 갱신 (수정) |

---

## 📊 실행 결과

| 파일 | 결과 |
|---|---|
| lg09_absorption3 | `## 종합 결론` → 검색 결과 기준 최신 흐름 → 긍정 요인 → 위험 요인 순서로 정리된 하나의 답변 |
| m01_01 (iris) | acc : DT 0.8333 / RF 0.9333 / GB 0.9667 / XGB 0.9333 |
| m01_02 (diabetes) | r2 : DT 0.1580 / RF 0.5266 / GB 0.5585 / XGB 0.3907 |
| m01_03 | Gain score `f2 2.8167` 가 가장 크다, `feature_importances_` = Gain 정규화 값 |
| m01_04 | F score `f2 145 / f3 83 / f1 66 / f0 63`, Gain `f2 > f3 > f1 > f0` 막대 그래프 |

---

## 💻 핵심 개념

### concat merge vs LLM merge

```
concat (absorption2_concat)                 LLM merge (absorption3)
search_result + '=====' + analysis_result    llm.invoke( [웹 검색 결과] + [AI 분석 결과] + 지시 )
→ ('assistant', result)                      → AIMessage 그대로
검색 결과 딕셔너리가 그대로 보인다              하나의 정리된 답변 (LLM 호출 1번 추가)
```

### 머신러닝 모델 흐름

```
model = XGBClassifier(random_state=...)     # 모델 구성
model.fit(x_train, y_train)                 # 훈련 (compile · epochs 없음)
model.score(x_test, y_test)                 # Classifier → accuracy / Regressor → R2
model.feature_importances_                  # 컬럼별 중요도 (합 1)
```

### XGBoost 중요도

```
importance_type='weight' → F score    : 분기에 쓰인 횟수 (빈도수)
importance_type='gain'   → Gain score : 분기로 손실함수가 개선된 정도

model.get_booster().get_score(importance_type='gain')   → {'f0': ..., 'f1': ...}
model.feature_importances_                               → Gain 을 합 1 로 나눈 값
plot_importance(model, importance_type=...)              → 막대 그래프
```

---

## 성과

- 병렬 실행 그래프의 merge 노드를 LLM 통합으로 바꿔, 웹 검색 결과와 기존 지식 분석을 하나의 정리된 답변으로 만들었다
- 트리 계열 모델 4종을 for 문 하나로 분류(iris) · 회귀(diabetes)에 적용하고 성능을 비교했다
- `feature_importances_`로 데이터에서 어떤 컬럼이 중요한지 확인했다
- XGBoost의 F score와 Gain score를 숫자와 그래프로 비교했다

---

## 💡 주요 학습 포인트

1. **merge 노드도 LLM을 부를 수 있다**: 두 결과를 프롬프트에 나눠 넣고 `llm.invoke` → 하나의 답변
2. **`llm.invoke` 결과는 그대로 messages에**: `return {'messages' : [response]}` → `add_messages`가 AIMessage로 추가
3. **머신러닝은 `fit` 한 번**: compile · epochs · batch_size 없이 훈련한다
4. **`score`는 모델 종류에 따라 다르다**: Classifier → accuracy, Regressor → R2
5. **회귀에는 stratify를 쓰지 않는다**: y가 연속값이라 클래스 비율을 맞출 수 없다
6. **`feature_importances_`는 합이 1**: 값이 클수록 예측에 많이 기여한 컬럼
7. **F score = 빈도수, Gain = 개선 정도**: 자주 쓰인 컬럼이 꼭 많이 개선한 컬럼은 아니다
8. **XGBoost `feature_importances_`는 Gain 기준**: `get_score(importance_type='gain')`을 합 1로 나눈 값
9. **`plot_importance`로 바로 그래프**: `importance_type`으로 weight / gain을 고른다

---

[⬅️ Day34](Day34.md) · [🏠 전체 목차](../README.md) · [Day36 ➡️](Day36.md)
