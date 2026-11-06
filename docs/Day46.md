# Day46 - RandomizedSearchCV 하이퍼파라미터 탐색 & KerasRegressor 신경망 하이퍼파라미터 튜닝

**학습 기간:** 2026-11-06

> Day45 의 `GridSearchCV` (42 조합 전부 탐색)를 `RandomizedSearchCV` 로 바꿔, **조합 중 일부(`n_iter`, 기본값 10 개)만 무작위로 뽑아** 교차 검증했다. iris · digits · diabetes 세 데이터에 적용해 GridSearchCV 와 점수 · 시간을 비교했다(m09).
>
> 이어서 keras 신경망을 `build_model` 함수로 만들고 `KerasRegressor` 로 감싸서 sklearn 모델처럼 `RandomizedSearchCV` 에 넣었다. batch_size · optimizer · dropout · activation · 노드 수를 하이퍼파라미터로 정해 diabetes 회귀에 튜닝했다(keras77, 작성 중).

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| RandomizedSearchCV | `param_distributions` 의 조합 중 `n_iter` 개만 무작위로 뽑아 교차 검증 |
| param_grid → param_distributions | GridSearchCV 는 `param_grid`, RandomizedSearchCV 는 `param_distributions` (잘못 쓰면 TypeError) |
| n_iter · random_state | 뽑을 조합 개수 (기본값 10) · 뽑히는 조합 고정 |
| 훈련 횟수 비교 | GridSearchCV 42 × 5 = 210 번 → RandomizedSearchCV 10 × 5 = 50 번 |
| 회귀는 stratify 불가 | 연속값 y 에 `stratify=y` → ValueError, `KFold` 사용 |
| build_model | 하이퍼파라미터를 인자로 받아 compile 한 keras 모델을 돌려주는 함수 |
| KerasRegressor | `KerasRegressor(build_fn=build_model)` → keras 모델을 sklearn 모델처럼 사용 |
| 신경망 하이퍼파라미터 | `batch_size` · `optimizer` · `drop` · `activation` · `node1 ~ node5` 를 dict 로 탐색 |

---

## 📖 핵심 학습 내용

### 1. RandomizedSearchCV (m09_RandomSearchCV_00)

Day45 의 m08_00 (iris + XGBClassifier + GridSearchCV) 을 베이스로 탐색 방법만 바꿨다.

```python
from sklearn.model_selection import RandomizedSearchCV

parameters = [                                                                                          # 총 42번
    {"n_estimators" : [100, 500], "max_depth" : [6, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},     # 18번
    {"max_depth" : [6, 8, 10, 12], "learning_rate" : [0.1, 0.01, 0.001]},                               # 12번
    {"min_child_weight" : [2, 3, 5, 10], "learning_rate" : [0.1, 0.01, 0.001]},                         # 12번
]

#2. 모델 구성
xgb = XGBClassifier()

model = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=parameters,     # 탐색할 하이퍼파라미터 조합 (이 중 일부만 무작위로 뽑는다)
    cv=kfold,
    verbose=1,
    n_jobs=-1,
    refit=True,
    random_state=333,                   # 무작위로 뽑는 조합을 실행마다 같게 고정
    # n_iter=3                          # 뽑을 조합 개수 (기본값 10)
)
```

```
Fitting 5 folds for each of 10 candidates, totalling 50 fits
model.score :  0.9
best_score :  0.9333333333333332
걸린시간 :  3.89 초
```

- GridSearchCV 의 `param_grid` 를 그대로 쓰면 `TypeError: RandomizedSearchCV.__init__() got an unexpected keyword argument 'param_grid'` → RandomizedSearchCV 는 같은 역할의 인자 이름이 **`param_distributions`**
- 42 개 조합 중 `n_iter` (기본값 10) 개만 뽑아서 10 × 5 fold = 50 번 훈련 (GridSearchCV 는 210 번)
- 뽑힌 10 개 중 최적 : `{'min_child_weight': 2, 'learning_rate': 0.1}`
- best_score 0.9333 으로 GridSearchCV 의 0.95 보다 조금 낮다 → GridSearchCV 의 최적 조합(n_estimators=500 · max_depth=6)이 무작위 10 개에 뽑히지 않았다

### 2. RandomizedSearchCV 데이터셋 적용 (m09_01_digits · m09_02_diabetes)

| 파일 | 베이스 | 바뀐 점 |
|---|---|---|
| m09_01 (digits) | m09_00 | `load_iris` → `load_digits` (나머지 같음) |
| m09_02 (diabetes) | m09_00 | `XGBRegressor`, `stratify` 제거, `StratifiedKFold` → `KFold`, 점수는 R2 |

```python
# m09_02 - 회귀
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
)

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

xgb = XGBRegressor()
```

- diabetes 에 `stratify=y` 를 두면 y 값을 클래스로 보는데, 1 개뿐인 값이 많아서 `ValueError: The least populated classes in y have only 1 member` → 회귀는 `stratify` 를 빼고 `KFold`

| 데이터 | 탐색 | 최적 파라미터 | best_score | model.score | 훈련 횟수 | 걸린 시간 |
|---|---|---|:---:|:---:|:---:|:---:|
| iris | Grid (m08_00) | `learning_rate=0.1, max_depth=6, n_estimators=500` | 0.95 | 0.9 | 210 | 4.39 초 |
| iris | Random (m09_00) | `min_child_weight=2, learning_rate=0.1` | 0.9333 | 0.9 | 50 | 3.89 초 |
| digits | Grid (m08_01) | `learning_rate=0.1, max_depth=6, n_estimators=500` | 0.961 | 0.9472 | 210 | 18.52 초 |
| digits | Random (m09_01) | `n_estimators=100, max_depth=12, learning_rate=0.1` | 0.9589 | 0.9417 | 50 | 7.5 초 |
| diabetes | Grid (m08_02) | `learning_rate=0.01, min_child_weight=10` | r2 0.3763 | r2 0.352 | 210 | 8.16 초 |
| diabetes | Random (m09_02) | `min_child_weight=10, learning_rate=0.1` | r2 0.3595 | r2 0.3154 | 50 | 5.41 초 |

- RandomizedSearchCV 는 훈련 횟수가 1/4 이하로 줄어 더 빠르고, 점수는 GridSearchCV 와 비슷하거나 조금 낮다
- digits 는 시간이 18.52 초 → 7.5 초로 줄면서 best_score 는 0.961 → 0.9589 로 거의 같다

### 3. KerasRegressor + RandomizedSearchCV 신경망 튜닝 (keras77_hyperParameter02_diabets, 작성 중)

keras11_2_diabetes 를 베이스로, keras 모델을 함수로 만들어 sklearn 의 RandomizedSearchCV 로 하이퍼파라미터를 탐색한다.

```python
#2. 모델 구성
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(10,), name='inputs')

    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model

def create_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    ...     # node2 ~ node5 도 [128, 64, 32, 16]
    return {
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        ...
    }
hyperparameters = create_hyperparameter()

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1,)

model = RandomizedSearchCV(keras_model, hyperparameters,
                           cv=3,
                           random_state=333,
                           n_iter=10,
                           verbose=1,
                           refit=True,
                        )

# 3. 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=8, verbose=1, factor=0.8)

model.fit(x_train, y_train, epochs=1000, verbose=1, callbacks=[es, rlr],
          validation_split=0.1)
```

- `build_model(...)` : 하이퍼파라미터를 인자로 받아 함수형 모델(`Input` → `Dense` · `Dropout` → `Model`)을 만들고 `compile` 까지 해서 돌려준다
- `KerasRegressor(build_fn=build_model)` : keras 모델을 sklearn 회귀 모델처럼 감싸서 `RandomizedSearchCV` 에 넣을 수 있게 한다
- `create_hyperparameter()` 의 dict 키는 `build_model` 의 인자 이름(`drop` · `optimizer` · `activation` · `node1 ~ node5`)과 fit 인자(`batch_size`)에 맞춘다
- RandomizedSearchCV 의 두 번째 인자로 dict 를 넘겼다 (`param_distributions` 자리)
- `n_iter=10` · `cv=3` → 10 조합 × 3 fold = 30 번 훈련 + refit
- `model.fit` 에 준 `epochs` · `callbacks` (EarlyStopping · ReduceLROnPlateau) · `validation_split` 은 조합마다 keras 모델의 fit 으로 그대로 전달된다

```
model.score :  -26464.02734375
best_score :  -29698.218098958332
걸린시간 :  7.64 초
```

- KerasRegressor 의 score 는 loss(mse) 에 마이너스를 붙인 값 → 클수록(0 에 가까울수록) 좋다 (sklearn 은 점수가 높은 조합을 고르기 때문)
- 아직 작성 중인 파일 : `build_model` 의 `lr` 인자는 정의만 되어 있고 모델 안에서 쓰이지 않는다, 최적 파라미터 출력 결과는 기록하지 않았다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `ml/m09_RandomSearchCV_00.py` | iris `RandomizedSearchCV` + `XGBClassifier` → 42 조합 중 10 개 × `StratifiedKFold` 5 |
| `ml/m09_RandomSearchCV_01_digits.py` | digits `RandomizedSearchCV` + `XGBClassifier` |
| `ml/m09_RandomSearchCV_02_diabetes.py` | diabetes `RandomizedSearchCV` + `XGBRegressor` + `KFold` (R2) |
| `keras2/keras77_hyperParameter02_diabets.py` | `build_model` + `KerasRegressor` + `RandomizedSearchCV` 로 신경망 하이퍼파라미터 탐색 (작성 중) |
| `docs/Day46.md` | Day46 학습 기록 신규 작성 |
| `docs/Day45.md` | 하단 nav에 Day46 링크 추가 (수정) |
| `README.md` | 학습 일지 / 디렉토리 구조(keras2/ · ml/ Day46) / 진행도(46일, 58%) 갱신 (수정) |

---

## 📊 실행 결과

| 파일 | 결과 |
|---|---|
| m09_00 (iris) | 50 fits · `{'min_child_weight': 2, 'learning_rate': 0.1}` · best_score 0.9333 · test 0.9 (3.89 초) |
| m09_01 (digits) | 50 fits · `{'n_estimators': 100, 'max_depth': 12, 'learning_rate': 0.1}` · best_score 0.9589 · test 0.9417 (7.5 초) |
| m09_02 (diabetes) | 50 fits · `{'min_child_weight': 10, 'learning_rate': 0.1}` · best_score r2 0.3595 · test r2 0.3154 (5.41 초) |
| keras77 (diabetes) | model.score -26464.03 · best_score -29698.22 (score = -mse) · 7.64 초 |

---

## 💻 핵심 개념

### GridSearchCV vs RandomizedSearchCV

```
GridSearchCV(estimator, param_grid=parameters, cv=kfold)
  → 모든 조합 (42) × fold (5) = 210 번 훈련           : 빠짐없이 찾지만 오래 걸린다
RandomizedSearchCV(estimator, param_distributions=parameters, cv=kfold, n_iter=10, random_state=333)
  → 무작위 n_iter 조합 (10) × fold (5) = 50 번 훈련    : 빠르지만 최적 조합을 놓칠 수 있다
```

### keras 모델을 sklearn 탐색에 넣는 흐름

```
build_model(하이퍼파라미터 인자)  → compile 된 keras 모델 반환
KerasRegressor(build_fn=build_model)  → sklearn 모델처럼 감싸기
RandomizedSearchCV(keras_model, {인자 이름 : 후보 리스트}, cv=3, n_iter=10)
model.fit(x_train, y_train, epochs=..., callbacks=[...], validation_split=...)  → 조합마다 keras fit
```

---

## 성과

- GridSearchCV 를 RandomizedSearchCV 로 바꿔 iris · digits · diabetes 에서 점수 · 시간을 비교했다
- `param_grid` / `param_distributions` 인자 이름 차이로 나는 TypeError 와, 회귀 데이터의 `stratify` ValueError 를 확인하고 고쳤다
- keras 신경망을 `build_model` 함수 + `KerasRegressor` 로 감싸 RandomizedSearchCV 로 하이퍼파라미터를 탐색했다

---

## 💡 주요 학습 포인트

1. **RandomizedSearchCV 는 일부 조합만 탐색**: `n_iter` (기본값 10) 개 × fold 수만큼 훈련
2. **인자 이름이 다르다**: GridSearchCV 는 `param_grid`, RandomizedSearchCV 는 `param_distributions`
3. **`random_state` 로 뽑히는 조합 고정**: 실행마다 같은 결과
4. **속도 vs 정확도**: Random 은 훨씬 빠르지만 최적 조합이 안 뽑히면 점수가 조금 낮다
5. **회귀는 stratify · StratifiedKFold 불가**: 연속값 y → `KFold`
6. **keras 모델도 sklearn 탐색 가능**: `build_model` 함수 → `KerasRegressor(build_fn=...)`
7. **dict 키 = build_model 인자 이름 · fit 인자**: `drop` · `optimizer` · `node1` · `batch_size` 등
8. **KerasRegressor 의 score 는 -loss**: 0 에 가까울수록 좋다

---

[⬅️ Day45](Day45.md) · [🏠 전체 목차](../README.md)
