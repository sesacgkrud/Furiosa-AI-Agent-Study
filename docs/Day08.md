# Day08 - 다중 분류(Multiclass Classification)와 One-Hot Encoding

**학습 기간:** 2026-09-09

### 핵심 학습 내용
- **이진 분류 -> 다중 분류(Multiclass Classification)** 로 확장 - 3개 이상의 클래스를 판별
- 다중 분류 3종 세트: 출력층 `activation='softmax'` + `loss='categorical_crossentropy'` + 출력 노드 수 = 클래스 개수
- `softmax`는 각 클래스에 속할 확률을 출력하고, 전체 확률의 합이 1이 됨
  (예: `[2.3, 0.8, -1.2]` -> softmax -> `[0.80, 0.18, 0.02]`)
- `categorical_crossentropy` - One-Hot Encoding된 정답과 softmax 예측 결과를 비교해 오차를 계산하는 손실함수
- **One-Hot Encoding** - 정답 y를 `[0, 1, 2]` 같은 숫자에서 `[[1,0,0], [0,1,0], [0,0,1]]` 형태로 변환
  - 숫자 0/1/2를 그대로 쓰면 모델이 클래스 간에 크기·순서 관계가 있다고 오해하므로 변환이 필요
  - 모델 구성이 아니라 **정답 데이터의 전처리**이므로 `#1. 데이터` 영역에서 처리
- One-Hot Encoding 3가지 방법 비교
  1. **TensorFlow**: `to_categorical(y)` - 가장 간단하지만 **0번 클래스부터 자동 생성**
  2. **pandas**: `pd.get_dummies(y, dtype=int)` / `pd.get_dummies(df['target']).values`
  3. **sklearn**: `OneHotEncoder(sparse_output=False)` - 입력이 반드시 **2차원**이어야 해서 `y.reshape(-1, 1)` 필요
- `to_categorical`의 함정: y가 1부터 시작하면(fetch_covtype: 1~7) 0번 자리가 비어 있는 **컬럼 8개**가 생성됨 -> 사용하지 않는 컬럼이 낭비됨
- `np.argmax(axis=1)` - One-Hot / softmax 확률값을 **가장 큰 값의 위치(클래스 번호)** 로 되돌리는 함수
  - `accuracy_score(y_test, y_predict)`에 넣기 전 y_test와 y_predict 둘 다 argmax 변환 필요
- `metrics=['acc']` 사용으로 `model.evaluate()` 반환값이 `[loss, acc]` 리스트 -> `result[0]`, `result[1]`로 접근
- `time.time()`으로 훈련 시작/종료 시간을 기록해 **소요 시간 측정** (대용량 데이터에서 특히 중요)
- 데이터 규모에 따른 `batch_size` 조절 - 소규모(digits)는 4, 대규모(covtype)는 32 이상으로 크게
- sklearn 다중 분류 데이터셋 4종 실습: Iris(3클래스), Wine(3클래스), Covtype(7클래스), Digits(10클래스)

### 학습 파일
- `keras23_softmax1_OneHot_iris.py`: Iris 3중 분류 (150×4). One-Hot Encoding 3가지 방법(to_categorical / pandas get_dummies / sklearn OneHotEncoder)을 모두 작성해 비교, sklearn 방식 최종 적용. 30-20-10-3 구조, patience=20, batch_size=16 (loss: 0.0652, acc: 1.0, accuracy_score: 1.0, 8초)
- `keras23_softmax2_wine.py`: Wine 3중 분류 (178×13). `to_categorical` 사용, 200-150-100-50-10-3 구조, patience=50, epochs=2000, batch_size=32 (목표 acc 0.95 이상 / loss: 0.1233, acc: 0.963, accuracy_score: 0.9630, 16초)
- `keras23_softmax3_fetch_covtype.py`: Covtype 7중 분류 (581,012×54) - 대용량 데이터. `to_categorical`이 (581012, 8)을 만드는 문제 확인, 300-200-100-100-100-8 구조, patience=20, batch_size=32 (목표 acc 0.93 이상 / loss: 0.3469, acc: 0.861, accuracy_score: 0.8612, 1463초)
- `keras23_softmax4_digits.py`: Digits 10중 분류 (1797×64) - 손글씨 숫자. `to_categorical` 사용, 50-40-35-30-25-20-15-10 구조, train_size=0.6, patience=100, epochs=3000, batch_size=4 (목표 acc 1.00 / loss: 0.1295, acc: 0.965, accuracy_score: 0.9652, 34초)
- `_data/kaggle_santander/submit/submit_0908_1642.csv`: Santander 제출 파일 재실행으로 갱신 (수정)

### 성과
- 이진 분류에서 다중 분류로 확장하여 클래스 3개 / 7개 / 10개 문제를 모두 구현
- One-Hot Encoding을 TensorFlow · pandas · sklearn 3가지 방법으로 직접 작성하고 차이를 비교
- sklearn `OneHotEncoder`가 2차원 입력을 요구하는 이유와 `reshape(-1, 1)` 처리 습득
- `to_categorical`이 0부터 시작한다는 특성 때문에 covtype에서 빈 컬럼이 생기는 문제를 직접 확인
- `np.argmax`로 확률값을 클래스 번호로 되돌려 `accuracy_score`를 계산하는 흐름 완성
- Iris에서 accuracy 1.0, Wine 0.963, Digits 0.965 달성 (목표 대비 Wine 합격)
- 58만 행 대용량 데이터를 24분(1463초) 훈련하며 시간 측정과 batch_size 조절의 중요성 체감

### 핵심 개념
```python
# ===== 이진 분류 vs 다중 분류 =====
# 이진분류 : 출력 노드 1개  / activation='sigmoid' / loss='binary_crossentropy'
# 다중분류 : 출력 노드 N개  / activation='softmax' / loss='categorical_crossentropy'

# 데이터셋 정보 및 클래스 확인
datasets = load_iris()
print(datasets.DESCR)                          # 데이터셋 설명
print(datasets.feature_names)                  # ['sepal length (cm)', ... ] 4개
x = datasets.data                              # (150, 4) 입력 데이터
y = datasets['target']                         # (150,)   정답 0 / 1 / 2
print(np.unique(y, return_counts=True))        # (array([0,1,2]), array([50,50,50]))

# ===== One-Hot Encoding - 왜 필요한가 =====
# [0, 0, 1, 1, 2]  (5,)
#        ↓
# [[1,0,0],
#  [1,0,0],
#  [0,1,0],
#  [0,1,0],
#  [0,0,1]]        (5, 3)
# 숫자 0/1/2 그대로 두면 "2가 0보다 크다"는 잘못된 관계를 모델이 학습한다

########## 방법 1. TensorFlow ##########
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)          # (150,) -> (150, 3)
# 주의: 무조건 0번 클래스부터 만든다
# fetch_covtype 의 y 는 1 ~ 7 이라서 -> (581012, 8) 0번 컬럼이 통째로 비게 됨

########## 방법 2. pandas ##########
y = pd.get_dummies(y, dtype=int)                       # 선생님 코드
# y_pandas = pd.DataFrame({'target': y})
# y = pd.get_dummies(y_pandas['target']).values        # (150, 3)

########## 방법 3. sklearn ##########
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
# y = ohe.fit_transform(y)     # ❌ 1차원(벡터)은 안 됨
y = y.reshape(-1, 1)           # ✅ (150,) -> (150, 1) 2차원으로 변환
y = ohe.fit_transform(y)       # (150, 3)

# 분할 - 다중 분류에서도 stratify 로 클래스 비율 유지
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=77, shuffle=True,
    stratify=y,
)

#2. 모델 구성 - 출력층 노드 수 = 클래스 개수, activation='softmax'
model = Sequential()
model.add(Dense(30, input_dim=4, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))   # 클래스 3개 -> 노드 3개
# softmax : [2.3, 0.8, -1.2] -> [0.80, 0.18, 0.02]  (합이 1인 확률)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience=20, restore_best_weights=True)

import time
start_time = time.time()                    # 훈련 시작 시간
model.fit(x_train, y_train, epochs=1000, batch_size=16, verbose=1,
          validation_split=0.3, callbacks=[es])
end_time = time.time()                      # 훈련 종료 시간

#4. 평가, 예측
result = model.evaluate(x_test, y_test)     # metrics 때문에 [loss, acc] 리스트
print('loss :', result[0])
print('acc  :', round(result[1], 3))

y_predict = model.predict(x_test)           # 아직 0/1/2 가 아닌 확률값

# argmax - 확률(One-Hot)을 클래스 번호로 되돌린다
y_test_arg    = np.argmax(y_test, axis=1)      # [[0,1,0]] -> [1]
y_predict_arg = np.argmax(y_predict, axis=1)   # [[0.1,0.8,0.1]] -> [1]

acc_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score :', acc_score)
print('소요 시간 :', round(end_time - start_time), '초')
```

### 💡 주요 학습 포인트
1. **다중 분류 3종 세트**: 출력 노드 = 클래스 개수 + `activation='softmax'` + `loss='categorical_crossentropy'`
2. **softmax**: 각 클래스일 확률을 출력하고 전부 더하면 1 -> 가장 확률이 높은 클래스가 정답
3. **One-Hot Encoding이 필요한 이유**: 0/1/2를 그대로 쓰면 모델이 클래스에 크기·순서가 있다고 오해한다
4. **One-Hot 3가지 방법**: `to_categorical` (간단) / `pd.get_dummies` (판다스) / `OneHotEncoder` (2차원 필수)
5. **`reshape(-1, 1)`**: sklearn `OneHotEncoder`는 2차원 입력만 받으므로 (150,) -> (150,1) 변환 필수
6. **`to_categorical`의 함정**: 0부터 시작하므로 y가 1~7인 covtype은 쓸모없는 0번 컬럼이 생겨 (N, 8)이 된다
7. **`np.argmax(axis=1)`**: One-Hot / 확률값 -> 클래스 번호. y_test와 y_predict 둘 다 변환해야 accuracy_score 계산 가능
8. **One-Hot 전처리 위치**: 모델이 아니라 정답 데이터를 바꾸는 작업이므로 `#1. 데이터` 영역에서 처리
9. **시간 측정**: `time.time()`으로 훈련 소요 시간 기록 - covtype 58만 행은 1463초(약 24분) 소요
10. **batch_size 조절**: 데이터가 크면 batch_size를 크게(32 이상), 작은 데이터는 작게(4) - 작게 주면 대용량에서 훈련이 끝나지 않는다

---

[⬅️ Day07](Day07.md) · [🏠 전체 목차](../README.md)
