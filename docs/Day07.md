# Day07 - EarlyStopping 확대 적용 및 이진 분류(Binary Classification) 입문

**학습 기간:** 2026-09-08

---

## 핵심 학습 내용
- EarlyStopping을 California 외 4개 데이터셋(Diabetes, Boston, 따릉이, Kaggle Bike)으로 확대 적용
- `patience=20`, `restore_best_weights=True` 조합으로 최적 가중치 복원 실습
- EarlyStopping 적용 전(keras19 과적합 시각화) / 후(keras20)의 loss·val_loss 값을 파일 하단에 주석으로 기록해 비교
- **회귀(Regression)와 분류(Classification)의 차이** 이해 - 연속값 예측 vs 0/1 판별
- 이진 분류 3종 세트: 출력층 `activation='sigmoid'` + `loss='binary_crossentropy'` + `metrics=['acc']`
- sigmoid 함수는 결과를 0~1 사이 확률로 압축 -> 0.5 기준으로 반올림해 0 또는 1로 판정
- `binary_crossentropy`(이진 교차 엔트로피)를 이진 분류 손실함수로 사용 (회귀의 mse 대신)
- `metrics`는 **훈련(가중치 갱신)에 관여하지 않는 참고용 지표** - `['acc']`와 `['accuracy']`는 동일
- `metrics` 추가 시 `model.evaluate()`의 반환값이 스칼라가 아닌 **리스트 `[loss, acc]`** 로 바뀜
- 데이터 불균형(Imbalanced Data) 확인 방법 3가지: `np.unique(y, return_counts=True)`, `pd.DataFrame(y).value_counts()`, `pd.Series(y).value_counts()`
- `train_test_split(stratify=y)` - y의 클래스 비율을 train/test에 동일하게 유지해서 분할
- `accuracy_score(y_test, y_predict)` 사용 시 `np.round()`로 0/1 변환 필수
  (변환 없이 넣으면 `ValueError: Classification metrics can't handle a mix of binary and continuous targets`)
- `datasets.DESCR`, `datasets.feature_names`로 데이터셋 설명 및 컬럼명 확인
- sklearn 유방암 데이터셋 `load_breast_cancer()` 활용 (569 샘플, 30개 특성, 0:212 / 1:357)
- Kaggle Santander 대규모 이진 분류 데이터 처리 (200,000 샘플, 200개 특성, 0:179,902 / 1:20,098)
- 상대경로(`./_data/...`)와 절대경로(`c:/furiosa_study/_data/...`) 차이 및 사용 구분
- 분류 문제의 제출 파일(submission.csv) 생성 방법

---

## 학습 파일

| 파일 | 내용 |
|---|---|
| `keras19_overfit2_diabetes.py` | 실행 결과 기록 추가 (loss: 2066.94 / val_loss: 3627.65, 평가 loss: 3002.05) (수정) |
| `keras19_overfit3_boston.py` | 실행 결과 기록 추가 (loss: 5.47 / val_loss: 19.26, 평가 loss: 23.07) (수정) |
| `keras19_overfit4_dacon_ddareung.py` | 실행 결과 기록 추가 (loss: 1947.17 / val_loss: 2505.24, 평가 loss: 3137.05) (수정) |
| `keras19_overfit5_kaggle_bike.py` | 실행 결과 기록 추가 (loss: 21725.20 / val_loss: 22583.54, 평가 loss: 21853.41) (수정) |
| `keras20_EarlyStopping2_diabetes.py` | Diabetes + EarlyStopping(patience=20) + `validation_split=0.3` (평가 loss: 2874.47) |
| `keras20_EarlyStopping3_boston.py` | Boston Housing(Keras 내장) + EarlyStopping(patience=20) (평가 loss: 22.82) |
| `keras20_EarlyStopping4_dacon_ddareung.py` | 따릉이 + EarlyStopping + `validation_data`(방법 1) + r2/mse/RMSE (평가 loss: 3192.95) |
| `keras20_EarlyStopping5_kaggle_bike.py` | Kaggle Bike + EarlyStopping + `validation_split`(방법 2) + r2/mse/RMSE (평가 loss: 21894.74) |
| `keras21_sigmoid_metrics_cancer.py` | 유방암 데이터 이진 분류 (sigmoid, binary_crossentropy, metrics=['acc'], stratify=y, accuracy_score / loss: 0.1793, acc: 0.9357, acc_score: 0.9240) |
| `keras22_sigmoid_santander.py` | Kaggle Santander 고객 거래 예측 이진 분류 (200,000×200, 500-250-125-60-30-1 구조, stratify=y, 제출 파일 생성) |

---

## 성과
- EarlyStopping을 5개 데이터셋 전체에 적용 완료하고, 적용 전후 loss 값을 직접 비교
- 회귀에서 분류로 넘어가는 전환점 - 이진 분류 모델을 처음부터 끝까지 구현
- sigmoid / binary_crossentropy / metrics 3요소의 역할을 각각 구분해서 이해
- 클래스 불균형 데이터를 `stratify=y`로 안전하게 분할하는 방법 습득
- `accuracy_score` 사용 시 발생하는 타입 에러의 원인을 직접 겪고 `np.round()`로 해결
- 200,000행 200컬럼의 대규모 분류 데이터셋을 처리하고 Kaggle 제출 파일까지 생성

---

## 핵심 개념

```python
# EarlyStopping 확대 적용 (patience=20)
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,                 # 20 epoch 동안 개선 없으면 중단
    restore_best_weights=True,   # val_loss 최저 시점의 가중치로 복원
)
model.fit(x_train, y_train, epochs=500, batch_size=16,
          validation_split=0.3, callbacks=[es])
```

### 회귀 vs 이진 분류

```python
# 회귀    : 출력층 activation 없음       / loss='mse'
# 이진분류 : 출력층 activation='sigmoid' / loss='binary_crossentropy'

#2. 모델 구성 - 출력층에 무조건 sigmoid
model.add(Dense(32, input_dim=30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 결과를 0 ~ 1 사이 확률로 압축

#3. 컴파일 - 이진 분류 손실함수 + 참고 지표
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])      # metrics=['accuracy'] 와 동일
# metrics 는 훈련(가중치 갱신)에 영향을 주지 않는 "구경용" 지표

#4. 평가 - metrics 를 넣으면 반환값이 리스트가 된다
loss = model.evaluate(x_test, y_test)
print(loss)                        # [0.1793, 0.9357] -> [loss, acc]
print("loss :", loss[0])
print("acc  :", round(loss[1], 4))
```

### 클래스 개수 확인 (데이터 불균형 체크) - 3가지 방법

```python
print(np.unique(y, return_counts=True))   # (array([0, 1]), array([212, 357]))
print(pd.DataFrame(y).value_counts())     # 1: 357 / 0: 212
print(pd.Series(y).value_counts())        # 1: 357 / 0: 212
```

### stratify - y의 클래스 비율을 그대로 유지하며 분할

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42,
    stratify=y,     # 불균형 데이터에서 한쪽 클래스가 쏠리는 것을 방지
)
```

### accuracy_score - 예측값을 반드시 0/1로 반올림해야 한다

```python
from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)   # [[0.976], [0.099], ...] 0~1 실수
y_predict = np.round(y_predict)     # [[1.], [0.], ...]      0 또는 1
acc_score = accuracy_score(y_test, y_predict)
# np.round 를 빼면 ->
# ValueError: Classification metrics can't handle a mix of binary and continuous targets
```

### 데이터셋 정보 확인

```python
print(datasets.DESCR)           # 데이터셋 설명 (describe)
print(datasets.feature_names)   # 컬럼명 (유방암 데이터는 30개)
```

### 경로 지정 - 상대경로 vs 절대경로

```python
# path = './_data/kaggle_santander/'              # 상대경로 (실행 위치 기준)
path = 'c:/furiosa_study/_data/kaggle_santander/' # 절대경로 (항상 동일)
```

### 분류 문제 제출 파일 생성

```python
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['target'] = model.predict(test_csv)
submission.to_csv(path + 'submit/' + 'submit_0908_1642.csv')
```


---

## 💡 주요 학습 포인트
1. **EarlyStopping 실전화**: patience=20으로 5개 데이터셋 전체에 적용, 적용 전/후 loss를 주석으로 남겨 비교
2. **회귀 vs 분류**: 회귀는 연속값을 맞히고, 분류는 0이냐 1이냐를 판별한다
3. **이진 분류 3종 세트**: 출력층 sigmoid + loss='binary_crossentropy' + metrics=['acc']
4. **sigmoid**: 결과를 0~1 확률로 압축 -> 0.5 기준 반올림해서 0 또는 1로 결론
5. **metrics의 정체**: 훈련에 관여하지 않는 참고 지표. 넣는 순간 evaluate 반환값이 [loss, acc] 리스트가 된다
6. **데이터 불균형 확인**: np.unique(return_counts=True) / pd.DataFrame().value_counts() / pd.Series().value_counts()
7. **stratify=y**: 클래스 비율을 유지한 채 분할 -> 불균형 데이터에서 한쪽으로 쏠리는 사고 방지
8. **accuracy_score의 함정**: np.round()로 0/1 변환하지 않으면 binary + continuous 혼합 ValueError 발생
9. **대규모 분류 데이터**: Santander 200,000행 × 200컬럼, 0:179,902 / 1:20,098의 심한 불균형
10. **경로 지정**: 상대경로는 실행 위치에 따라 깨질 수 있고, 절대경로는 항상 동일하게 동작

---

[⬅️ Day06](Day06.md) · [🏠 전체 목차](../README.md) · [Day08 ➡️](Day08.md)
