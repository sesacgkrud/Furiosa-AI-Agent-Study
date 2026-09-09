# Furiosa AI Agent - 학습 진행 현황

AI 개발자 양성 과정 학습 기록입니다. 각 Day를 클릭하면 그날의 상세 학습 내용(핵심 개념, 학습 파일, 코드 예제)으로 이동합니다.

---

## 📚 학습 일지

| Day | 학습 내용 | 날짜 | 핵심 키워드 |
|:---:|---------|:---:|-----------|
| **[Day01](docs/Day01.md)** | Keras 기초 | 2026-08-31 | Sequential, Dense, Batch Processing, Deep Learning |
| **[Day02](docs/Day02.md)** | 다중 입력/출력 | 2026-09-01 | Matrix, Transpose, Multi-Input/Output, Train/Test 분할 |
| **[Day03](docs/Day03.md)** | 데이터 분할 고급 & 실제 데이터셋 | 2026-09-02 | train_test_split, Matplotlib 시각화, 공개 데이터셋 활용 |
| **[Day04](docs/Day04.md)** | 회귀 모델 평가 메트릭 & EDA | 2026-09-03 | R2 Score, RMSE, MSE, Pandas EDA, 결측치 분석 |
| **[Day05](docs/Day05.md)** | Kaggle 실제 데이터셋 & 제출 | 2026-09-04 | 대규모 데이터 처리, 결측치 처리 방법, 하이퍼파라미터 튜닝 |
| **[Day06](docs/Day06.md)** | Validation & 과적합 시각화 | 2026-09-07 | verbose, validation_data/split, History 시각화, EarlyStopping |
| **[Day07](docs/Day07.md)** | EarlyStopping 확대 & 이진 분류 | 2026-09-08 | patience/restore_best_weights, sigmoid, binary_crossentropy, metrics, stratify |
| **[Day08](docs/Day08.md)** | 다중 분류 & One-Hot Encoding | 2026-09-09 | softmax, categorical_crossentropy, to_categorical, OneHotEncoder, np.argmax |

---

## 📁 디렉토리 구조

```
C:\furiosa_study\
├── keras/
│   ├── keras01.py ~ keras06_batch.py                           (Day01)
│   ├── keras07_matrix.py ~ keras09_train_test1.py              (Day02)
│   ├── keras09_train_test2.py ~ keras12_R2_RMSE_boston.py      (Day03)
│   ├── keras12_R2_RMSE_01_boston.py ~ keras13_ddareung.py      (Day04)
│   ├── keras13_ddareung01.py ~ keras14_kaggle_bike1.py         (Day05)
│   ├── keras15_verbose.py ~ keras20_EarlyStopping1_...py       (Day06)
│   ├── keras20_EarlyStopping2_diabetes.py ~ keras22_...py      (Day07)
│   ├── keras23_softmax1_OneHot_iris.py ~ keras23_softmax4_...py (Day08)
│   └── ...
├── _data/
│   ├── ddareung/           (Dacon 따릉이 데이터)
│   ├── kaggle_bike/        (Kaggle Bike Sharing 데이터)
│   └── kaggle_santander/   (Kaggle Santander 고객 거래 예측 데이터)
├── docs/                   (Day별 상세 학습 기록)
│   ├── Day01.md
│   ├── Day02.md
│   └── ... Day08.md
├── README.md               (전체 목차)
├── .gitignore
└── .vscode/
```

---

## 📊 학습 진행도 (완료율)

**기간:** 2026-08-31 ~ 2026-12-24 (토요일, 일요일, 법정 공휴일 제외: 추석 3일, 한글날 1일, 대체공휴일 1일)  
**총 수업일:** 80일 (640시간)  
**완료:** 8일 (Day01 ~ Day08)  
**완료 시간:** 64시간  
**완료율: 10.0%**

---

**마지막 업데이트:** 2026-09-09  
**최근 학습:** [Day08 - 다중 분류(Multiclass Classification)와 One-Hot Encoding](docs/Day08.md)
