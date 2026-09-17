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
| **[Day09](docs/Day09.md)** | 데이터 스케일링 & 모델 구조 확인 | 2026-09-10 | MinMaxScaler, fit/transform, 데이터 누수, model.summary, input_shape |
| **[Day10](docs/Day10.md)** | 스케일러 4종 비교 & 모델 저장/불러오기 | 2026-09-11 | StandardScaler, MaxAbsScaler, RobustScaler, model.save, load_model |
| **[Day11](docs/Day11.md)** | 가중치 저장 & ModelCheckpoint & Dropout & 함수형 모델 | 2026-09-14 | save_weights/load_weights, ModelCheckpoint, save_best_only, Dropout, Input/Model, 시드 고정 |
| **[Day12](docs/Day12.md)** | CPU/GPU 속도 비교 & CNN 입문 | 2026-09-15 | list_physical_devices, 소요 시간 비교, Conv2D, mnist, 이미지 스케일링 |
| **[Day13](docs/Day13.md)** | CNN 이미지 분류 실습 & padding · strides · MaxPooling | 2026-09-16 | Flatten, 4차원 reshape, fashion_mnist, cifar10/100, padding, strides, MaxPooling2D, val_acc |
| **[Day14](docs/Day14.md)** | MaxPooling · GAP 적용 & DNN vs CNN & 정형 데이터 Conv2D | 2026-09-17 | MaxPooling2D, GlobalAveragePooling2D, DNN reshape, BatchNormalization, 정형 데이터 4차원 reshape, (2,1) 커널 |

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
│   ├── keras24_kaggle_santander_categorical.py ~ keras27_...py  (Day09)
│   ├── keras28_Scaler01_california.py ~ keras28_Scaler10_...py  (Day09~Day10)
│   ├── keras29_1_save_model.py ~ keras29_4_load_model2.py       (Day10)
│   ├── keras29_5_save_weights.py ~ keras34_hamsu10_digits.py    (Day11)
│   ├── keras35_gpu_test00.py ~ keras36_cnn2_mnist_imshow.py     (Day12)
│   ├── keras36_cnn3_mnist1.py ~ keras39_MaxPolling0.py          (Day13)
│   └── keras39_MaxPooling1_mnist.py ~ keras42_cnn10_digits.py   (Day14)
├── _save/
│   ├── keras29/            (model.save / save_weights 로 저장한 파일)
│   ├── keras30/            (ModelCheckpoint 로 저장한 모델 파일)
│   ├── keras31/            (MCP 저장 모델 + keras32 가 불러와서 다시 저장한 모델)
│   ├── keras33/            (Dropout 적용 모델의 MCP 저장 파일)
│   └── keras34/            (함수형 모델의 MCP 저장 파일 / keras35 GPU 테스트 / keras42 Conv2D 모델도 여기에 저장)
├── _data/
│   ├── ddareung/           (Dacon 따릉이 데이터)
│   ├── kaggle_bike/        (Kaggle Bike Sharing 데이터)
│   └── kaggle_santander/   (Kaggle Santander 고객 거래 예측 데이터)
├── docs/                   (Day별 상세 학습 기록)
│   ├── Day01.md
│   ├── Day02.md
│   └── ... Day14.md
├── README.md               (전체 목차)
├── .gitignore
└── .vscode/
```

---

## 📊 학습 진행도 (완료율)

**기간:** 2026-08-31 ~ 2026-12-24 (토요일, 일요일, 법정 공휴일 제외: 추석 3일, 한글날 1일, 대체공휴일 1일)  
**총 수업일:** 80일 (640시간)  
**완료:** 14일 (Day01 ~ Day14)  
**완료 시간:** 112시간  
**완료율: 17.5%**

---

**마지막 업데이트:** 2026-09-17  
**최근 학습:** [Day14 - MaxPooling · GlobalAveragePooling 적용, DNN vs CNN 비교, 정형 데이터에 Conv2D 적용](docs/Day14.md)
