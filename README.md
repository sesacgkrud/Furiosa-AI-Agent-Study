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
| **[Day15](docs/Day15.md)** | 함수형 전환 & ImageDataGenerator & npy 저장/불러오기 | 2026-09-18 | Input/Model, flow_from_directory, DirectoryIterator, sigmoid, batch_size/OOM, ModelCheckpoint, np.save/np.load |
| **[Day16](docs/Day16.md)** | 내 폴더 이미지 분류 & npy 저장/훈련/예측 분리 & 이미지 증폭 | 2026-09-21 | class_mode categorical, class_indices, train_test_split/stratify, load_img/img_to_array, np.expand_dims, model.save/load_model, datagen.flow |
| **[Day17](docs/Day17.md)** | 이미지 증폭으로 훈련 데이터 늘리기 & 클래스 불균형 맞추기 | 2026-09-22 | np.tile, flow(x, y) 튜플, randint/choice(replace=False), np.concatenate, 증폭 후 스케일링, np.where, validation_split vs validation_data |
| **[Day18](docs/Day18.md)** | learning_rate 직접 지정 & ReduceLROnPlateau & RNN · LSTM 입문 | 2026-09-23 | Adam(learning_rate), ReduceLROnPlateau, factor/patience, es vs rlr, SimpleRNN, (N, timesteps, features), input_length/input_dim, LSTM 파라미터 4배 |
| **[Day19](docs/Day19.md)** | GRU 파라미터 & split 함수로 시계열 자르기 & 범위 밖 예측 개선 | 2026-09-28 | GRU 파라미터 390, split_x, bbb[:, :-1] / bbb[:, -1], (6, 5, 2) 3차원 슬라이싱, input_shape=(4, 2), 스케일링, activation='linear', 수정 전/후 비교 |
| **[Day20](docs/Day20.md)** | split 함수로 여러 값 예측 & return_sequences · Flatten & Jena 기후 시계열 예측 | 2026-09-29 | split_x(size-1) 예측 데이터, reshape(-1, 2) → (N, 5, 2), Dense(2), return_sequences, ndim 에러, RNN + Flatten, Kaggle Jena Climate, 144칸 어긋난 x/y, 2차원 스케일링, float32 npy, Dense(144), RMSE |
| **[Day21](docs/Day21.md)** | Bidirectional RNN & Jena 기온 예측 비교 & LangChain 입문 | 2026-09-30 | Bidirectional, units × 2, 파라미터 × 2, model.add 누락, Bi-LSTM, LSTM vs Bidirectional, ChatOpenAI, invoke, API 키 관리, os.environ, .env / load_dotenv, base_url, PromptTemplate, LCEL, StrOutputParser |
| **[Day22](docs/Day22.md)** | Tokenizer와 One-Hot & 텍스트 감정 분류 (DNN · LSTM · One-Hot LSTM) & Embedding 층 & OpenAI 임베딩 | 2026-10-01 | Tokenizer, fit_on_texts, word_index, texts_to_sequences, pad_sequences, Embedding, OpenAIEmbeddings, embed_query, dimensions |
| **[Day23](docs/Day23.md)** | Reuters · IMDB 텍스트 분류 & sparse_categorical_crossentropy & 표 데이터 LSTM & Reshape 층 | 2026-10-02 | reuters, imdb, num_words, test_split, sparse_categorical_crossentropy, Reshape, target_shape |

---

## 📁 디렉토리 구조

```
C:\furiosa_study\
├─── keras1/                                                          (Day01 ~ Day21 Keras 실습)
│   ├── keras01.py ~ keras06_batch.py                                 (Day01)
│   ├── keras07_matrix.py ~ keras09_train_test1.py                    (Day02)
│   ├── keras09_train_test2.py ~ keras12_R2_RMSE_boston.py            (Day03)
│   ├── keras12_R2_RMSE_01_boston.py ~ keras13_ddareung.py            (Day04)
│   ├── keras13_ddareung01.py ~ keras14_kaggle_bike1.py               (Day05)
│   ├── keras15_verbose.py ~ keras20_EarlyStopping1_...py             (Day06)
│   ├── keras20_EarlyStopping2_diabetes.py ~ keras22_...py            (Day07)
│   ├── keras23_softmax1_OneHot_iris.py ~ keras23_softmax4_...py      (Day08)
│   ├── keras24_kaggle_santander_categorical.py ~ keras27_...py       (Day09)
│   ├── keras28_Scaler01_california.py ~ keras28_Scaler10_...py       (Day09~Day10)
│   ├── keras29_1_save_model.py ~ keras29_4_load_model2.py            (Day10)
│   ├── keras29_5_save_weights.py ~ keras34_hamsu10_digits.py         (Day11)
│   ├── keras35_gpu_test00.py ~ keras36_cnn2_mnist_imshow.py          (Day12)
│   ├── keras36_cnn3_mnist1.py ~ keras39_MaxPolling0.py               (Day13)
│   ├── keras39_MaxPooling1_mnist.py ~ keras42_cnn10_digits.py        (Day14)
│   ├── keras43_hamsu01_mnist.py ~ keras45_04_catdog_load_npy.py      (Day15)
│   ├── keras46_01_save_npy_horse.py ~ keras50_flow1.py               (Day16)
│   ├── keras50_flow2_next.py ~ keras51_augment5_...py                (Day17)
│   ├── keras52_optimizer01_california.py ~ keras55_LSTM1_summary.py  (Day18)
│   ├── keras55_LSTM2_scale.py ~ keras56_split2_samcode.py            (Day19)
│   ├── keras56_split3.py ~ keras58_kaggle_jena2.py                   (Day20)
│   └── keras59_Bidirectional1.py ~ keras59_Bidirectional3_jena.py    (Day21)
├─── keras2/                                                          (Day22 ~ Day23 Keras 실습)
│   ├── keras60_Tokenizer1.py ~ keras61_Embedding04_important.py      (Day22)
│   └── keras62_1_reuters.py ~ keras65_Reshape2.py                    (Day23)
├─── RAG/                                                             (LangChain 실습)
│   ├── rag01_key_insert.py ~ rag09_out_parser02.py                   (Day21)
│   └── rag10_Embedding01.py ~ rag10_Embedding02.py                   (Day22)
├─── _save/                                                           (저장한 모델 / 가중치 - .gitignore 제외)
│   ├── keras29/ ~ keras34/                                           (model.save / save_weights / ModelCheckpoint 저장 파일)
│   ├── keras44/                                                      (cat_dog MCP 저장 파일)
│   ├── keras45/                                                      (npy 실습 MCP 저장 파일)
│   ├── keras46/                                                      (horse-human / rps MCP 저장 파일)
│   ├── keras47/                                                      (npy 로 훈련한 모델 저장 파일)
│   ├── keras51/                                                      (man_woman 증폭 실습 MCP 저장 파일)
│   ├── keras58/                                                      (Jena Climate 시계열 MCP 저장 파일)
│   ├── keras59/                                                      (Jena Climate Bidirectional MCP 저장 파일)
│   ├── keras62/                                                      (Reuters / IMDB MCP · save_weights 저장 파일)
│   └── keras64/                                                      (표 데이터 LSTM MCP 저장 파일)
├─── _data/
│   ├── ddareung/                                                     (Dacon 따릉이 데이터)
│   ├── kaggle_bike/                                                  (Kaggle Bike Sharing 데이터)
│   ├── kaggle_santander/                                             (Kaggle Santander 고객 거래 예측 데이터 - .gitignore 제외)
│   ├── kaggle_jena/                                                  (Kaggle Jena Climate 기상 시계열 데이터 - .gitignore 제외)
│   ├── image/                                                        (brain / cat_dog / horse-human / rps / man_woman 이미지 - .gitignore 제외)
│   └── *_npy/                                                        (이미지를 npy 로 저장한 폴더 - .gitignore 제외)
├─── docs/                                                            (Day별 상세 학습 기록)
│   ├── Day01.md
│   ├── Day02.md
│   └── ... Day23.md
├─── README.md                                                        (전체 목차)
├─── .env                                                             (API 키 - .gitignore 제외)
├─── .gitignore
└─── .vscode/
```

---

## 📊 학습 진행도 (완료율)

**기간:** 2026-08-31 ~ 2026-12-24 (토요일, 일요일, 법정 공휴일 제외: 추석 3일, 한글날 1일, 대체공휴일 1일)  
**총 수업일:** 80일 (640시간)  
**완료:** 23일 (Day01 ~ Day23)  
**완료 시간:** 184시간  
**완료율: 29%**

---

**마지막 업데이트:** 2026-10-02  
**최근 학습:** [Day23 - Reuters · IMDB 텍스트 분류, sparse_categorical_crossentropy, 표 데이터 LSTM, Reshape 층](docs/Day23.md)
