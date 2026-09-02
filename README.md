# Furiosa AI Agent - 학습 진행 현황

## 📚 Day01 - Keras 기초 및 Deep Learning 구조
**학습 기간:** 2026-08-31

### 핵심 학습 내용
- TensorFlow/Keras Sequential 모델 기초
- Dense Layer를 이용한 신경망 구성
- MSE 손실함수 및 Adam 최적화 알고리즘
- Deep Learning 구조(다층 네트워크)로 모델 성능 향상
- Batch Processing을 통한 훈련 방식
- 모델 컴파일, 훈련, 평가, 예측의 4가지 단계

### 학습 파일
- `keras01.py`: 기본 신경망 구성 (1-1-1 구조)
- `keras02.py`: 선형 회귀 모델 (1-1-1 구조)
- `keras03.py`: epochs 조정을 통한 손실값 개선 (900 epochs)
- `keras04_deep1.py`: 4층 Deep Learning (500-300-1 구조)
- `keras05_deep2.py`: 3층 최적화 네트워크 (1000-300-1 구조)
- `keras06_batch.py`: Batch Processing 개념 이해

### 성과
- 기본 신경망 구성 완료
- 손실값 최적화 (0.3238)
- Deep Learning 개념 이해

---

## 📚 Day02 - 다중 입력/출력 신경망 및 데이터 분할
**학습 기간:** 2026-09-01

### 핵심 학습 내용
- Numpy 다차원 배열(Matrix, Tensor) 이해 및 shape 개념
- 다중 입력(Multi-Input) 신경망 구성
- 데이터 전치(Transpose) - `x.T` 및 `transpose()` 메서드
- 다중 출력(Multi-Output) 신경망 구성
- Train/Test 데이터 분할 (검증 개념 입문)
- MLP(Multi-Layer Perceptron) 고급 구조

### 학습 파일
- `keras07_matrix.py`: Numpy 다차원 배열 기초 (1D, 2D, 3D, 5D 데이터 구조)
- `keras08_mlp1_1.py`: 다중 입력 신경망 (2입력-1출력, shape: (5,2))
- `keras08_mlp1_2.py`: 데이터 전치 적용 (2입력-1출력, transpose 사용)
- `keras08_mlp2_1.py`: 3입력-1출력 MLP (10->5->2->1 구조, loss: 5.54e-05)
- `keras08_mlp2_2.py`: 3입력-2출력 MLP (20->10->5->2 구조)
- `keras08_mlp3_1.py`: 1입력-3출력 MLP (10->5->3 구조)
- `keras08_mlp4.py`: 완성된 MLP (정확도 높음)
- `keras09_train_test1.py`: Train/Test 분할 (검증 방식 학습)

### 성과
- 다중 입력/출력 신경망 완성
- 다차원 데이터 처리 능력 확보
- Train/Test 분할 개념 이해
- MLP 고급 구조 마스터

### 핵심 개념
```python
# 데이터 전치
x.T  # numpy array 전치
x.transpose()  # 메서드를 이용한 전치

# 다중 입력/출력
Dense(n_output, input_dim=n_input)  # n_input 개의 입력, n_output 개의 출력

# Train/Test 분할
x_train, x_test = x[:7], x[7:]
y_train, y_test = y[:7], y[7:]
model.fit(x_train, y_train, ...)  # 훈련
model.evaluate(x_test, y_test)    # 검증
```

---

## 📚 Day03 - 데이터 분할 고급 및 실제 데이터셋 활용
**학습 기간:** 2026-09-02

### 핵심 학습 내용
- Numpy 슬라이싱을 이용한 Train/Test 분할 (수동 방식)
- Scikit-learn의 `train_test_split()` 함수 활용
- `random_state` 파라미터를 통한 난수 재현성 확보
- `train_size`, `test_size` 비율 설정 (7:3, 8:2 등)
- `shuffle` 파라미터를 통한 데이터 섞기
- Matplotlib을 이용한 데이터 시각화 (scatter plot, line plot)
- 실제 머신러닝 데이터셋 활용 (California Housing, Diabetes, Boston Housing)
- 회귀 모델(Regression Model) 성능 평가

### 학습 파일
- `keras09_train_test2.py`: Numpy 슬라이싱을 이용한 7:3 분할
- `keras09_train_test3.py`: Scikit-learn train_test_split() 기본 활용
- `keras10_scatter1.py`: train_test_split() + matplotlib 산점도 그리기
- `keras10_scatter2.py`: 예측 결과를 그래프로 시각화
- `keras11_1_california.py`: California Housing 데이터셋 활용 (8개 특성, 20640 샘플)
- `keras11_2_diabetes.py`: Diabetes 데이터셋 활용 (10개 특성, 442 샘플)
- `keras11_3_boston_tf.py`: Boston Housing 데이터셋 활용 (13개 특성, 506 샘플)
- `keras12_R2_RMSE_boston.py`: Boston Housing으로 회귀 모델 평가

### 성과
- 실제 데이터셋으로 모델 훈련 완료
- 데이터 분할 방법 마스터 (Numpy vs Scikit-learn)
- 모델 예측 결과 시각화 능력 획득

### 핵심 개념
```python
# Numpy 슬라이싱
x_train, x_test = x[:7], x[7:]

# Scikit-learn train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,      # 70% 훈련 데이터
    test_size=0.3,       # 30% 테스트 데이터
    shuffle=True,        # 데이터 섞기
    random_state=1234    # 난수 고정 (재현성)
)

# Matplotlib 시각화
plt.scatter(x, y)              # 원본 데이터 점찍기
plt.plot(x, result, color='red')  # 예측 결과 선그래프
plt.show()
```

---

## 🎯 학습 진행도

| 단계 | 학습 내용 | 완료 | 주요 내용 |
|------|---------|------|---------|
| Day01 | Keras 기초 | ✅ | Sequential, Dense, Batch Processing, Deep Learning |
| Day02 | 다중 입력/출력 | ✅ | Matrix, Transpose, Multi-Input/Output, Train/Test 분할 |
| Day03 | 데이터 분할 고급 & 실제 데이터셋 | ✅ | train_test_split, Matplotlib 시각화, 공개 데이터셋 활용 |

---

## 📁 디렉토리 구조

```
C:\furiosa_study\
├── keras/
│   ├── keras01.py ~ keras06_batch.py              (Day01)
│   ├── keras07_matrix.py ~ keras09_train_test1.py (Day02)
│   ├── keras09_train_test2.py ~ keras12_R2_RMSE_boston.py (Day03)
│   └── ...
├── README.md
└── .vscode/
```

---

## 💡 주요 학습 포인트

### Day01
1. **모델 구성**: Sequential() → add(Dense()) → compile() → fit()
2. **손실함수**: MSE (Mean Squared Error)
3. **최적화**: Adam optimizer
4. **성능 개선**: 레이어 깊이 증가, epochs 조정, 뉴런 수 증가

### Day02
1. **데이터 형태**: (sample, features) 형태 유지
2. **Transpose**: 데이터를 올바른 형태로 변환
3. **다중 출력**: 최종 Dense layer 출력값 = 원하는 출력 개수
4. **Train/Test**: 전체 데이터를 훈련/검증 데이터로 분할

### Day03
1. **데이터 분할**: Numpy 슬라이싱 vs train_test_split() 비교
2. **random_state**: 난수 고정으로 재현성 확보 (매번 동일한 분할 결과)
3. **shuffle=True**: 데이터를 무작위로 섞어서 공정한 훈련/테스트 분할
4. **Matplotlib 시각화**: scatter plot으로 원본 데이터, plot으로 예측값 표시
5. **공개 데이터셋**: sklearn의 California Housing, Diabetes, Boston Housing 등 활용

---

**마지막 업데이트:** 2026-09-02  
**총 학습 시간:** Day01, Day02, Day03 완료
