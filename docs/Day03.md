# Day03 - 데이터 분할 고급 및 실제 데이터셋 활용

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

### 💡 주요 학습 포인트
1. **데이터 분할**: Numpy 슬라이싱 vs train_test_split() 비교
2. **random_state**: 난수 고정으로 재현성 확보 (매번 동일한 분할 결과)
3. **shuffle=True**: 데이터를 무작위로 섞어서 공정한 훈련/테스트 분할
4. **Matplotlib 시각화**: scatter plot으로 원본 데이터, plot으로 예측값 표시
5. **공개 데이터셋**: sklearn의 California Housing, Diabetes, Boston Housing 등 활용

---

[⬅️ Day02](Day02.md) · [🏠 전체 목차](../README.md) · [Day04 ➡️](Day04.md)
