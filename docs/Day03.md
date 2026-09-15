# Day03 - 데이터 분할 고급 및 실제 데이터셋 활용

**학습 기간:** 2026-09-02

> `train_test_split` 으로 데이터를 섞어 나누고, sklearn이 제공하는 실제 데이터셋으로 회귀 모델을 만들었다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| `train_test_split` | 데이터를 **섞은 뒤** 비율대로 나눠 준다 (수동 슬라이싱보다 안전) |
| `random_state` | 섞는 방식을 고정하는 번호 - 같은 번호면 항상 같게 나뉜다 |
| `train_size` / `test_size` | 둘의 합이 1.0을 넘으면 에러. 하나만 써도 된다 (기본 0.75) |
| `shuffle` | 기본값 `True` - 데이터를 섞어서 나눈다 |
| Matplotlib | `scatter` = 실제 데이터 점 / `plot` = 모델 예측선 |
| 실제 데이터셋 | `datasets.data` = x, `datasets.target` = y |

---

## 📖 핵심 학습 내용

### 데이터 분할 2가지 방법
- **Numpy 슬라이싱** - `x[:7]`, `x[7:]` 로 직접 자른다. 순서대로 잘리므로 **섞이지 않는다**
- **`train_test_split()`** - 섞은 뒤 비율대로 나눠 준다. 앞으로 계속 쓰는 방식

### train_test_split 주요 파라미터
- `train_size=0.7` / `test_size=0.3` - 둘의 합이 1.0을 넘으면 `ValueError`
- `shuffle=True` - 기본값. 데이터를 무작위로 섞는다
- `random_state=1234` - 섞는 방식을 고정하는 번호. 같은 번호면 몇 번을 돌려도 같게 나뉜다
  - 고정하지 않으면 실행할 때마다 분할이 달라져서 결과 비교가 안 된다

### Matplotlib 시각화
- `plt.scatter(x, y)` - 실제 데이터를 점으로 찍는다
- `plt.plot(x, result, color='red')` - 모델이 예측한 값을 선으로 그린다
- 선이 점들 사이를 지나가면 잘 학습된 것

### sklearn 공개 데이터셋
- `datasets.data` = x(입력), `datasets.target` = y(정답)
- x의 **열 개수가 곧 `input_dim`**
- California Housing(20640×8), Diabetes(442×10), Boston Housing(506×13)
- 케라스 내장 `boston_housing.load_data()` 는 처음부터 train/test로 나뉘어 들어온다

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras09_train_test2.py` | Numpy 슬라이싱을 이용한 7:3 분할 |
| `keras09_train_test3.py` | Scikit-learn `train_test_split()` 기본 활용 |
| `keras10_scatter1.py` | `train_test_split()` + matplotlib 산점도 그리기 |
| `keras10_scatter2.py` | 데이터 20개로 늘려 예측 결과를 그래프로 시각화 |
| `keras11_1_california.py` | California Housing 데이터셋 활용 (8개 특성, 20640 샘플) |
| `keras11_2_diabetes.py` | Diabetes 데이터셋 활용 (10개 특성, 442 샘플) |
| `keras11_3_boston_tf.py` | Boston Housing 데이터셋 활용 (13개 특성, 506 샘플) |
| `keras12_R2_RMSE_boston.py` | Boston Housing으로 회귀 모델 평가 |

---

## 💻 핵심 개념

### Numpy 슬라이싱

```python
x_train, x_test = x[:7], x[7:]
```

### Scikit-learn train_test_split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,      # 70% 훈련 데이터
    test_size=0.3,       # 30% 테스트 데이터
    shuffle=True,        # 데이터 섞기
    random_state=1234    # 난수 고정 (재현성)
)
```

### Matplotlib 시각화

```python
plt.scatter(x, y)                 # 원본 데이터 점찍기
plt.plot(x, result, color='red')  # 예측 결과 선그래프
plt.show()
```

---

## 성과
- 실제 데이터셋으로 모델 훈련 완료
- 데이터 분할 방법 마스터 (Numpy vs Scikit-learn)
- 모델 예측 결과 시각화 능력 획득

---

## 💡 주요 학습 포인트
1. **데이터 분할**: Numpy 슬라이싱 vs `train_test_split()` 비교
2. **random_state**: 난수 고정으로 재현성 확보 (매번 동일한 분할 결과)
3. **shuffle=True**: 데이터를 무작위로 섞어서 공정한 훈련/테스트 분할
4. **Matplotlib 시각화**: scatter로 원본 데이터, plot으로 예측값 표시
5. **공개 데이터셋**: sklearn의 California Housing, Diabetes, Boston Housing 등 활용

---

[⬅️ Day02](Day02.md) · [🏠 전체 목차](../README.md) · [Day04 ➡️](Day04.md)
