# Day02 - 다중 입력/출력 신경망 및 데이터 분할

**학습 기간:** 2026-09-01

---

## 핵심 학습 내용
- Numpy 다차원 배열(Matrix, Tensor) 이해 및 shape 개념
- 다중 입력(Multi-Input) 신경망 구성
- 데이터 전치(Transpose) - `x.T` 및 `transpose()` 메서드
- 다중 출력(Multi-Output) 신경망 구성
- Train/Test 데이터 분할 (검증 개념 입문)
- MLP(Multi-Layer Perceptron) 고급 구조

---

## 학습 파일

| 파일 | 내용 |
|---|---|
| `keras07_matrix.py` | Numpy 다차원 배열 기초 (1D, 2D, 3D, 5D 데이터 구조) |
| `keras08_mlp1_1.py` | 다중 입력 신경망 (2입력-1출력, shape: (5,2)) |
| `keras08_mlp1_2.py` | 데이터 전치 적용 (2입력-1출력, transpose 사용) |
| `keras08_mlp2_1.py` | 3입력-1출력 MLP (10->5->2->1 구조, loss: 5.54e-05) |
| `keras08_mlp2_2.py` | 3입력-2출력 MLP (20->10->5->2 구조) |
| `keras08_mlp3_1.py` | 1입력-3출력 MLP (10->5->3 구조) |
| `keras08_mlp4.py` | 완성된 MLP (정확도 높음) |
| `keras09_train_test1.py` | Train/Test 분할 (검증 방식 학습) |

---

## 성과
- 다중 입력/출력 신경망 완성
- 다차원 데이터 처리 능력 확보
- Train/Test 분할 개념 이해
- MLP 고급 구조 마스터

---

## 핵심 개념

```python
# 데이터 전치
x.T  # numpy array 전치
x.transpose()  # 메서드를 이용한 전치
```

### 다중 입력/출력

```python
Dense(n_output, input_dim=n_input)  # n_input 개의 입력, n_output 개의 출력
```

### Train/Test 분할

```python
x_train, x_test = x[:7], x[7:]
y_train, y_test = y[:7], y[7:]
model.fit(x_train, y_train, ...)  # 훈련
model.evaluate(x_test, y_test)    # 검증
```


---

## 💡 주요 학습 포인트
1. **데이터 형태**: (sample, features) 형태 유지
2. **Transpose**: 데이터를 올바른 형태로 변환
3. **다중 출력**: 최종 Dense layer 출력값 = 원하는 출력 개수
4. **Train/Test**: 전체 데이터를 훈련/검증 데이터로 분할

---

[⬅️ Day01](Day01.md) · [🏠 전체 목차](../README.md) · [Day03 ➡️](Day03.md)
