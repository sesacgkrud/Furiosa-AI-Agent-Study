# Day05 - Kaggle 실제 데이터셋 및 대규모 회귀 모델 개발

**학습 기간:** 2026-09-04

> 대회 데이터로 **제출 파일(submission.csv)** 을 만드는 전체 흐름을 익히고, 결측치 처리 2가지 방법을 비교했다.

---

## 🎯 한눈에 보기

| 주제 | 핵심 한 줄 |
|---|---|
| 결측치 처리 2가지 | `dropna()` = 행 삭제 / `fillna(평균)` = 평균값으로 채우기 |
| 제출용 test.csv | 정답이 없어서 `evaluate` 를 못 쓴다 -> `predict` 결과를 저장 |
| 제출 데이터는 지우면 안 된다 | 행을 지우면 제출 개수가 안 맞아서 `fillna` 로 채운다 |
| x 만들 때 주의 | 정답을 알려주는 컬럼(casual, registered)까지 `drop` |
| 하이퍼파라미터 튜닝 | 층 수 / epochs / batch_size / activation 을 바꿔가며 r2·RMSE 비교 |
| 기록 습관 | 시도할 때마다 조건과 결과를 파일 하단에 주석으로 남긴다 |

---

## 📖 핵심 학습 내용

### 결측치 처리 2가지 방법
- **`dropna()`** - 빈 칸이 있는 행을 통째로 지운다 (훈련 데이터에 사용)
- **`fillna(df.mean())`** - 컬럼 평균으로 채운다 (**제출용 test 데이터에 사용**)
- 제출용 데이터는 행을 지우면 제출 개수(715개)가 안 맞으므로 지우지 않고 채워야 한다

### 제출 파일(submission.csv) 만들기
- 제출용 `test.csv` 에는 정답(count) 컬럼이 없다 -> `evaluate` 를 쓸 수 없다
- `model.predict(test_csv)` 결과를 `submission` 의 정답 칸에 넣고 `to_csv()` 로 저장
- 파일명은 실습마다 다르게 준다 (재사용하면 이전 결과가 덮어써진다)

### x, y 분리할 때 주의할 점
- 캐글 자전거 데이터는 `casual + registered = count` 라서 이 둘을 x에 남기면 정답을 그대로 알려주는 셈
- 게다가 제출용 test.csv에는 이 두 컬럼이 아예 없어서 컬럼 수(8개)가 안 맞는다 -> 3개를 같이 `drop`

### 하이퍼파라미터 튜닝
- Dense 층 수(2 ~ 4개), epochs(1000 ~ 2000), batch_size(16 ~ 32), activation(relu 유무)을 바꿔가며 비교
- 시도할 때마다 **조건과 결과를 파일 하단에 주석으로 기록**해서 무엇이 좋아졌는지 추적

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras13_ddareung01.py` | Dacon 따릉이 데이터 (`dropna()` 로 결측치 삭제, r2 0.7157 / RMSE 46.1) |
| `keras13_ddareung02_submit.py` | 따릉이 + 결측치 대체 (`fillna()` 평균값) + 제출 파일 생성 (r2 0.6854 / RMSE 43.05) |
| `keras14_kaggle_bike1.py` | Kaggle 자전거 공유 데이터 (10,886 훈련 / 6,493 테스트, r2 0.3211 / RMSE 147.91) |

---

## 💻 핵심 개념

### 결측치 처리 방법 비교

```python
train_df = train_df.dropna()               # 방법1: 결측치 있는 행 삭제 (훈련 데이터)
test_df = test_df.fillna(test_df.mean())   # 방법2: 평균값으로 대체 (제출용 데이터)
# 제출용은 행을 지우면 제출 개수가 안 맞으므로 반드시 채워서 쓴다
```

### x, y 분리 - 정답을 알려주는 컬럼은 빼기

```python
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)  # axis=1 : 컬럼 방향
y = train_csv['count']
# casual + registered = count 이므로 남겨두면 정답을 알려주는 셈이 된다
# 제출용 test.csv 에도 두 컬럼이 없어서 컬럼 수를 맞추려면 반드시 지워야 한다
```

### 제출 파일 생성

```python
submission['count'] = model.predict(test_df)
submission.to_csv(path + "submit/" + "submit_0904_1142.csv")
```

### 하이퍼파라미터 튜닝 기록

```python
# 1차 시도 (random_state=77, Dense=64,32,16,1, epochs=2000, batch=16)  r2 : 0.2606
# 2차 시도 (random_state=77, Dense=64,32,1,    epochs=1000, batch=32)  r2 : 0.3423
# 3차 시도 (random_state=100, Dense=32,16,1,   epochs=1000, batch=16)  r2 : 0.3145
```

---

## 성과
- 대규모 실제 데이터셋으로 회귀 모델 구축 완료
- 결측치 처리 방법 실전 적용
- Kaggle 형식의 submission.csv 생성 및 제출 프로세스 이해
- 모델 성능 튜닝을 통한 반복적 개선 능력 습득

---

## 💡 주요 학습 포인트
1. **결측치 처리 방법 비교**: `dropna()` 는 행 제거, `fillna()` 는 평균값 대체
2. **대규모 데이터셋 처리**: Kaggle 형식 데이터셋 (10,886 훈련 / 6,493 테스트)
3. **제출 파일 생성**: submission.csv 형식 이해 및 생성 방법
4. **하이퍼파라미터 튜닝**: Dense 층 수, epochs, batch_size, activation 반복 조정
5. **모델 성능 최적화**: r2, RMSE를 지표로 반복 개선
6. **다중 회귀 분석 실전**: 8개 특성을 입력으로 하는 실제 예측 모델 구축

---

[⬅️ Day04](Day04.md) · [🏠 전체 목차](../README.md) · [Day06 ➡️](Day06.md)
