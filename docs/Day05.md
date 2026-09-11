# Day05 - Kaggle 실제 데이터셋 및 대규모 회귀 모델 개발

**학습 기간:** 2026-09-04

---

## 핵심 학습 내용
- 대규모 실제 데이터셋 처리 (Kaggle Bike Sharing Demand: 10,886 샘플, 8개 특성)
- 결측치 처리 방법 비교 (dropna() vs fillna() - 평균값 대체)
- 제출 파일(submission.csv) 생성 및 형식 이해
- 모델 하이퍼파라미터 튜닝 (Dense 레이어 수, epochs, batch_size, activation)
- 회귀 모델 성능 최적화 (r2, MSE, RMSE 지표를 통한 반복 개선)
- 다중 회귀 분석 (multi-feature regression) 실전 경험

---

## 학습 파일

| 파일 | 내용 |
|---|---|
| `keras13_ddareung01.py` | Dacon 따릉이 데이터 활용 (dropna()로 결측치 삭제, r2: 0.7157, RMSE: 46.1) |
| `keras13_ddareung02_submit.py` | 따릉이 데이터 + 결측치 대체 (fillna() 평균값, 제출 파일 생성, r2: 0.6854, RMSE: 43.05) |
| `keras14_kaggle_bike1.py` | Kaggle 자전거 공유 데이터셋 (10,886 훈련 샘플, 6,493 테스트 샘플, r2: 0.3211, RMSE: 147.91) |

---

## 성과
- 대규모 실제 데이터셋으로 회귀 모델 구축 완료
- 결측치 처리 방법 실전 적용
- Kaggle 형식의 submission.csv 생성 및 제출 프로세스 이해
- 모델 성능 튜닝을 통한 반복적 개선 능력 습득

---

## 핵심 개념

```python
# 결측치 처리 방법 비교
train_df = train_df.dropna()                    # 방법1: 결측치 있는 행 삭제
test_df = test_df.fillna(test_df.mean())        # 방법2: 평균값으로 대체
```

### 제출 파일 생성

```python
submission['count'] = model.predict(test_df)
submission.to_csv(path + "submit/" + "submit.csv")
```

### 하이퍼파라미터 튜닝 반복

```python
# Dense 레이어 수: 2~4개 조정
# epochs: 1000~2000 조정
# batch_size: 16~32 조정
# activation: relu vs sigmoid vs none 비교
```


---

## 💡 주요 학습 포인트
1. **결측치 처리 방법 비교**: dropna()는 행 제거, fillna()는 평균값 대체 선택
2. **대규모 데이터셋 처리**: Kaggle 형식 데이터셋 (10,886 훈련, 6,493 테스트 샘플)
3. **제출 파일 생성**: submission.csv 형식 이해 및 생성 방법
4. **하이퍼파라미터 튜닝**: Dense 레이어 수, epochs, batch_size, activation 반복 조정
5. **모델 성능 최적화**: R2, RMSE를 지표로 반복 개선 (트레이드오프 이해)
6. **다중 회귀 분석 실전**: 8개 특성을 입력으로 하는 실제 예측 모델 구축

---

[⬅️ Day04](Day04.md) · [🏠 전체 목차](../README.md) · [Day06 ➡️](Day06.md)
