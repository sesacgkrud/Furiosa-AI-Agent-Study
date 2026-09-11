# Day01 - Keras 기초 및 Deep Learning 구조

**학습 기간:** 2026-08-31

---

## 핵심 학습 내용
- TensorFlow/Keras Sequential 모델 기초
- Dense Layer를 이용한 신경망 구성
- MSE 손실함수 및 Adam 최적화 알고리즘
- Deep Learning 구조(다층 네트워크)로 모델 성능 향상
- Batch Processing을 통한 훈련 방식
- 모델 컴파일, 훈련, 평가, 예측의 4가지 단계

---

## 학습 파일

| 파일 | 내용 |
|---|---|
| `keras01.py` | 기본 신경망 구성 (1-1-1 구조) |
| `keras02.py` | 선형 회귀 모델 (1-1-1 구조) |
| `keras03.py` | epochs 조정을 통한 손실값 개선 (900 epochs) |
| `keras04_deep1.py` | 4층 Deep Learning (500-300-1 구조) |
| `keras05_deep2.py` | 3층 최적화 네트워크 (1000-300-1 구조) |
| `keras06_batch.py` | Batch Processing 개념 이해 |

---

## 성과
- 기본 신경망 구성 완료
- 손실값 최적화 (0.3238)
- Deep Learning 개념 이해

---

## 💡 주요 학습 포인트
1. **모델 구성**: Sequential() → add(Dense()) → compile() → fit()
2. **손실함수**: MSE (Mean Squared Error)
3. **최적화**: Adam optimizer
4. **성능 개선**: 레이어 깊이 증가, epochs 조정, 뉴런 수 증가

---

[🏠 전체 목차](../README.md) · [Day02 ➡️](Day02.md)
