# [목적]
# learning rate(lr) 가 가중치 갱신에 어떻게 쓰이는지, 가중치 w 하나짜리 모델 y = x * w 로 직접 계산해 본다.
#
# [흐름]
#   - x = 10, 정답 y = 10 → 정답 w 는 1
#   - 초기 w = 0.001, lr = 0.999, epochs = 2
#   1. 현재 w 로 예측 (hypothesis = x * w) 하고 loss = (예측 - y)² 계산 (bias 와 평균(Σ / n) 은 생략한 mse)
#   2. w 를 lr 만큼 올린 경우 (w + lr) 와 내린 경우 (w - lr) 의 loss 를 각각 계산
#   3. loss 가 더 작은 쪽으로 w 를 lr 만큼 이동
#      → 미분 없이 "올려 보고 내려 봐서" 방향을 정하는 방식, lr 이 한 번에 움직이는 크기
#
# [결과]
#   1 epoch : w = 0.001 → 예측 0.01, loss 99.8001
#             올린 w = 1.0 (loss 0) < 내린 w = -0.998 (loss 399.2) → w = 1.0
#   2 epoch : w = 1.0 → 예측 10.0, loss 0.0
#   - lr = 0.999 가 마침 정답까지의 거리 (1 - 0.001) 와 같아서 한 번에 정답에 닿았다
#     lr 이 너무 작으면 여러 번 움직여야 하고, 너무 크면 정답을 지나쳐 왔다 갔다 한다

x = 10
y = 10
w = 0.001
lr = 0.999
epochs = 2

for i, e in enumerate(range(epochs)):       # enumerate : (순번 i, 값 e) 를 함께 꺼낸다 (여기서는 i 만 사용)
    hypothesis = x * w
    loss = (hypothesis - y) ** 2 # mse, bias와 ∑ 신경 안 쓰고
    print(i + 1, "Loss: ", round(loss, 4), "\tPredict: ", round(hypothesis, 4))

    up_predict = x * (w + lr)               # w 를 lr 만큼 올렸을 때
    up_loss = (y - up_predict) ** 2

    down_predict = x * (w - lr)             # w 를 lr 만큼 내렸을 때
    down_loss = (y - down_predict) ** 2

    if (up_loss > down_loss):               # loss 가 더 작은 쪽으로 w 이동
        w = w - lr
    else:
        w = w + lr

# 1 Loss:  99.8001        Predict:  0.01
# 2 Loss:  0.0    Predict:  10.0
