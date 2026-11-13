# Day51 - softmax 직접 구현 · 원그래프 & 경사 하강법(gradient descent) 직접 구현 & learning rate 가중치 갱신 & TensorBoard 콜백 · RNGD eager · overlap 스케줄링 처리량 비교

**학습 기간:** 2026-11-13

> Day50 활성화 함수에 이어 softmax 를 numpy 로 직접 만들고 출력 비율을 원그래프(`plt.pie`) 로 그렸다 (keras78_09).
>
> 2차 함수 f(x) = x² - 4x + 6 의 모양을 그린 뒤 (keras78_gradient1), 미분한 기울기로 x 를 갱신하는 경사 하강법을 직접 구현해 최솟값 (2, 2) 로 수렴하는 과정을 출력했다 (keras78_gradient2).
>
> 가중치 하나짜리 모델 y = x * w 에서 w 를 lr 만큼 올리고 내려 loss 가 작은 쪽으로 움직이는 방식으로 learning rate 의 역할을 확인했다 (keras79_lr).
>
> keras36_cnn3_mnist1 CNN 에 TensorBoard 콜백을 붙여 훈련 로그를 저장했다 (keras80_Tensorboard).
>
> RNGD 에서는 furiosa-llm 서버(Qwen3-8B-FP8) 를 기본 eager 모드와 `--enable-overlap-scheduling` 을 켠 overlap 모드로 각각 띄우고, Day49 의 순차 요청(f25) · 동시 요청 수별 비교(f28) 코드로 전체 시간 · RPS · Output TPS 를 비교했다 (f30_overlap_eager1 · 2).

---

## 🎯 한눈에 보기

| 파일 | 내용 | 결과 |
|---|---|---|
| keras78_09_softmax | softmax([1, 2, 3, 4]) 원그래프 | [0.0321, 0.0871, 0.2369, 0.6439], 합 1.0 |
| keras78_gradient1 | f(x) = x² - 4x + 6 그래프 · 최솟값 점 | 최솟값 (2, 2) |
| keras78_gradient2 | 경사 하강법 x = x - lr · f'(x), 50 epoch | x : -10 → 1.99983 / f(x) : 146 → 2.00000 |
| keras79_lr | w 를 ± lr 해 보고 loss 작은 쪽으로 이동 | 1 epoch loss 99.8001 → 2 epoch loss 0.0 |
| keras80_Tensorboard | mnist CNN + EarlyStopping + TensorBoard (GPU) | acc 0.9899 / 137.94 초 |
| f30_overlap_eager1 | 질문 8 개 순차 요청, eager vs overlap | TPS 91.458 vs 91.095 (차이 없음) |
| f30_overlap_eager2 | 질문 16 개 동시 요청 (workers 4), eager vs overlap | TPS 327.2 vs 332.76 (약 1.7 % ↑) |

---

## 📖 핵심 학습 내용

### 1. softmax 직접 구현 · 원그래프 (keras78_09)

```python
x = np.arange(1, 5)     # [1 2 3 4]

softmax = lambda x : np.exp(x) / np.sum(np.exp(x))
y = softmax(x)          # [0.0321 0.0871 0.2369 0.6439], 합 1.0

ratio = y
labels = y

plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.show()
```

- softmax(x_i) = e^(x_i) / Σ e^(x_j) : 모든 출력이 0 ~ 1, 합이 1 → 확률 분포
- 값이 클수록 비율이 지수적으로 커진다 (4 → 64%)
- 다중 분류 출력층 `activation='softmax'` + `categorical_crossentropy`, 예측은 `np.argmax`
- `plt.pie(x, explode=None, labels=None, ...)` : 두 번째 위치 인자는 explode(조각 띄우기) 라서 이름표는 `labels=labels` 로 이름을 붙여 넘긴다
- `shadow=True` 그림자, `startangle=90` 12 시 방향에서 시작

### 2. 최솟값을 찾을 2차 함수 그리기 (keras78_gradient1)

```python
x = np.linspace(-1, 6, 100)
print(x, len(x))        # [-1. -0.92929293 ... 6.] 100

f = lambda x : x**2 - 4*x + 6
y = f(x)

plt.plot(x, y, "k-")
plt.plot(2, 2, "sk", color="red")   # 최솟값 지점 (2, 2)
```

- `np.linspace(시작, 끝, 개수)` : 끝을 포함해 같은 간격으로 개수만큼 (`np.arange` 는 간격을 정하고 끝 미포함)
- f(x) = (x - 2)² + 2 → x = 2 에서 최솟값 2
- `"k-"` 검은 실선, `"sk"` 검은 네모 점 (color="red" 가 색을 덮어씀)

### 3. 경사 하강법 직접 구현 (keras78_gradient2)

```python
f = lambda x : x**2 - 4*x + 6
gradient = lambda x : 2*x - 4       # f'(x)

x = -10.0 # 초깃값
epochs = 50
learning_rate = 0.1

for i in range(epochs):
    x = x - learning_rate * gradient(x)
    print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(i+1, x, f(x)))
```

| epoch | x | f(x) |
|---|---|---|
| 00 | -10.00000 | 146.00000 |
| 01 | -7.60000 | 94.16000 |
| 05 | -1.93216 | 17.46188 |
| 10 | 0.71151 | 3.66021 |
| 20 | 1.86165 | 2.01914 |
| 30 | 1.98514 | 2.00022 |
| 50 | 1.99983 | 2.00000 |

- 갱신식 `x = x - learning_rate * f'(x)` : 기울기가 음수면 x 증가, 양수면 x 감소 → 항상 최솟값 쪽으로
- 최솟값에 가까울수록 기울기가 작아져 이동 폭도 줄어든다 (처음엔 2.4 씩, 나중엔 0.0001 씩)
- 신경망에서는 x = 가중치, f(x) = loss, learning_rate = 한 번에 움직이는 비율
- `"{:02d}"` 두 자리 정수, `"{:6.5f}"` 소수 5 자리 실수 포맷

### 4. learning rate 로 가중치 갱신 (keras79_lr)

```python
x = 10
y = 10
w = 0.001
lr = 0.999
epochs = 2

for i, e in enumerate(range(epochs)):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2 # mse, bias와 ∑ 신경 안 쓰고

    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2

    if (up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr

# 1 Loss:  99.8001        Predict:  0.01
# 2 Loss:  0.0    Predict:  10.0
```

- w 를 lr 만큼 올린 경우 · 내린 경우의 loss 를 비교해 더 작은 쪽으로 이동 (미분 없이 방향 결정)
- 1 epoch : w + lr = 1.0 (loss 0) < w - lr = -0.998 (loss 399.2) → w = 1.0
- lr = 0.999 가 정답까지 거리(1 - 0.001) 와 같아 2 epoch 에 loss 0
- lr 이 작으면 여러 번 움직여야 하고, 크면 정답을 지나쳐 왔다 갔다 한다

### 5. TensorBoard 콜백 (keras80_Tensorboard)

```python
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(monitor="val_loss", mode="min", patience=50,
                   verbose=1, restore_best_weights=True)

tb = TensorBoard(
    log_dir = "C:/furiosa_study/_save/tensorboard/_keras80/",
    histogram_freq=0,
    write_graph=True,
    write_images=True,
)

model.fit(x_train, y_train, epochs=50, batch_size=128,
          verbose=1, validation_split=0.2,
          callbacks=[es, tb])
```

- keras36_cnn3_mnist1 (mnist Conv2D 6 층 + Dense) 베이스에 EarlyStopping · TensorBoard 콜백 추가
- `log_dir` 아래에 `train` / `validation` 로그 폴더가 생긴다 (`_save/` 는 .gitignore)
- `histogram_freq=0` 가중치 히스토그램 기록 안 함, `write_graph=True` 모델 구조 기록, `write_images=True` 가중치를 이미지로 기록
- patience=50 = epochs=50 이라 이 설정에서는 중간에 멈추지 않는다

| 실행 | loss | acc | accuracy_score | 소요 시간 |
|---|---|---|---|---|
| keras36_cnn3_mnist1 GPU | 0.0448 | 0.9897 | 0.9897 | 141.31 초 |
| keras80 GPU (TensorBoard 적용) | 0.0394 | 0.9899 | 0.9899 | 137.94 초 |

### 6. RNGD eager 모드 vs overlap 스케줄링 (f30_overlap_eager1 · 2)

```bash
# eager 모드 (기본, --enable-overlap-scheduling 기본값 False)
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:0 --host 0.0.0.0 --port 8000

# overlap 모드
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:0 --host 0.0.0.0 --port 8000 --enable-overlap-scheduling
```

- overlap scheduling : 지금 배치를 NPU 가 계산하는 동안 CPU 가 다음 배치를 미리 스케줄링해, CPU 스케줄링 시간을 NPU 계산 뒤에 숨긴다
- `furiosa-llm serve --help` 설명 : 전체 처리량 · TPOT 개선, 대신 TTFT 가 조금 늘 수 있다
- 클라이언트 코드는 그대로 두고 서버 옵션만 바꿔 같은 코드를 두 번 실행해 비교

```python
# f30_overlap_eager1 : f25 순차 요청 + 출력 토큰 합계
for prompt in prompts:                       # 질문 8 개, max_tokens=100
    response = llm.invoke(prompt)
    usage = response.usage_metadata or {}
    tokens = usage.get("output_tokens")
    total_tokens += tokens

print("RPS: ", round(len(prompts) / elapsed, 3))
print("TPS: ", round(total_tokens / elapsed, 3))
```

| f30_overlap_eager1 (순차 8 개) | 출력 토큰 | 전체 시간 | RPS | TPS |
|---|---|---|---|---|
| eager | 800 | 8.747 초 | 0.915 | 91.458 |
| overlap | 800 | 8.782 초 | 0.911 | 91.095 |

```python
# f30_overlap_eager2 : f28 동시 요청 수별 비교 그대로 (질문 16 개)
for workers in [1, 2, 3, 4]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(ask, prompts))   # (지연, 토큰) 16 개
```

| f30_overlap_eager2 (workers 4, 16 개) | 전체 시간 | 평균 지연 | req/s | Output TPS |
|---|---|---|---|---|
| eager | 4.89 초 | 1.22 초 | 3.27 | 327.2 |
| overlap | 4.81 초 | 1.2 초 | 3.33 | 332.76 |

- 순차 요청은 배치에 요청이 1 개뿐이라 미리 스케줄링할 다음 배치가 거의 없다 → 두 모드 차이 0.4 % 안쪽
- 동시 요청은 continuous batching 으로 배치가 계속 이어져 overlap 효과가 보인다 → 전체 시간 · 평균 지연 · 처리량 모두 조금 개선 (TPS 약 1.7 %)

---

## 📂 학습 파일

| 파일 | 내용 |
|---|---|
| `keras2/keras78_09_softmax.py` | softmax 를 def · lambda 로 만들고 [1, 2, 3, 4] 출력 비율을 `plt.pie` 원그래프로 |
| `keras2/keras78_gradient1.py` | `np.linspace` 로 f(x) = x² - 4x + 6 그래프, 최솟값 (2, 2) 점 표시 |
| `keras2/keras78_gradient2.py` | 기울기 f'(x) = 2x - 4 로 경사 하강법 50 epoch, x · f(x) 수렴 출력 |
| `keras2/keras79_lr.py` | y = x * w 에서 w ± lr 의 loss 비교로 가중치 갱신 (lr = 0.999) |
| `keras2/keras80_Tensorboard.py` | mnist CNN 에 EarlyStopping · TensorBoard 콜백 적용 |
| `furiosa02_rngd/f30_overlap_eager1.py` | 질문 8 개 순차 요청으로 eager · overlap 모드 전체 시간 · RPS · TPS 비교 (f25 베이스) |
| `furiosa02_rngd/f30_overlap_eager2.py` | 질문 16 개 동시 요청 수 1 ~ 4 로 eager · overlap 모드 비교 (f28 베이스) |
| `docs/Day51.md` | Day51 학습 기록 신규 작성 |
| `docs/Day50.md` | 하단 nav에 Day51 링크 추가 (수정) |
| `README.md` | 학습 일지(퓨리오사 NPU 학습 내용 · 키워드 `/ (퓨리오사 NPU)` 구분) / 디렉토리 구조(keras2/ · furiosa02_rngd/ Day51, _save/tensorboard/) / 진행도(51일, 64%) 갱신 (수정) |

---

## 💻 핵심 개념

### softmax

```
softmax(x_i) = e^(x_i) / Σ e^(x_j)   : 0 ~ 1, 합 1 (확률 분포)
[1, 2, 3, 4] → [0.032, 0.087, 0.237, 0.644]
```

### 경사 하강법

```
x = x - learning_rate * f'(x)
f'(x) < 0 → x 증가 / f'(x) > 0 → x 감소 / f'(x) = 0 → 최솟값 (멈춤)
가중치 = x, loss = f(x), learning_rate = 이동 비율
```

### np.arange vs np.linspace

```
np.arange(-5, 5, 0.1)     : 간격 0.1 지정, 끝(5) 미포함 → 100 개
np.linspace(-1, 6, 100)   : 개수 100 지정, 끝(6) 포함 → 간격 약 0.0707
```

### TensorBoard 인자

```
log_dir         : 로그 저장 폴더 (train / validation 하위 폴더 생성)
histogram_freq  : 가중치 히스토그램 기록 주기 (0 = 안 함)
write_graph     : 모델 구조 기록
write_images    : 가중치를 이미지로 기록
callbacks=[es, tb]
```

### eager vs overlap scheduling

```
eager   : 배치 계산(NPU) → 다음 배치 스케줄링(CPU) → 배치 계산(NPU) ... 차례대로
overlap : 배치 계산(NPU) 중에 다음 배치 스케줄링(CPU) 을 겹쳐 실행 → CPU 시간이 가려진다
켜는 법 : furiosa-llm serve ... --enable-overlap-scheduling (기본값 False)
효과    : 동시 요청(배치가 이어질 때) 처리량 ↑, 순차 요청은 차이 없음
```

---

## 성과

- softmax 를 직접 구현하고 출력 비율을 원그래프로 확인했다
- 2차 함수의 최솟값을 기울기로 찾아가는 경사 하강법을 직접 구현해 50 epoch 만에 (2, 2) 로 수렴하는 것을 확인했다
- w 를 ± lr 해 보는 방식으로 learning rate 가 가중치 이동 크기라는 것을 계산으로 확인했다
- mnist CNN 에 TensorBoard 콜백을 적용해 훈련 로그를 저장했다 (acc 0.9899)
- RNGD 서버를 eager · overlap 모드로 띄워 비교해, overlap scheduling 은 순차 요청에서는 차이가 없고 동시 요청에서 처리량이 조금 오르는 것을 확인했다

---

## 💡 주요 학습 포인트

1. **softmax**: e^x 를 전체 합으로 나눠 합 1 인 확률로 만든다
2. **plt.pie**: 두 번째 위치 인자는 explode, 이름표는 `labels=` 로
3. **np.linspace**: 개수 지정 · 끝 포함 (arange 는 간격 지정 · 끝 미포함)
4. **경사 하강법**: x = x - lr · f'(x), 기울기 반대 방향으로 이동해 최솟값에 수렴
5. **learning rate**: 한 번에 움직이는 크기 — 작으면 느리고 크면 지나친다
6. **TensorBoard**: callbacks 에 넣어 훈련 로그를 log_dir 에 저장
7. **--enable-overlap-scheduling**: NPU 계산 중에 CPU 가 다음 배치를 미리 스케줄링 (기본은 eager)
8. **overlap 효과는 동시 요청에서**: 배치가 이어질 때만 숨길 CPU 시간이 생긴다

---

[⬅️ Day50](Day50.md) · [🏠 전체 목차](../README.md)
