# keras17_val1_california.py
# train 데이터 중 일부를 validation(검증)으로 떼어내서 val_loss 를 확인하는 실습

from sklearn.datasets import fetch_california_housing

# import 가 안 되는 경우
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)   # (20640, 8) (20640,)

##############################################################################
# [설명] train / test / validation 이 각각 뭔지
#
# 데이터는 항상 3덩어리로 생각한다.
#   x_train : 모델이 "공부"하는 데이터           -> model.fit() 에 넣는다
#   x_val   : 공부 도중 "모의고사"를 보는 데이터 -> val_loss 로 찍힌다 (가중치 갱신 X)
#   x_test  : 마지막에 딱 한 번 보는 "수능"      -> model.evaluate() 에 넣는다
#
# 중요: val 은 반드시 x_train 안에서 떼어낸다. x_test 를 val 로 쓰면
#       시험 문제를 미리 보고 공부하는 것과 같아서 성능이 부풀려진다(데이터 누수).
##############################################################################

# ---------------------------------------------------------------------------
# [방법 1] train_test_split 을 2번 써서 x_val 을 직접 만들기
#
#   1단계 : 전체(x, y)  -> x_train(80%) / x_test(20%)
#   2단계 : x_train     -> x_train(80%) / x_val(20%)   <- test 가 아니라 train 을 또 자른다!
#
#   train_test_split(A, B, test_size=0.2) 는
#   "A와 B를 같은 순서로 섞은 뒤, 뒤쪽 20%를 두 번째 반환값으로 준다" 는 뜻이다.
#   반환 순서는 항상 (A_앞, A_뒤, B_앞, B_뒤) 로 고정이라 받는 변수 이름 순서를 지켜야 한다.
#
#   x_train, x_test, y_train, y_test = train_test_split(
#       x, y,
#       test_size=0.2,          # train_size=0.8 과 같은 의미 (둘 중 하나만 써도 됨)
#       random_state=100,       # 고정하면 매번 같은 분할 -> 실험끼리 비교가 가능해진다
#   )
#
#   x_train, x_val, y_train, y_val = train_test_split(
#       x_train, y_train,       # ★ x_test 가 아니라 x_train 을 다시 자른다
#       test_size=0.2,          # 전체 기준으로는 0.8 * 0.2 = 16% 가 val 이 된다
#       random_state=100,
#   )
#   -> 이렇게 만들면 model.fit(..., validation_data=(x_val, y_val)) 로 넘긴다.
#
#   언제 쓰나 : val 데이터를 내가 직접 확인하거나 따로 손대야 할 때,
#              그리고 데이터가 정렬돼 있어서 반드시 섞어서 떼어내야 할 때.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# [방법 2] validation_split 으로 fit 이 알아서 떼어가게 하기  (이 파일에서 사용)
#
#   x_val 변수를 따로 만들지 않고 model.fit 에 validation_split=0.2 만 적으면
#   케라스가 x_train 의 20%를 검증용으로 빼서 쓴다. 코드가 짧아서 편하다.
#
#   ★ 주의점 (모르면 함정에 빠지는 부분)
#   1) 떼어가는 기준은 "x_train 의 맨 뒤에서부터" 이고, 섞지 않고 순서대로 자른다.
#      (fit 의 shuffle=True 는 val 을 떼어낸 "뒤"에 train 부분만 섞는다)
#      -> 원본 데이터가 y값 순으로 정렬돼 있으면 val 이 한쪽으로 치우친다.
#      -> 지금은 위 train_test_split 이 이미 섞어줬으므로 안전하다.
#   2) 비율의 기준은 전체(x)가 아니라 x_train 이다.
#      전체 20640개 -> x_train 16512개 -> 그 중 20%인 3302개가 val 로 빠진다.
#      결국 fit 이 실제 학습에 쓰는 건 13210개.
#   3) validation_data 와 validation_split 을 둘 다 쓰면 validation_data 가 이긴다.
#      (validation_split 은 무시됨) 그래서 아래 fit 에서는 하나만 쓴다.
# ---------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,     # [변경] 원본은 비율 생략(기본값 0.75) -> 의도를 명확히 하려고 명시
    random_state=100,
)

# print(x_train.shape, x_test.shape)   # (16512, 8) (4128, 8)

#################### [참고] 스케일링 ####################
# california 데이터는 컬럼별 값 범위가 크게 다르다.
#   MedInc(소득)은 ~8 정도인데 Population(인구)은 수천 단위
# 이러면 학습이 불안정하고 loss 가 잘 안 떨어진다.
# -> keras28_Scaler01_california.py 에서 MinMaxScaler 로 해결한다.
#########################################################

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))   # [변경] relu 추가
model.add(Dense(75, activation='relu'))                 # [변경] relu 추가
model.add(Dense(50, activation='relu'))                 # [변경] relu 추가
model.add(Dense(25, activation='relu'))                 # [변경] relu 추가
model.add(Dense(1))                                     # 회귀는 출력층에 activation 안 붙인다

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,               # [추가] x_train 의 20%를 검증용으로 사용 -> val_loss 출력
    # validation_data=(x_val, y_val),   # [방법 1] 로 만들었다면 위 줄 대신 이 줄을 쓴다
)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
# [삭제] 원본의 loss = model.evaluate(x, y) 는 훈련에 쓴 x_train 까지 포함해서 평가하는 것이라
#        점수가 실제보다 좋게 나온다(데이터 누수). 평가는 반드시 x_test 로만 한다.
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

# [참고] 훈련 중 출력되는 loss 와 val_loss 를 같이 보면서 과적합을 판단한다.
#   loss ↓ , val_loss ↓  -> 아직 잘 학습되는 중
#   loss ↓ , val_loss ↑  -> 과적합 시작. 여기부터는 훈련할수록 오히려 나빠진다.
#
# epochs=1000 은 너무 많아서 뒤쪽은 과적합 구간일 가능성이 높다.
# 훈련 로그에서 val_loss 가 더 이상 안 떨어지고 올라가기 시작하는 지점을 눈으로 찾아,
# 그 근처 숫자로 epochs 를 줄여서 다시 돌려보면 된다.
