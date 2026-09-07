# https://dacon.io/competitions/open/235576/overview/description
# [방법 1] train_test_split 을 2번 써서 x_val 을 직접 만들고 validation_data 로 넘기는 방식

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = "./_data/ddareung/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission = pd.read_csv(path + "submission.csv", index_col=0)

########## 결측치 처리 1. 삭제 ##########

train_csv = train_csv.dropna()

# ★★★ train_csv를 x와 y로 분리 ★★★
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

##############################################################################
# [설명] train / test / validation 이 각각 뭔지
#
#   x_train : 모델이 "공부"하는 데이터           -> model.fit() 에 넣는다
#   x_val   : 공부 도중 "모의고사"를 보는 데이터 -> val_loss 로 찍힌다 (가중치 갱신 X)
#   x_test  : 마지막에 딱 한 번 보는 "수능"      -> model.evaluate() 에 넣는다
#
# 중요: val 은 반드시 x_train 안에서 떼어낸다. x_test 를 val 로 쓰면
#       시험 문제를 미리 보고 공부하는 것과 같아서 성능이 부풀려진다(데이터 누수).
#
# 헷갈리기 쉬운 점: 이 파일의 test_csv(대회 제출용 파일)는 위의 x_test 와 완전히 다르다.
#   x_test    : train.csv 를 잘라서 만든 것. 정답(y_test)이 있어서 점수를 매길 수 있다.
#   test_csv  : 정답이 없는 대회 제출용 데이터. predict 만 할 수 있고 evaluate 는 못 한다.
##############################################################################

##############################################################################
# [방법 1] train_test_split 을 2번 써서 x_val 을 직접 만들기
#
#   1단계 : 전체(x, y)  -> x_train / x_test
#   2단계 : x_train     -> x_train / x_val    ★ x_test 가 아니라 x_train 을 또 자른다
#
#   train_test_split(A, B, test_size=0.2) 의 의미
#     "A 와 B 를 같은 순서로 섞은 뒤, 뒤쪽 20% 를 두 번째 반환값으로 준다"
#   반환 순서는 항상 (A_앞, A_뒤, B_앞, B_뒤) 로 고정이라 받는 변수 이름 순서를 지켜야 한다.
#   함수는 내가 붙인 변수 이름이 뭔지 모른다. 두 번째 자리에 x_val 이라고 적으면
#   그게 val 이 되는 것뿐이다.
#
#   test_size 와 train_size 는 둘 중 하나만 쓰면 된다 (0.2 = 20%).
#   random_state 를 고정하면 매번 같은 위치에서 잘려서 실험끼리 비교가 가능해진다.
#
#   [방법 2] 인 validation_split=0.2 와의 차이
#     - 방법 1 은 train_test_split 이 "섞은 뒤" 잘라주고, x_val 을 내가 직접 들고 있어서
#       따로 확인하거나 손댈 수 있다.
#     - 방법 2 는 x_train 의 "맨 뒤에서부터 섞지 않고" 순서대로 잘라간다. 코드는 짧지만
#       데이터가 정렬돼 있으면 val 이 한쪽으로 치우친다.
#     - 두 개를 같이 쓰면 validation_data 가 이기고 validation_split 은 무시된다.
##############################################################################

# 1단계 : 전체 -> train(80%) / test(20%)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,     # [변경] 원본은 비율 생략(기본값 0.75) -> 의도를 명확히 하려고 명시
    random_state=222,
)

# 2단계 : train -> train(80%) / val(20%)
x_train, x_val, y_train, y_val = train_test_split(   # [추가] x_val, y_val 을 여기서 만든다
    x_train, y_train,   # ★ x_test 가 아니라 위에서 만든 x_train 을 다시 자른다
    train_size=0.8,     # 전체 기준으로는 0.8 * 0.2 = 16% 가 val 이 된다
    random_state=222,
)

# print(x_train.shape, x_val.shape, x_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=16,
    validation_data=(x_val, y_val),   # [추가] 위에서 만든 val 세트를 넘긴다 -> val_loss 출력
    # validation_split=0.2,           # [방법 2] 는 x_val 없이 이 줄 하나로 끝낸다
)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
# 평가는 훈련에도 검증에도 쓰지 않은 x_test 로만 한다.
loss = model.evaluate(x_test, y_test, )

print("loss(mse) :", loss)
y_predict = model.predict(x_test).ravel()

# [삭제] 여기에 r2_score, mean_squared_error 를 다시 import 하는 줄이 있었는데
#        맨 위에서 이미 import 했으므로 중복이라 지웠다.

r2 = r2_score(y_test, y_predict)

print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# [참고] 훈련 중 출력되는 loss 와 val_loss 를 같이 보면서 과적합을 판단한다.
#   loss ↓ , val_loss ↓  -> 아직 잘 학습되는 중
#   loss ↓ , val_loss ↑  -> 과적합 시작. 여기부터는 훈련할수록 오히려 나빠진다.
#
# 결측치를 지우고 나면 데이터가 1300개 정도밖에 안 되므로 epochs=1000 은 과적합되기 쉽다.
# 훈련 로그에서 val_loss 가 더 이상 안 떨어지고 올라가기 시작하는 지점을 눈으로 찾아,
# 그 근처 숫자로 epochs 를 줄여서 다시 돌려보면 된다.
