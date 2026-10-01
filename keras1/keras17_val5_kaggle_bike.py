# keras17_val5_kaggle_bike.py

# https://www.kaggle.com/competitions/bike-sharing-demand/data
# [방법 2] validation_split 으로 fit 이 x_train 에서 알아서 val 을 떼어가게 하는 방식

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = "./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv) # [10886 rows x 11 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [6493 rows x 8 columns]

submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

#################### x, y 분리 ####################

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# print(x) # [10886 rows x 8 columns]

y = train_csv['count']
print(y, y.shape) # Name: count, Length: 10886, dtype: int64 (10886,)

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
#   test_csv  : 정답(count)이 없는 대회 제출용 데이터. predict 만 하고 evaluate 는 못 한다.
##############################################################################

##############################################################################
# [방법 2] validation_split 으로 fit 이 알아서 떼어가게 하기
#
#   x_val 이라는 변수를 아예 만들지 않는다.
#   model.fit(..., validation_split=0.2) 한 줄만 적으면
#   케라스가 x_train 의 20% 를 검증용으로 빼서 쓰고 val_loss 를 찍어준다.
#
#   ★ 주의점 (모르면 함정에 빠지는 부분)
#   1) 비율의 기준은 전체(x)가 아니라 x_train 이다.
#      전체 10886개 -> x_train 8708개 -> 그 중 20% 인 1741개가 val 로 빠진다.
#      결국 fit 이 실제 학습에 쓰는 건 6967개.
#   2) 떼어가는 기준은 "x_train 의 맨 뒤에서부터" 이고, 섞지 않고 순서대로 자른다.
#      (fit 의 shuffle=True 는 val 을 떼어낸 "뒤"에 train 부분만 섞는다)
#      -> 이 데이터가 특히 위험한 케이스다. train.csv 는 datetime 순으로 정렬돼 있어서
#         그냥 잘랐다면 val 이 "뒤쪽 날짜"만 몰려서 들어갔을 것이다.
#      -> 다행히 아래 train_test_split 이 이미 한 번 섞어줬기 때문에 안전하다.
#         만약 train_test_split 없이 순서대로 잘랐다면 방법 1 을 써야 한다.
#   3) validation_data 와 validation_split 을 둘 다 쓰면 validation_data 가 이긴다.
#      (validation_split 은 무시됨) 그래서 둘 중 하나만 쓴다.
#
#   [참고] 이 파일을 방법 1 로 바꾸려면 x_train 을 한 번 더 자르면 된다.
#     x_train, x_val, y_train, y_val = train_test_split(
#         x_train, y_train,   # ★ x_test 가 아니라 x_train 을 자른다
#         train_size=0.8,
#         random_state=100,
#     )
#     -> 그리고 fit 에 validation_data=(x_val, y_val) 로 넘긴다.
##############################################################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,     # [변경] 원본은 비율 생략(기본값 0.75) -> 의도를 명확히 하려고 명시
    random_state=100,   # shuffle 은 기본값 True -> 여기서 날짜 순서가 섞인다
)

# print(x_train.shape, x_test.shape)   # (8708, 8) (2178, 8)

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu')) # 통상적으로 마지막에는 relu 넣지 않음
# 여기서 relu 를 쓴 이유는 count(대여 수)가 음수가 될 수 없어서다.
# 다만 relu 는 음수를 전부 0으로 만들기 때문에, 학습이 잘못 흘러가면 예측이 전부 0으로
# 굳어버리고(gradient 도 0이라) 회복이 안 될 수 있다. loss 가 안 떨어지고 r2 가 0 근처에서
# 멈춘다면 이 줄을 model.add(Dense(1)) 로 바꿔서 비교해보면 된다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=16,
    validation_split=0.2,             # [추가] x_train 의 20%를 검증용으로 사용 -> val_loss 출력
    # validation_data=(x_val, y_val), # [방법 1] 로 x_val 을 만들었다면 위 줄 대신 이 줄을 쓴다
)

print("========== ========== ========== ========== ==========")

#4. 평가 예측
# 평가는 훈련에도 검증에도 쓰지 않은 x_test 로만 한다.
loss = model.evaluate(x_test, y_test, )

print("loss(mse) :", loss)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)     # [변경] print 를 감싸고 있던 불필요한 괄호 제거

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

# [참고] 훈련 중 출력되는 loss 와 val_loss 를 같이 보면서 과적합을 판단한다.
#   loss ↓ , val_loss ↓  -> 아직 잘 학습되는 중
#   loss ↓ , val_loss ↑  -> 과적합 시작. 여기부터는 훈련할수록 오히려 나빠진다.
#
# epochs=1000 은 너무 많아서 뒤쪽은 과적합 구간일 가능성이 높다.
# 훈련 로그에서 val_loss 가 더 이상 안 떨어지고 올라가기 시작하는 지점을 눈으로 찾아,
# 그 근처 숫자로 epochs 를 줄여서 다시 돌려보면 된다.

##### submission.csv 만들기 // count 컬럼에 값 넣기 #####
# y_submit = model.predict(test_csv)

# submission['count'] = y_submit
# submission.to_csv(path + "submit/" + "submit_0904_1620.csv")
