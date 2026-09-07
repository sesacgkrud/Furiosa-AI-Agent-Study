# keras17_val3_boston.py
# [방법 2] validation_split 으로 fit 이 x_train 에서 알아서 val 을 떼어가게 하는 방식

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

#1. 데이터
# 이 데이터는 train_test_split 을 쓸 필요가 없다.
# load_data() 가 이미 train / test 로 나눠서 (404개 / 102개) 돌려주기 때문.
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)

##############################################################################
# [설명] train / test / validation 이 각각 뭔지
#
#   x_train : 모델이 "공부"하는 데이터           -> model.fit() 에 넣는다
#   x_val   : 공부 도중 "모의고사"를 보는 데이터 -> val_loss 로 찍힌다 (가중치 갱신 X)
#   x_test  : 마지막에 딱 한 번 보는 "수능"      -> model.evaluate() 에 넣는다
#
# 중요: val 은 반드시 x_train 안에서 떼어낸다. x_test 를 val 로 쓰면
#       시험 문제를 미리 보고 공부하는 것과 같아서 성능이 부풀려진다(데이터 누수).
##############################################################################

##############################################################################
# [방법 2] validation_split 으로 fit 이 알아서 떼어가게 하기
#
#   x_val 이라는 변수를 아예 만들지 않는다.
#   model.fit(..., validation_split=0.2) 한 줄만 적으면
#   케라스가 x_train 의 20% 를 검증용으로 빼서 쓰고 val_loss 를 찍어준다.
#
#   이 파일처럼 데이터가 이미 train / test 로 나뉘어 있을 때 특히 편하다.
#   (자를 게 x_train 하나뿐이라 train_test_split 을 또 부를 이유가 없다)
#
#   ★ 주의점 (모르면 함정에 빠지는 부분)
#   1) 비율의 기준은 전체가 아니라 x_train 이다.
#      x_train 404개 -> 그 중 20% 인 80개가 val 로 빠지고,
#      실제로 학습에 쓰는 건 324개가 된다. (x_test 102개는 건드리지 않는다)
#   2) 떼어가는 기준은 "x_train 의 맨 뒤에서부터" 이고, 섞지 않고 순서대로 자른다.
#      (fit 의 shuffle=True 는 val 을 떼어낸 "뒤"에 train 부분만 섞는다)
#      -> 데이터가 y값 순으로 정렬돼 있으면 val 이 한쪽으로 치우쳐서 val_loss 를 믿을 수 없다.
#      -> boston_housing.load_data() 는 내부에서 이미 섞어서 돌려주므로 여기서는 안전하다.
#         만약 정렬된 데이터라면 방법 1(train_test_split)로 섞어서 잘라야 한다.
#   3) validation_data 와 validation_split 을 둘 다 쓰면 validation_data 가 이긴다.
#      (validation_split 은 무시됨) 그래서 둘 중 하나만 쓴다.
#
#   [참고] 이 파일을 방법 1 로 바꾸려면 x_train 을 한 번 더 자르면 된다.
#     from sklearn.model_selection import train_test_split
#     x_train, x_val, y_train, y_val = train_test_split(
#         x_train, y_train,   # ★ x_test 가 아니라 x_train 을 자른다
#         train_size=0.8,
#         random_state=42,
#     )
#     -> 그리고 fit 에 validation_data=(x_val, y_val) 로 넘긴다.
##############################################################################

#################### [참고] 스케일링 ####################
# boston 데이터는 컬럼별 값 범위가 크게 다르다.
#   NOX(농도)는 0.5 정도인데 TAX(세율)는 700 단위
# 이러면 학습이 불안정하고 loss 가 잘 안 떨어진다. 나중에 배울 StandardScaler 를 쓰면
# 성능이 크게 좋아진다. (아직 진도 전이라 주석으로만 남김)
#########################################################

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=13, activation='relu'))  # [변경] relu 추가
model.add(Dense(50, activation='relu'))                 # [변경] relu 추가
model.add(Dense(25, activation='relu'))                 # [변경] relu 추가
model.add(Dense(1))                                     # 회귀는 출력층에 activation 안 붙인다

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
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

# [참고] 훈련 중 출력되는 loss 와 val_loss 를 같이 보면서 과적합을 판단한다.
#   loss ↓ , val_loss ↓  -> 아직 잘 학습되는 중
#   loss ↓ , val_loss ↑  -> 과적합 시작. 여기부터는 훈련할수록 오히려 나빠진다.
#
# boston 은 학습에 쓰는 게 324개뿐인 작은 데이터라 epochs=1000 이면 과적합되기 쉽다.
# 훈련 로그에서 val_loss 가 더 이상 안 떨어지고 올라가기 시작하는 지점을 눈으로 찾아,
# 그 근처 숫자로 epochs 를 줄여서 다시 돌려보면 된다.
