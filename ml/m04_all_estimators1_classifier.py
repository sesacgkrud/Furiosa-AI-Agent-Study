# m03_kfold_train_test_split00_iris.py 베이스

# [목적]
# sklearn 에 들어 있는 분류 모델(classifier) 전체를 iris 에 하나씩 돌려 보고 정답률을 비교한다
# - all_estimators(type_filter="classifier") : sklearn 분류 모델의 (이름, 모델 클래스) 튜플 리스트 (sklearn 1.9.1 기준 44 개)
#   → type_filter="regressor" 로 바꾸면 회귀 모델 목록 (55 개)
# - for name, model in all : model 은 아직 클래스라서 model() 로 기본값 모델을 만든 뒤 fit · score
# - 모델 중에는 다른 모델을 인자로 받아야 하거나 음수 입력을 못 받는 것들이 있어 오류가 난다
#   → try / except 로 오류 난 모델은 "모델 오류" 만 찍고 다음 모델로 넘어간다
# - warnings.filterwarnings("ignore") : 수렴 경고 등이 모델마다 쏟아지는 것을 숨긴다
# - 이 파일은 교차 검증 없이 train / test 한 번 나눈 점수(model.score) 로만 비교한다

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings("ignore")

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
all = all_estimators(type_filter="classifier")
print('all models : ', all)
print("sk version : ", sk.__version__)          # sk version :  1.9.1
print("모델의 갯수 : ", len(all))                 # 모델의 갯수(classifier) :  44

# all = all_estimators(type_filter="regressor")
# print('all models : ', all)
# print("sk version : ", sk.__version__)          # sk version :  1.9.1
# print("모델의 갯수 : ", len(all))                 # 모델의 갯수(regressor) :  55

for name, model in all:
    # print(name)
    # print(model)
    try:
        model = model()

        #3. 훈련
        model.fit(x_train, y_train)

        #4. 평가
        acc = model.score(x_test, y_test)
        print(name, "의 정답률 : ", acc)

    except:
        print(name, "모델 오류")
# 결과 (44 개 중 31 개 성공 · 13 개 모델 오류)
# LinearDiscriminantAnalysis 의 정답률 :  0.9666666666666667       ← 가장 높음
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9666666666666667
# GaussianNB · GradientBoosting · LogisticRegression · RandomForest · SVC 등 : 0.9333333333333333
# DecisionTreeClassifier 의 정답률 :  0.9
# BernoulliNB 의 정답률 :  0.7
# DummyClassifier 의 정답률 :  0.3333333333333333                  ← 한 클래스로만 찍는 기준 모델
# → test 30 개라서 1 개 차이가 0.0333 이다. random_state 를 안 준 모델(트리 · MLP 등) 은 실행마다 조금씩 달라진다
#
# 모델 오류 13 개 (try / except 로 건너뜀)
# - CategoricalNB · ComplementNB · MultinomialNB : x 에 음수가 있으면 안 된다 (StandardScaler 를 거쳐 음수가 생김)
# - ClassifierChain · MultiOutputClassifier · OneVsOneClassifier · OneVsRestClassifier · OutputCodeClassifier
#   SelfTrainingClassifier · StackingClassifier · VotingClassifier
#   FixedThresholdClassifier · TunedThresholdClassifierCV : 안에 넣을 모델(estimator) 을 인자로 줘야 한다
