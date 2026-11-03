# m04_all_estimators1_classifier.py 베이스

# [목적]
# m04_1 의 all_estimators (sklearn 분류 모델 전체 돌리기) 에 m03 의 KFold 교차 검증을 합친다
# - m04 : 모델마다 fit 한 번 → model.score(x_test, y_test) 한 번 (test 를 어떻게 나눴느냐에 점수가 좌우된다)
# - m05 : 모델마다 cross_val_score(model, x_train, y_train, cv=kfold) 로 5 번 검증한 평균 acc 를 본다
#         → 이어서 cross_val_predict 로 x_test 를 예측하고 accuracy_score 로 test acc 도 확인한다 (m03 과 같은 흐름)
# - for 문 안에서 모델 클래스(model) 를 model() 로 만들어 cross_val_score 에 넘긴다
#   → 오류 나는 모델은 try / except 로 건너뛴다

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings("ignore")

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)     # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델 구성
all = all_estimators(type_filter="classifier")
print("sk version : ", sk.__version__)
print("모델의 갯수 : ", len(all))

for name, model in all:
    try:
        model = model()

        #3. 훈련, 평가 (교차 검증)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)     # fold 별 accuracy 5 개

        #4. 예측
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_predict)

        print("==============", name, "==============")
        print("acc : ", scores, "\n평균 acc : ", round(np.mean(scores), 4))
        print("cross_val_predict acc : ", round(acc, 4))

    except:
        print(name, "모델 오류")

# 결과 (44 개 중 30 개 성공 · 14 개 모델 오류, 평균 acc 순)
# LinearDiscriminantAnalysis      평균 acc : 0.975    cross_val_predict acc : 0.9     ← 교차 검증 1 등
# GradientBoostingClassifier      평균 acc : 0.9667   cross_val_predict acc : 0.9
# LogisticRegressionCV            평균 acc : 0.9667   cross_val_predict acc : 0.9
# RandomForestClassifier          평균 acc : 0.9667   cross_val_predict acc : 0.9
# BaggingClassifier               평균 acc : 0.9583   cross_val_predict acc : 0.9333
# ExtraTreesClassifier            평균 acc : 0.9583   cross_val_predict acc : 0.9333
# HistGradientBoostingClassifier  평균 acc : 0.9417   cross_val_predict acc : 0.1667
# DummyClassifier                 평균 acc : 0.225    cross_val_predict acc : 0.1667  ← 가장 많은 클래스 하나로만 찍는 기준 모델
# → cross_val_predict 는 x_test 30 개를 다시 5 조각으로 나눠 24 개로 훈련 · 6 개 예측하므로 평균 acc 보다 낮다
# → HistGradientBoosting 은 잎 하나에 최소 20 개(min_samples_leaf=20) 가 필요해서, 24 개로는 트리를 나누지 못하고 한 클래스로만 찍는다
# → random_state 를 안 준 모델(트리 · MLP 등) 은 실행마다 조금씩 달라진다
#
# 모델 오류 14 개 (try / except 로 건너뜀)
# - CategoricalNB · ComplementNB · MultinomialNB : x 에 음수가 있으면 안 된다 (StandardScaler 를 거쳐 음수가 생김)
# - ClassifierChain · MultiOutputClassifier · OneVsOneClassifier · OneVsRestClassifier · OutputCodeClassifier
#   SelfTrainingClassifier · StackingClassifier · VotingClassifier
#   FixedThresholdClassifier · TunedThresholdClassifierCV : 안에 넣을 모델(estimator) 을 인자로 줘야 한다
# - RadiusNeighborsClassifier : 반경(radius=1.0) 안에 이웃이 없는 데이터가 있어 오류
