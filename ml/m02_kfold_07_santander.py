# m02_kfold_00_iris.py 베이스
# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

# [목적]
# m02_00 의 KFold + cross_val_score 교차 검증을 Kaggle Santander (고객 거래 여부 이진 분류) 에 적용한다
# - 데이터 : train.csv (200000, 200 컬럼 + target) 를 pd.read_csv 로 불러와 x / y 를 나눈다
# - 모델 : DecisionTreeClassifier → XGBClassifier (m01 에서 사용, 20만 행에서도 빠르다)
#   → random_state 를 줘서 실행할 때마다 같은 점수가 나오게 했다
# - Classifier 라서 cross_val_score 는 fold 별 accuracy 를 반환한다
#   → target 이 0 : 약 90% / 1 : 약 10% 로 치우쳐 있어서, 전부 0 으로 찍어도 acc 가 약 0.9 다
#     acc 가 0.9 보다 얼마나 높은지를 봐야 한다

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier

#1. 데이터
path = "./_data/kaggle_santander/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)    # [200000 rows x 201 columns]

x = train_csv.drop(['target'], axis=1)      # (200000, 200)
y = train_csv['target']                     # (200000,)
print(x.shape, y.shape)
print(y.value_counts())                     # 0 : 179902 / 1 : 20098

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델 구성
model = XGBClassifier(random_state=333)

#3. 훈련, 평가
scores = cross_val_score(model, x, y, cv=kfold)

print("acc : ", scores, "\n평균 acc : ", round(np.mean(scores), 4))
# acc :  [0.91415  0.911175 0.91325  0.91105  0.9145  ]
# 평균 acc :  0.9128
# → 전부 0 으로 찍었을 때의 acc (179902 / 200000 = 0.8995) 보다 약 0.013 높다
