# m01_feature_importances03_gain.py 베이스

# [목적]
# m01_03 에서 숫자로 본 중요도를 xgboost 의 plot_importance 로 막대 그래프로 그린다
# - plot_importance(model, importance_type='weight') : F score (분기에 쓰인 횟수) 그래프
# - plot_importance(model, importance_type='gain')   : Gain score (손실함수 개선 정도) 그래프
# - 막대 끝에 값이 표시되고, 값이 큰 컬럼이 위로 정렬된다

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=random_state,
    stratify=y,
)

#2. 모델 구성
model = XGBClassifier(random_state=random_state)

#3. 컴파일, 훈련
model.fit(x_train, y_train)
print('====================', model.__class__.__name__, '====================')

#4. 평가, 예측
score = model.score(x_test, y_test)
print('acc: ', score)
print('feature_importances_: ', model.feature_importances_)

gain_score = model.get_booster().get_score(importance_type='gain')
print('Gain score: ', gain_score)

import matplotlib.pyplot as plt
from xgboost import plot_importance

# F-Score Graph
# plot_importance(model, importance_type='weight')
# plt.title('F-Score')
# plt.show()
# → f2 145 / f3 83 / f1 66 / f0 63

# Gain-Score Graph
plot_importance(model, importance_type='gain')
plt.title('Gain-Score')
plt.show()
# → f2 2.8167... / f3 0.8144... / f1 0.0943... / f0 0.0928... (get_score(importance_type='gain') 값이 그대로 표시된다)
